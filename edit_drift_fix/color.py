"""Colour drift correction: fit an edit -> source colour mapping on the trusted background.

All inputs are linear RGB (1,3,H,W), already aligned. The fitted model is applied to the
whole edited image (including the edit zone), so pasted content matches the source grade.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .util import _gen, diag_px, dilate, gaussian_blur, lightness, quantile, sobel_mag

_MAX_SAMPLES = 200_000
_NBINS = 8                 # luminance strata for sampling
_CLIP_LO, _CLIP_HI = -1.0, 0.98   # blacks are kept: they are what pins the bias down
_GAIN_LIM = (0.6, 1.6)
_OFFDIAG_LIM = 0.35
_BIAS_LIM = 0.03
_LOCAL_LIM = (0.8, 1.25)


def apply_color_model(edt_lin: torch.Tensor, matrix, bias) -> torch.Tensor:
    m = torch.as_tensor(matrix, dtype=edt_lin.dtype, device=edt_lin.device).reshape(3, 3)
    b = torch.as_tensor(bias, dtype=edt_lin.dtype, device=edt_lin.device).view(1, 3, 1, 1)
    return torch.einsum("ij,bjhw->bihw", m, edt_lin) + b


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------
def _sample(src, edt, bg_weight):
    """Stratified pixel pairs on the trusted background -> (S (N,3), E (N,3), parity (N,) bool)."""
    h, w = src.shape[-2:]
    trusted = bg_weight[0, 0] > 0.5
    valid = trusted.clone()
    for x in (src, edt):
        valid &= ((x[0] > _CLIP_LO) & (x[0] < _CLIP_HI)).all(0)
    mag = torch.maximum(sobel_mag(lightness(src)), sobel_mag(lightness(edt)))
    if int(valid.sum()) > 0:
        strong = dilate((mag > quantile(mag[0, 0][valid], 0.85)).float(), 1)[0, 0] > 0.5
        no_edge = valid & ~strong
        if int(no_edge.sum()) >= 1000:  # fully textured image: keep edges rather than nothing
            valid = no_edge
    idx = valid.flatten().nonzero().squeeze(1)
    n = idx.numel()
    if n == 0:
        return None
    g = _gen(src.device)
    lum = lightness(src).flatten()[idx]
    bins = (lum * _NBINS).long().clamp(0, _NBINS - 1)
    # random order inside each bin, bins contiguous
    order = torch.argsort(bins.to(src.dtype) + torch.rand(n, generator=g, device=src.device))
    counts = torch.bincount(bins, minlength=_NBINS).tolist()
    # largest per-bin cap k with sum(min(c, k)) <= _MAX_SAMPLES (equal representation of tones)
    lo, hi = 0, max(counts)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(c, mid) for c in counts) <= _MAX_SAMPLES:
            lo = mid
        else:
            hi = mid - 1
    picks, start = [], 0
    for c in counts:
        picks.append(order[start:start + min(c, lo)])
        start += c
    sel = idx[torch.cat(picks)]
    S = src[0].reshape(3, -1)[:, sel].T
    E = edt[0].reshape(3, -1)[:, sel].T
    bs = max(16, round(min(h, w) / 8))
    parity = ((sel // w) // bs + (sel % w) // bs) % 2 == 0
    return S, E, parity


# ----------------------------------------------------------------------------
# robust fitting
# ----------------------------------------------------------------------------
def _features(E, diagonal):
    n = E.shape[0]
    one = torch.ones(n, 3, 1, dtype=E.dtype, device=E.device)
    if diagonal:
        return torch.cat([E.unsqueeze(2), one], 2)                           # (N,3,2)
    return torch.cat([E.unsqueeze(1).expand(n, 3, 3), one], 2)                 # (N,3,4)


def _beta0(diagonal, ref):
    if diagonal:
        b = torch.tensor([[1.0, 0.0]] * 3)
    else:
        b = torch.cat([torch.eye(3), torch.zeros(3, 1)], 1)
    return b.to(ref)


def _irls(X, Y, beta0, iters=6, ridge=3e-4):
    """Per-output-channel Huber IRLS with a small ridge toward beta0. X (N,3,k), Y (N,3)."""
    dt = X.dtype
    XX = X.unsqueeze(3) * X.unsqueeze(2)       # (N,3,k,k); elementwise sums beat a long-K GEMM on GPU
    XY = X * Y.unsqueeze(2)                    # (N,3,k)
    b0 = beta0.double()
    eye = torch.eye(X.shape[2], dtype=torch.float64, device=X.device)
    wts = torch.ones_like(Y)
    beta = beta0
    for _ in range(iters):
        lam = (ridge * wts.sum(0, dtype=torch.float64)).view(3, 1, 1)
        A = (XX * wts[:, :, None, None]).sum(0, dtype=torch.float64) + lam * eye
        r = (XY * wts[:, :, None]).sum(0, dtype=torch.float64) + lam[:, :, 0] * b0
        beta = torch.linalg.solve(A, r).to(dt)
        res = Y - (X * beta).sum(2)
        scale = 1.4826 * (res - res.median(0).values).abs().median(0).values
        delta = (1.345 * scale).clamp_min(1e-4)
        wts = (delta / res.abs().clamp_min(1e-12)).clamp(max=1.0)
    return beta


def _to_model(beta, diagonal):
    if diagonal:
        g = beta[:, 0].clamp(*_GAIN_LIM)
        return torch.diag(g), beta[:, 1].clamp(-_BIAS_LIM, _BIAS_LIM)
    return beta[:, :3], beta[:, 3]


def _pw(S):
    """Perceptual row weights: an error in linear light matters ~ d(lightness)/dY = Y^(-2/3).

    Without this a tiny linear bias that is invisible in highlights tints every shadow."""
    y = S @ torch.tensor([0.2126, 0.7152, 0.0722]).to(S)
    p = (y.clamp_min(0) + 4e-3) ** (-2.0 / 3.0)
    return (p / p.mean()).unsqueeze(1)


def _fit(S, E, diagonal):
    p = _pw(S)
    return _to_model(_irls(_features(E, diagonal) * p.unsqueeze(2), S * p, _beta0(diagonal, S)), diagonal)


def _err(S, E, M, b):
    """Robust mean abs error (lowest 90% of per-pixel L1 errors)."""
    e = ((E @ M.T + b - S) * _pw(S)).abs().sum(1)
    e = e[e <= quantile(e, 0.9)]
    return float(e.mean()) if e.numel() else 0.0


def _holdout(S, E, parity, diagonal):
    folds = [(parity, ~parity), (~parity, parity)]
    if min(int(parity.sum()), int((~parity).sum())) < 500:
        folds = [(torch.ones_like(parity), torch.ones_like(parity))]
    errs = []
    for tr, te in folds:
        if diagonal is None:  # identity
            M, b = torch.eye(3).to(S), torch.zeros(3).to(S)
        else:
            M, b = _fit(S[tr], E[tr], diagonal)
        errs.append(_err(S[te], E[te], M, b))
    return sum(errs) / len(errs)


def _within_limits(M, b):
    d = torch.diagonal(M)
    off = M - torch.diag(d)
    return bool((d >= _GAIN_LIM[0]).all() and (d <= _GAIN_LIM[1]).all()
                and off.abs().max() <= _OFFDIAG_LIM and b.abs().max() <= _BIAS_LIM)


# ----------------------------------------------------------------------------
# local low-frequency gain field
# ----------------------------------------------------------------------------
def _push_pull(v, c):
    """Fill v where confidence c is low with a coarse-to-fine average of confident neighbours."""
    h, w = v.shape[-2:]
    if h <= 2 or w <= 2:
        tot = c.sum()
        if float(tot) < 1e-6:
            fill = torch.ones_like(v[:, :, :1, :1])
        else:
            fill = (v * c).sum((2, 3), keepdim=True) / tot
        return c * v + (1 - c) * fill
    size = ((h + 1) // 2, (w + 1) // 2)
    cs = F.interpolate(c, size=size, mode="area")
    vs = F.interpolate(v * c, size=size, mode="area") / cs.clamp_min(1e-8)
    vs = torch.where(cs > 1e-8, vs, torch.ones_like(vs))
    coarse = _push_pull(vs, (4 * cs).clamp(max=1.0))
    return c * v + (1 - c) * F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)


def _local_gain(src, cor, weight, sigma):
    h, w = src.shape[-2:]
    f = max(1, int(sigma / 4))  # work at a resolution where sigma ~ 4 px
    size = (max(1, round(h / f)), max(1, round(w / f)))
    sg = sigma / f
    ws = F.interpolate(weight, size=size, mode="area")
    num_s = gaussian_blur(F.interpolate(src * weight, size=size, mode="area"), sg)
    num_c = gaussian_blur(F.interpolate(cor * weight, size=size, mode="area"), sg)
    cov = gaussian_blur(ws, sg)
    gain = (num_s / num_c.clamp_min(1e-8)).clamp(*_LOCAL_LIM)
    t = (cov / 0.25).clamp(0, 1)
    conf = t * t * (3 - 2 * t)
    conf = conf * ((num_c / cov.clamp_min(1e-8)) > 5e-3).to(conf)  # ratio of near-black is noise
    gain = torch.where(conf > 0, gain, torch.ones_like(gain))
    gain = _push_pull(gain, conf.expand_as(gain))
    gain = gaussian_blur(gain, 0.5 * sg)
    gain = F.interpolate(gain, size=(h, w), mode="bicubic", align_corners=False)
    return gain.clamp(*_LOCAL_LIM)


# ----------------------------------------------------------------------------
# public
# ----------------------------------------------------------------------------
def _eye_info(model, **kw):
    info = {"model": model, "matrix": torch.eye(3).tolist(), "bias": [0.0, 0.0, 0.0],
            "local_field": False, "samples": 0,
            "holdout_error_identity": 0.0, "holdout_error_model": 0.0}
    info.update(kw)
    return info


def color_correct(src_lin: torch.Tensor, edt_lin: torch.Tensor, bg_weight: torch.Tensor,
                  mode: str = "global", strength: float = 1.0):
    """Returns (corrected edit (1,3,H,W) linear, clamped >= 0 only; JSON-serialisable info)."""
    if mode == "off" or strength <= 0:
        return edt_lin, _eye_info("off")
    if mode not in ("global", "global_plus_local"):
        raise ValueError(f"unknown colour mode {mode!r}")
    h, w = src_lin.shape[-2:]
    n_trusted = int((bg_weight > 0.5).sum())
    if n_trusted < max(2000, 0.005 * h * w):
        return edt_lin, _eye_info("none", warning=f"trusted background too small ({n_trusted} px); "
                                                   "colour correction skipped")
    smp = _sample(src_lin, edt_lin, bg_weight)
    if smp is None or smp[0].shape[0] < 500:
        return edt_lin, _eye_info("none", warning="too few usable background samples (clipped/edges); "
                                                   "colour correction skipped")
    S, E, parity = smp
    warn = []
    e_id = _holdout(S, E, parity, None)
    e_diag = _holdout(S, E, parity, True)
    e_mat = _holdout(S, E, parity, False)
    model = "identity"
    M, b = torch.eye(3).to(S), torch.zeros(3).to(S)
    e_model = e_id
    if e_mat < 0.95 * e_diag and e_mat < e_id:
        Mm, bm = _fit(S, E, False)
        if _within_limits(Mm, bm):
            model, M, b, e_model = "matrix3x3", Mm, bm, e_mat
        else:
            warn.append("3x3 colour matrix outside limits; fell back to diagonal")
    if model == "identity" and e_diag < 0.99 * e_id:
        M, b = _fit(S, E, True)
        model, e_model = "diagonal", e_diag

    cor = apply_color_model(edt_lin, M, b) if model != "identity" else edt_lin
    local = False
    if mode == "global_plus_local":
        ok = ((src_lin < _CLIP_HI) & (cor < _CLIP_HI)).all(1, keepdim=True).to(src_lin)
        cor = cor.clamp_min(0) * _local_gain(src_lin, cor.clamp_min(0), bg_weight.clamp(0, 1) * ok,
                                             diag_px(h, w, 4.0))
        local = True
    if strength < 1.0:
        cor = edt_lin + float(strength) * (cor - edt_lin)
    info = {"model": model, "matrix": M.cpu().tolist(), "bias": b.cpu().tolist(),
            "local_field": local, "samples": int(S.shape[0]),
            "holdout_error_identity": float(e_id), "holdout_error_model": float(e_model)}
    if warn:
        info["warning"] = "; ".join(warn)
    return cor.clamp_min(0.0), info
