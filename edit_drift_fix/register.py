"""Registration of an edited image onto the original.

The drift produced by edit models is, physically, a per-axis scale plus a shift
(hidden resize of the reference, rounding of sides to 8/16/32, centre crops).
So the motion models are deliberately small:
    identity -> translation -> scale_shift (4 dof) -> similarity (4 dof)
A richer model is accepted only when it changes the geometry noticeably and
lowers the robust residual on the background.

Everything is expressed as 3x3 matrices in pixel-centre coordinates that map
SOURCE pixels to EDITED pixels, so the edited image is resampled exactly once.

Ideas (not code) for edit-zone exclusion and Cauchy/MAD weighting follow
WepeNerd Align Composite (MIT) and 1hew Image Align Change Mask (Apache-2.0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .util import dilate, gaussian_blur, lightness, sobel_mag

MAX_SCALE_DEV = 0.15      # |scale - 1| limit of the residual transform
MAX_ROT_DEG = 3.0
MODELS = ("auto", "translation", "scale_shift", "similarity", "none")


@dataclass
class Registration:
    matrix: torch.Tensor                 # 3x3 float64 (cpu): source px -> edited px
    model: str = "identity"
    info: dict = field(default_factory=dict)
    size_matrix: torch.Tensor | None = None   # the pure size-normalisation part
    work: torch.Tensor | None = None          # residual transform in working coords (2x3, centred)


# ----------------------------------------------------------------------------
# matrices
# ----------------------------------------------------------------------------
def _mat(rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def size_matrix(src_hw, edt_hw, fit: str = "stretch") -> torch.Tensor:
    """Initial source->edited mapping that only accounts for the size change.

    stretch : the whole source maps onto the whole edit (independent axis scales)
    cover   : uniform scale, the edit is a centre crop of the source
    contain : uniform scale, the edit shows the whole source plus extra margins
    """
    h, w = src_hw
    he, we = edt_hw
    sx, sy = we / w, he / h
    if fit == "cover":
        sx = sy = max(sx, sy)
    elif fit == "contain":
        sx = sy = min(sx, sy)
    # xe + 0.5 = sx * (x + 0.5 - w/2) + we/2
    return _mat([[sx, 0.0, sx * (0.5 - w / 2) + we / 2 - 0.5],
                 [0.0, sy, sy * (0.5 - h / 2) + he / 2 - 0.5],
                 [0.0, 0.0, 1.0]])


def _resize_matrix(from_hw, to_hw) -> torch.Tensor:
    """Pixel-centre mapping from a grid of size `from_hw` to the same extent at `to_hw`."""
    return size_matrix(from_hw, to_hw, "stretch")


def _work_to_matrix(p: torch.Tensor, hw) -> torch.Tensor:
    """Centred 2x3 working transform -> 3x3 matrix in working pixel coords."""
    cy, cx = (hw[0] - 1) / 2, (hw[1] - 1) / 2
    lin = p[:, :2]
    c = _mat([cx, cy])
    t = c + p[:, 2] - lin @ c
    m = torch.eye(3, dtype=torch.float64)
    m[:2, :2] = lin
    m[:2, 2] = t
    return m


# ----------------------------------------------------------------------------
# warping (the only place where the edited image is resampled)
# ----------------------------------------------------------------------------
def warp(img: torch.Tensor, matrix: torch.Tensor, out_hw, mode: str = "bicubic"):
    """Resample `img` (1,C,He,We) onto a grid of size out_hw.

    `matrix` maps output pixel coords -> img pixel coords.
    Returns (warped, valid) where valid (1,1,H,W) marks pixels fully backed by data."""
    h, w = out_hw
    he, we = img.shape[-2:]
    m = matrix.to(torch.float64)
    scale = math.sqrt(abs(float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])))
    if scale > 1.15:  # minification: band-limit first to avoid aliasing
        img = gaussian_blur(img, 0.5 * math.sqrt(scale * scale - 1.0))
    n_out_inv = _mat([[w / 2, 0, w / 2 - 0.5], [0, h / 2, h / 2 - 0.5], [0, 0, 1]])
    n_in = _mat([[2 / we, 0, 1 / we - 1], [0, 2 / he, 1 / he - 1], [0, 0, 1]])
    theta = (n_in @ m @ n_out_inv)[:2].to(img.dtype).to(img.device)[None]
    grid = F.affine_grid(theta, (1, img.shape[1], h, w), align_corners=False)
    out = F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=False)
    mx, my = 1.0 - 2.0 / we, 1.0 - 2.0 / he      # one-pixel safety margin
    valid = ((grid[..., 0].abs() <= mx) & (grid[..., 1].abs() <= my)).to(img.dtype)[:, None]
    return out, valid


# ----------------------------------------------------------------------------
# features and pyramid
# ----------------------------------------------------------------------------
def _feature(light: torch.Tensor) -> torch.Tensor:
    """Gradient-magnitude map: insensitive to colour/brightness drift."""
    g = sobel_mag(gaussian_blur(light, 1.0))
    g = gaussian_blur(g, 0.8)
    return g / (g.mean() + 1e-6)


def _pyramid(light_s, light_e, weight, min_side=40, min_long=96, max_levels=6):
    hw = light_s.shape[-2:]
    levels = []
    f = 1
    while True:
        size = (max(1, round(hw[0] / f)), max(1, round(hw[1] / f)))
        if f == 1:
            s, e, w = light_s, light_e, weight
        else:
            s = F.interpolate(light_s, size=size, mode="area")
            e = F.interpolate(light_e, size=size, mode="area")
            w = F.interpolate(weight, size=size, mode="area")
            w = (w > 0.6).to(w.dtype) * w
        fe = _feature(e)
        dx = (F.pad(fe, (1, 1, 0, 0), mode="replicate")[..., 2:] - F.pad(fe, (1, 1, 0, 0), mode="replicate")[..., :-2]) * 0.5
        dy = (F.pad(fe, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(fe, (0, 0, 1, 1), mode="replicate")[..., :-2, :]) * 0.5
        fx, fy = hw[1] / size[1], hw[0] / size[0]
        levels.append({"S": _feature(s), "E3": torch.cat([fe, dx / fx, dy / fy], 1), "W": w, "f": f})
        nxt = f * 2
        if len(levels) >= max_levels or min(hw) / nxt < min_side or max(hw) / nxt < min_long:
            break
        f = nxt
    return levels[::-1]  # coarse -> fine


def _sample(level, p: torch.Tensor, hw):
    """Sample the edited feature (+ its gradients) at the positions given by `p`."""
    e3 = level["E3"]
    h, w = e3.shape[-2:]
    dev, dt = e3.device, e3.dtype
    cy, cx = (hw[0] - 1) / 2, (hw[1] - 1) / 2
    u = ((torch.arange(w, device=dev, dtype=dt) + 0.5) * (hw[1] / w) - 0.5 - cx)[None, :].expand(h, w)
    v = ((torch.arange(h, device=dev, dtype=dt) + 0.5) * (hw[0] / h) - 0.5 - cy)[:, None].expand(h, w)
    q = p.to(dt)
    xp = cx + q[0, 0] * u + q[0, 1] * v + q[0, 2]
    yp = cy + q[1, 0] * u + q[1, 1] * v + q[1, 2]
    gx = (xp + 0.5) / hw[1] * 2 - 1
    gy = (yp + 0.5) / hw[0] * 2 - 1
    out = F.grid_sample(e3, torch.stack([gx, gy], -1)[None], mode="bilinear", padding_mode="border", align_corners=False)
    valid = ((gx.abs() < 1 - 3.0 / w) & (gy.abs() < 1 - 3.0 / h)).to(dt)
    return out[0], valid, u, v


def _gain(ew, s, w):
    return float(((w * ew * s).sum() / ((w * ew * ew).sum() + 1e-8)).clamp(0.25, 4.0))


def _cost(level, p, hw) -> float:
    """Robust (L1) background residual of the transform `p` on this level."""
    out, valid, _, _ = _sample(level, p, hw)
    w = level["W"][0, 0] * valid
    s = level["S"][0, 0]
    g = _gain(out[0], s, w)
    return float((w * (g * out[0] - s).abs()).sum() / (w.sum() + 1e-6))


def _refine(level, p: torch.Tensor, hw, model: str, iters: int):
    """Gauss-Newton / IRLS (Cauchy weights) refinement of `p` on one level."""
    r_norm = 0.5 * math.hypot(hw[0], hw[1])
    cond = 1.0
    p = p.clone()
    for _ in range(iters):
        out, valid, u, v = _sample(level, p, hw)
        ew, ex, ey = out[0], out[1], out[2]
        s = level["S"][0, 0]
        w = level["W"][0, 0] * valid
        if float(w.sum()) < 64:
            break
        g = _gain(ew, s, w)
        r = g * ew - s
        sel = r[w > 0]
        if sel.numel() > 200_000:
            sel = sel[:: sel.numel() // 200_000 + 1]
        med = sel.median()
        sigma = float(1.4826 * (sel - med).abs().median()) + 1e-4
        w = w / (1.0 + (r / (2.0 * sigma)) ** 2)
        ex, ey = g * ex, g * ey
        if model == "translation":
            cols = [ex, ey]
        elif model == "scale_shift":
            cols = [ex * u / r_norm, ey * v / r_norm, ex, ey]
        else:  # similarity
            cols = [(ex * u + ey * v) / r_norm, (ey * u - ex * v) / r_norm, ex, ey]
        j = torch.stack([c.reshape(-1) for c in cols], 1)
        jw = j * w.reshape(-1, 1)
        a = (jw.T @ j).double().cpu()
        b = (jw.T @ r.reshape(-1, 1)).double().cpu()[:, 0]
        ev = torch.linalg.eigvalsh(a)
        cond = float(ev[0] / (ev[-1] + 1e-30))
        a = a + torch.eye(a.shape[0], dtype=torch.float64) * (1e-6 * float(ev[-1]) + 1e-12)
        try:
            d = -torch.linalg.solve(a, b)
        except Exception:
            break
        if not torch.isfinite(d).all():
            break
        step = float(d.abs().max())
        if step > 0.25 * r_norm:  # absurd step -> damp
            d = d * (0.25 * r_norm / step)
        if model == "translation":
            p[0, 2] += d[0]; p[1, 2] += d[1]
        elif model == "scale_shift":
            p[0, 0] += d[0] / r_norm; p[1, 1] += d[1] / r_norm
            p[0, 2] += d[2]; p[1, 2] += d[3]
        else:
            p[0, 0] += d[0] / r_norm; p[1, 1] += d[0] / r_norm
            p[0, 1] -= d[1] / r_norm; p[1, 0] += d[1] / r_norm
            p[0, 2] += d[2]; p[1, 2] += d[3]
        if step < 0.004:
            break
    return p, cond


def _phase_seed(level, hw):
    """Translation candidates (working px) from phase correlation on the coarsest level."""
    s = level["S"][0, 0] * level["W"][0, 0]
    e = level["E3"][0, 0] * level["W"][0, 0]
    h, w = s.shape
    win = torch.hann_window(h, device=s.device)[:, None] * torch.hann_window(w, device=s.device)[None, :]
    fs = torch.fft.rfft2((s - s.mean()) * win)
    fe = torch.fft.rfft2((e - e.mean()) * win)
    cross = fs * torch.conj(fe)
    corr = torch.fft.irfft2(cross / (cross.abs() + 1e-6), s=(h, w))
    idx = int(corr.argmax())
    dy, dx = idx // w, idx % w
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    fx, fy = hw[1] / w, hw[0] / h
    return [(dx * fx, dy * fy), (-dx * fx, -dy * fy)]


def _corner_shift(pa: torch.Tensor, pb: torch.Tensor, hw) -> float:
    """Largest displacement difference (working px) between two transforms at the corners."""
    cy, cx = (hw[0] - 1) / 2, (hw[1] - 1) / 2
    pts = _mat([[-cx, -cy, 1], [cx, -cy, 1], [-cx, cy, 1], [cx, cy, 1], [0, 0, 1]])
    return float(((pa - pb) @ pts.T).norm(dim=0).max())


def _within_limits(p, hw, max_shift_px) -> bool:
    sx = math.hypot(float(p[0, 0]), float(p[1, 0]))
    sy = math.hypot(float(p[0, 1]), float(p[1, 1]))
    rot = math.degrees(math.atan2(float(p[1, 0] - p[0, 1]), float(p[0, 0] + p[1, 1])))
    shift = math.hypot(float(p[0, 2]), float(p[1, 2]))
    return (abs(sx - 1) <= MAX_SCALE_DEV and abs(sy - 1) <= MAX_SCALE_DEV
            and abs(rot) <= MAX_ROT_DEG and shift <= max_shift_px and bool(torch.isfinite(p).all()))


def _residual_shift(level, p, hw, grid=8):
    """Diagnostic: median local translation still left on the background (working px)."""
    out, valid, _, _ = _sample(level, p, hw)
    ew, ex, ey = out[0], out[1], out[2]
    s = level["S"][0, 0]
    w = level["W"][0, 0] * valid
    g = _gain(ew, s, w)
    r = g * ew - s
    ex, ey = g * ex, g * ey
    maps = torch.stack([w * ex * ex, w * ex * ey, w * ey * ey, w * ex * r, w * ey * r, w])[None]
    pooled = F.adaptive_avg_pool2d(maps, (grid, grid))[0].reshape(6, -1).double().cpu()
    axx, axy, ayy, bx, by, cov = pooled
    det = axx * ayy - axy * axy
    tr = axx + ayy
    lam_min = tr / 2 - torch.sqrt((tr / 2) ** 2 - det).clamp_min(0)
    ok = (cov > 0.5) & (lam_min > 0.05 * float(tr.mean() + 1e-12)) & (det > 1e-12)
    if int(ok.sum()) < 3:
        return None, int(ok.sum())
    dx = -(ayy * bx - axy * by) / det.clamp_min(1e-12)
    dy = -(axx * by - axy * bx) / det.clamp_min(1e-12)
    mag = torch.sqrt(dx * dx + dy * dy)[ok]
    return float(mag.median()), int(ok.sum())


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def _pick_size_fit(src_light_small, edt_lin, src_hw):
    """When the aspect ratios differ decide whether the model stretched, cropped or padded."""
    best, best_score, scores = "stretch", -1.0, {}
    hw = src_light_small.shape[-2:]
    fs = _feature(src_light_small)
    for fit in ("stretch", "cover", "contain"):
        m = size_matrix(src_hw, edt_lin.shape[-2:], fit) @ _resize_matrix(hw, src_hw)
        e, valid = warp(edt_lin, m, hw, mode="bilinear")
        fe = _feature(lightness(e))
        a = gaussian_blur(fs, 1.5) * valid
        b = gaussian_blur(fe, 1.5) * valid
        a, b = a - a.sum() / valid.sum().clamp_min(1), b - b.sum() / valid.sum().clamp_min(1)
        score = float((a * b * valid).sum() / (torch.sqrt((a * a * valid).sum() * (b * b * valid).sum()) + 1e-9))
        scores[fit] = round(score, 4)
        if score > best_score + (0.02 if fit != "stretch" else 0.0):
            best, best_score = fit, score
    return best, scores


def register(src_lin: torch.Tensor, edt_lin: torch.Tensor, exclude: torch.Tensor | None = None,
             model: str = "auto", max_shift_pct: float = 5.0, init: Registration | None = None,
             work_long: int = 1536) -> Registration:
    """Estimate the source->edited mapping.

    exclude : (1,1,H,W) mask in source space, 1 = do not use these pixels (edit zone)
    init    : previous result to refine (second pass after the edit zone is known)
    """
    h, w = src_lin.shape[-2:]
    he, we = edt_lin.shape[-2:]
    warnings: list[str] = []
    info: dict = {"requested_model": model, "warnings": warnings}

    fw = max(1.0, max(h, w) / float(work_long))
    hw = (max(8, round(h / fw)), max(8, round(w / fw)))
    src_small = src_lin if hw == (h, w) else F.interpolate(src_lin, size=hw, mode="area")
    light_s = lightness(src_small)
    up = _resize_matrix(hw, (h, w))          # working px -> source px
    down = _resize_matrix((h, w), hw)

    # --- size normalisation --------------------------------------------------
    if init is not None and init.size_matrix is not None:
        a0 = init.size_matrix
        info["size_fit"] = init.info.get("size_fit", "stretch")
    else:
        fit = "stretch"
        if abs((we / w) / (he / h) - 1.0) > 0.004:
            small = F.interpolate(light_s, size=(max(8, round(hw[0] * 256 / max(hw))), max(8, round(hw[1] * 256 / max(hw)))), mode="area")
            fit, scores = _pick_size_fit(small, edt_lin, (h, w))
            info["size_fit_scores"] = scores
        a0 = size_matrix((h, w), (he, we), fit)
        info["size_fit"] = fit
    info["size_scale"] = [round(float(a0[0, 0]), 6), round(float(a0[1, 1]), 6)]

    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    if model == "none":
        info.update(scale=[1.0, 1.0], shift_px=[0.0, 0.0], rotation_deg=0.0, passes=0)
        return Registration(a0.clone(), "none", info, a0, identity)

    # --- working copies ------------------------------------------------------
    e_work, valid = warp(edt_lin, a0 @ up, hw, mode="bilinear")
    light_e = lightness(e_work)
    weight = valid.clone()
    b = 3
    weight[..., :b, :] = 0; weight[..., -b:, :] = 0; weight[..., :, :b] = 0; weight[..., :, -b:] = 0
    if exclude is not None:
        ex = exclude if exclude.shape[-2:] == hw else F.interpolate(exclude, size=hw, mode="bilinear", align_corners=False)
        weight = weight * (1.0 - dilate((ex > 0.05).to(weight.dtype), max(2, round(0.004 * math.hypot(*hw)))))
    bg_frac = float(weight.mean())
    info["background_fraction"] = round(bg_frac, 4)
    if bg_frac < 0.03:
        warnings.append("almost no background left for registration; size normalisation only")
        info.update(scale=[1.0, 1.0], shift_px=[0.0, 0.0], rotation_deg=0.0, passes=0, reliable=False)
        return Registration(a0.clone(), "identity", info, a0, identity)

    levels = _pyramid(light_s, light_e, weight)
    fine = levels[-1]
    max_shift = max_shift_pct * 0.01 * math.hypot(*hw)
    iters = [30, 20, 12, 8, 6, 5]

    def run(p0, mdl, lvls):
        p, cond = p0.clone(), 1.0
        for i, lv in enumerate(lvls):
            k = len(lvls) - 1 - i               # 0 = finest
            p, cond = _refine(lv, p, hw, mdl, iters[max(0, len(iters) - 1 - k)] if k < len(iters) else 30)
        return p, cond

    c_id = _cost(fine, identity, hw)
    cands: dict[str, tuple[torch.Tensor, float]] = {"identity": (identity, c_id)}
    conds: dict[str, float] = {}

    if init is not None and init.work is not None:
        start, lvls = init.work.clone(), levels[-2:]
    else:
        start, lvls = identity.clone(), levels
        coarse = levels[0]
        best_c = _cost(coarse, start, hw)
        for dx, dy in _phase_seed(coarse, hw):
            if math.hypot(dx, dy) > max_shift or math.hypot(dx, dy) < 1.0:
                continue
            cand = identity.clone(); cand[0, 2] = dx; cand[1, 2] = dy
            c = _cost(coarse, cand, hw)
            if c < 0.93 * best_c:
                start, best_c = cand, c

    want = ("translation", "scale_shift", "similarity") if model == "auto" else (model,)
    p_t = start
    if "translation" in want or model == "auto":
        # keep any scale/rotation already present in a warm start
        p_t, conds["translation"] = run(start, "translation", lvls)
        if init is None or model == "translation" or init.model in ("identity", "translation"):
            cands["translation"] = (p_t, _cost(fine, p_t, hw))
    for mdl in ("scale_shift", "similarity"):
        if mdl in want:
            base = p_t if model == "auto" else start
            if mdl == "similarity" and init is not None and init.model != "similarity":
                base = base.clone(); base[0, 0] = base[1, 1] = 0.5 * (base[0, 0] + base[1, 1])
            p, conds[mdl] = run(base, mdl, lvls)
            cands[mdl] = (p, _cost(fine, p, hw))

    # --- model selection -----------------------------------------------------
    chosen = "identity"
    order = [m for m in ("translation", "scale_shift", "similarity") if m in cands]
    for mdl in order:
        p, c = cands[mdl]
        cur_p, cur_c = cands[chosen]
        if not _within_limits(p, hw, max_shift):
            warnings.append(f"{mdl}: estimate outside limits, rejected")
            continue
        if mdl != "translation" and conds.get(mdl, 1.0) < 1e-7:
            warnings.append(f"{mdl}: ill-conditioned (too little texture), rejected")
            continue
        moved = _corner_shift(p, cur_p, hw)
        if model != "auto":
            # an explicitly requested model only has to be no worse than doing nothing
            if c <= c_id * 1.002:
                chosen = mdl
            else:
                warnings.append(f"{mdl}: residual not better than identity, rejected")
            continue
        if moved < 0.04:
            continue                                   # no geometric difference: keep the simpler model
        # unrelated content (global edit) still "improves" by <1%; real drift improves by far more
        need = {"translation": 0.985, "scale_shift": 0.985, "similarity": 0.975}[mdl]
        if c < cur_c * need:
            chosen = mdl
    p_fin, c_fin = cands[chosen]
    if chosen == "identity" and model not in ("auto",) and order:
        warnings.append("fell back to identity")

    res, n_blocks = _residual_shift(fine, p_fin, hw)
    textured = n_blocks >= 6
    if not textured:
        warnings.append("little texture on the background: registration is weakly constrained")
    scale_to_src = math.hypot(h, w) / math.hypot(*hw)
    sx = math.hypot(float(p_fin[0, 0]), float(p_fin[1, 0]))
    sy = math.hypot(float(p_fin[0, 1]), float(p_fin[1, 1]))
    info.update(
        scale=[round(sx, 6), round(sy, 6)],
        rotation_deg=round(math.degrees(math.atan2(float(p_fin[1, 0] - p_fin[0, 1]), float(p_fin[0, 0] + p_fin[1, 1]))), 4),
        shift_px=[round(float(p_fin[0, 2]) * scale_to_src, 3), round(float(p_fin[1, 2]) * scale_to_src, 3)],
        max_displacement_px=round(_corner_shift(p_fin, identity, hw) * scale_to_src, 3),
        residual_shift_px=None if res is None else round(res * scale_to_src, 3),
        textured_blocks=n_blocks,
        cost_identity=round(c_id, 5), cost_final=round(c_fin, 5),
        candidates={k: round(v[1], 5) for k, v in cands.items()},
        reliable=bool(textured and c_fin <= c_id * 1.002),
    )
    matrix = a0 @ up @ _work_to_matrix(p_fin, hw) @ down
    return Registration(matrix, chosen, info, a0, p_fin)


def is_identity(reg: Registration, src_hw, edt_hw) -> bool:
    """True when no resampling is needed at all (same size, no motion)."""
    return tuple(src_hw) == tuple(edt_hw) and bool((reg.matrix - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-9)
