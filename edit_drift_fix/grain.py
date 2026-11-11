"""Grain matching: add to the pasted region the noise the source has and the edit lacks.

Noise is measured and synthesised in sRGB-encoded values (roughly perceptually uniform grain).
"""
from __future__ import annotations

import math

import torch

from .util import (diag_px, dilate, erode, gaussian_blur, linear_to_srgb, quantile, robust_stats,
                   sobel_mag, srgb_to_linear)

_NB = 6           # luminance bins
_MIN_FLAT = 300   # flat pixels needed per bin
_LUMA = (0.2126, 0.7152, 0.0722)


def _hp_factor() -> float:
    """Std of (x - gaussian_blur(x, 1.0)) for unit white noise (same discrete kernel as util)."""
    k = [math.exp(-0.5 * t * t) for t in range(-3, 4)]
    s = sum(k)
    k = [v / s for v in k]
    return math.sqrt(sum(((1.0 if (i, j) == (3, 3) else 0.0) - k[i] * k[j]) ** 2
                         for i in range(7) for j in range(7)))


_HPF = _hp_factor()


def _luma(srgb):
    w = torch.tensor(_LUMA, dtype=srgb.dtype, device=srgb.device).view(1, 3, 1, 1)
    return (srgb * w).sum(1, keepdim=True)


def _measure(y, region):
    """Per-luminance-bin white-noise-equivalent sigma of luma y inside region (NaN = unknown)."""
    yb = gaussian_blur(y, 1.0)
    hp = (y - yb)[0, 0]
    grad = gaussian_blur(sobel_mag(yb), 2.0)[0, 0]
    sel = region[0, 0] > 0.5
    g, r = grad[sel], hp[sel]
    bins = (yb[0, 0][sel] * _NB).long().clamp(0, _NB - 1)
    sig = [float("nan")] * _NB
    for b in range(_NB):
        mb = bins == b
        if int(mb.sum()) < 2 * _MIN_FLAT:
            continue
        gb = g[mb]
        flat = gb <= quantile(gb, 0.4)  # lowest-gradient pixels: texture is not noise
        rr = r[mb][flat]
        if rr.numel() >= _MIN_FLAT:
            sig[b] = robust_stats(rr)[1] / _HPF
    return sig


def _fill_bins(sig):
    """Replace unknown bins by the nearest known one; None if nothing is known."""
    known = [i for i, v in enumerate(sig) if not math.isnan(v)]
    if not known:
        return None
    return [sig[min(known, key=lambda k: abs(k - i))] for i in range(len(sig))]


def estimate_noise(lin: torch.Tensor, region: torch.Tensor) -> float:
    """Median per-bin noise sigma (0..255 sRGB units) of `lin` inside region; NaN if unmeasurable."""
    s = [v for v in _measure(_luma(linear_to_srgb(lin)), region) if not math.isnan(v)]
    return float(torch.tensor(s).median()) * 255.0 if s else float("nan")


def _median(v):
    return float(torch.tensor(v).median()) * 255.0


def match_grain(src_lin: torch.Tensor, comp_lin: torch.Tensor, edt_lin: torch.Tensor,
                alpha: torch.Tensor, hard_mask: torch.Tensor, strength: float = 0.5, seed: int = 0):
    """Add missing grain (sigma_add = strength * sqrt(max(0, s_src^2 - s_edit^2))) where alpha > 0.

    Pixels with alpha == 0 are returned bit-exactly."""
    info = {"applied": False, "src_sigma": 0.0, "edit_sigma": 0.0, "added_sigma": 0.0}
    if strength <= 0:
        info["reason"] = "strength is 0"
        return comp_lin, info
    hard = (hard_mask > 0.5).to(src_lin.dtype)
    if float(hard.max()) <= 0:
        info["reason"] = "empty mask"
        return comp_lin, info
    h, w = src_lin.shape[-2:]
    ring_w = max(16, round(diag_px(h, w, 3.0)))
    ring = dilate(hard, ring_w + 3) - dilate(hard, 3)   # skip pixels whose high-pass sees the edit
    inner = erode(hard, 3)
    s_src = _fill_bins(_measure(_luma(linear_to_srgb(src_lin)), ring))
    s_edt = _fill_bins(_measure(_luma(linear_to_srgb(edt_lin)), inner))
    if s_src is None or s_edt is None:
        info["reason"] = "too few flat pixels to measure noise " + ("near the mask" if s_src is None else "in the edit")
        return comp_lin, info
    miss = [math.sqrt(max(0.0, a * a - b * b)) * float(strength) for a, b in zip(s_src, s_edt)]
    info.update(src_sigma=_median(s_src), edit_sigma=_median(s_edt), added_sigma=_median(miss))
    if max(miss) * 255.0 < 0.05:
        info["reason"] = "edit is not less noisy than the source"
        return comp_lin, info

    dev, dt = comp_lin.device, comp_lin.dtype
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    rnd = torch.randn((1, 5, h, w), generator=gen, device=dev, dtype=dt)
    soft = gaussian_blur(rnd[:, 1:2], 0.7)
    lum = 0.6 * rnd[:, 0:1] + 0.4 * soft / soft.std().clamp_min(1e-8)
    noise = lum + 0.3 * rnd[:, 2:5]             # mostly luminance, a little chroma
    yn = _luma(noise)
    # calibrate so the estimator sees unit white-equivalent sigma, like it does on the source
    noise = noise * (_HPF / max(robust_stats((yn - gaussian_blur(yn, 1.0)).flatten())[1], 1e-8))

    comp_srgb = linear_to_srgb(comp_lin)
    table = torch.tensor(miss, dtype=dt, device=dev)
    pos = (gaussian_blur(_luma(comp_srgb), 1.0) * _NB - 0.5).clamp(0, _NB - 1)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=_NB - 1)
    f = pos - i0.to(dt)
    sig = table[i0] * (1 - f) + table[i1] * f
    add = noise * sig * alpha
    # add in sRGB, return as a linear delta so values > 1 in comp are kept
    out = comp_lin + (srgb_to_linear(comp_srgb + add) - srgb_to_linear(comp_srgb))
    out = torch.where(alpha > 0, out, comp_lin)
    info["applied"] = True
    return out, info
