"""Compositing of the (colour-corrected) edit over the source, in linear RGB.

The caller performs the final `torch.where(hard_mask > 0, blended, source)`; functions here
only produce the blended image.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .util import cv2, distance_inside, erode, mask_to_np


def feather_alpha(hard_mask: torch.Tensor, feather_px: float) -> torch.Tensor:
    """Inward feather: 0 outside the mask, smoothstep up to 1 at `feather_px` inside.

    Each connected component's ramp is shortened to 0.9x its maximal inner distance so thin
    parts still reach alpha = 1. Image borders are not boundaries."""
    m = (hard_mask > 0.5).to(hard_mask.dtype)
    if feather_px <= 0.5 or float(m.max()) <= 0:
        return m
    d = distance_inside(m)
    n_lab, labels = cv2.connectedComponents(mask_to_np(m), connectivity=8)
    lab = torch.from_numpy(labels.astype(np.int64)).to(m.device).flatten()
    dmax = torch.zeros(n_lab, dtype=d.dtype, device=d.device).scatter_reduce(
        0, lab, d.flatten(), reduce="amax", include_self=True)
    ramp = (0.9 * dmax).clamp(max=float(feather_px)).clamp_min(1e-3)[lab].view_as(d)
    t = (d / ramp).clamp(0, 1)
    return t * t * (3 - 2 * t) * m


def alpha_blend(src_lin: torch.Tensor, edt_lin: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.lerp(src_lin, edt_lin, alpha)  # exactly src where alpha == 0


# ----------------------------------------------------------------------------
# Laplacian pyramid (Burt-Adelson)
# ----------------------------------------------------------------------------
def _blur5(x):
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=x.dtype, device=x.device) / 16.0
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"), k.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)


def _down(x):
    return _blur5(x)[..., ::2, ::2]


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _auto_levels(alpha, hard):
    """Pyramid depth ~ log2(feather width) so coarse bands do not reach far past the ramp."""
    ramp = ((alpha > 0.01) & (alpha < 0.99)).sum()
    boundary = (hard - erode(hard, 1)).sum().clamp_min(1)
    feather = float(ramp) / float(boundary)
    return int(min(6, max(1, math.floor(math.log2(max(feather, 2.0))))))


def multiband_blend(src_lin: torch.Tensor, edt_lin: torch.Tensor, alpha: torch.Tensor,
                    hard_mask: torch.Tensor, levels: int | None = None) -> torch.Tensor:
    hard = (hard_mask > 0.5).to(src_lin.dtype)
    nz = hard[0, 0].nonzero()
    if nz.numel() == 0:
        return src_lin.clone()
    H, W = src_lin.shape[-2:]
    if levels is None:
        levels = _auto_levels(alpha, hard)
    levels = max(0, min(int(levels), 6))
    pad = 3 * (2 ** levels) + 4
    y0, x0 = (int(v) for v in nz.min(0).values)
    y1, x1 = (int(v) + 1 for v in nz.max(0).values)
    y0, x0, y1, x1 = max(0, y0 - pad), max(0, x0 - pad), min(H, y1 + pad), min(W, x1 + pad)
    while levels > 0 and min(y1 - y0, x1 - x0) / 2 ** levels < 4:
        levels -= 1
    if levels == 0:
        return alpha_blend(src_lin, edt_lin, alpha)

    # src, edt and alpha share one pyramid pass
    g = torch.cat([src_lin[..., y0:y1, x0:x1], edt_lin[..., y0:y1, x0:x1], alpha[..., y0:y1, x0:x1]], 1)
    gauss = [g]
    for _ in range(levels):
        gauss.append(_down(gauss[-1]))
    a0 = alpha[..., y0:y1, x0:x1]
    top = gauss[-1]
    # coarsest band: weight 1 wherever the area-averaged alpha is 1 (no source colour deep inside)
    w_top = torch.maximum(top[:, 6:7], F.interpolate(a0, size=top.shape[-2:], mode="area"))
    out = torch.lerp(top[:, 0:3], top[:, 3:6], w_top)
    for lvl in range(levels - 1, -1, -1):
        cur, size = gauss[lvl], gauss[lvl].shape[-2:]
        up = _up(gauss[lvl + 1], size)
        lap_s, lap_e = cur[:, 0:3] - up[:, 0:3], cur[:, 3:6] - up[:, 3:6]
        out = _up(out, size) + torch.lerp(lap_s, lap_e, cur[:, 6:7])
    res = src_lin.clone()
    res[..., y0:y1, x0:x1] = out
    return res


def composite(src_lin: torch.Tensor, edt_lin: torch.Tensor, alpha: torch.Tensor,
              hard_mask: torch.Tensor, mode: str = "alpha") -> torch.Tensor:
    if mode == "alpha":
        return alpha_blend(src_lin, edt_lin, alpha)
    if mode == "multiband":
        return multiband_blend(src_lin, edt_lin, alpha, hard_mask)
    raise ValueError(f"unknown blend mode {mode!r}")
