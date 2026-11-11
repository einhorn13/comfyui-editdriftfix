"""Change mask: where does the (aligned, colour-corrected) edit really differ from the source?

The threshold is not a constant: it is derived from the noise of this very pair
(median + k * 1.4826 * MAD of the difference on the background), separately for
flat and textured areas, because VAE error grows with texture.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .util import (cv2, delta_e, dilate, erode, gaussian_blur, lightness, mask_to_np, np_to_mask, sobel_mag, _gen)

MASK_MODES = ("auto", "auto_inside_mask", "auto_plus_mask", "mask_only", "whole_image")
_BINS = 8


def _zscore(d: torch.Tensor, tex: torch.Tensor, bg: torch.Tensor, edges: torch.Tensor, floor: float,
            cap: float = float("inf")) -> torch.Tensor:
    """Per-pixel robust z-score of `d`, with noise statistics binned by local texture.

    `cap` bounds what may be called noise: when most of the frame is repainted the
    "background" statistics describe the edit itself and must not hide it."""
    idx = torch.bucketize(tex, edges)                      # 0.._BINS-1
    med = torch.zeros(_BINS, device=d.device)
    mad = torch.zeros(_BINS, device=d.device)
    sel_all = bg > 0.5
    g_med = d[sel_all].median() if bool(sel_all.any()) else d.median()
    for b in range(_BINS):
        v = d[sel_all & (idx == b)]
        if v.numel() < 200:
            med[b], mad[b] = g_med, floor
            continue
        if v.numel() > 60_000:
            v = v[:: v.numel() // 60_000 + 1]
        m = v.median()
        med[b] = m
        mad[b] = 1.4826 * (v - m).abs().median()
    # noise cannot be lower in a more textured bin; enforce monotone, smooth statistics
    mad = torch.cummax(mad.clamp(floor, max(floor, 0.5 * cap)), 0).values
    med = torch.cummax(med.clamp_max(cap), 0).values
    return (d - med[idx]) / mad[idx]


def detect_changes(src_lin: torch.Tensor, edt_lin: torch.Tensor, valid: torch.Tensor, sensitivity: float = 1.0,
                   min_region_pct: float = 0.02, exclude_hint: torch.Tensor | None = None, work_long: int = 2048,
                   minor_filter: float = 0.5):
    """Binary core mask of changed pixels (no safety margin yet) + info dict.

    src_lin / edt_lin : aligned, ideally colour-corrected, linear RGB
    valid             : 1 where the warped edit has data
    exclude_hint      : pixels known to be edited (kept out of the noise statistics)
    minor_filter      : 0..1, how eagerly weak side changes (background repaints) are dropped;
                        0 keeps every detected region, 0.5 is the tuned default
    """
    full_hw = src_lin.shape[-2:]
    f = max(1.0, max(full_hw) / float(work_long))
    if f > 1.0:
        hw = (round(full_hw[0] / f), round(full_hw[1] / f))
        src_lin = F.interpolate(src_lin, size=hw, mode="area")
        edt_lin = F.interpolate(edt_lin, size=hw, mode="area")
        valid = (F.interpolate(valid, size=hw, mode="area") > 0.99).float()
        if exclude_hint is not None:
            exclude_hint = (F.interpolate(exclude_hint, size=hw, mode="area") > 0.01).float()
    h, w = src_lin.shape[-2:]
    diag = math.hypot(h, w)
    sens = max(0.05, float(sensitivity))
    mf = min(max(float(minor_filter), 0.0), 1.0)
    s_f = max(1.0, 0.0008 * diag)
    s_c = 3.0 * s_f

    sb, eb = gaussian_blur(src_lin, s_f), gaussian_blur(edt_lin, s_f)
    d_f = delta_e(sb, eb)[0, 0]
    d_c = delta_e(gaussian_blur(src_lin, s_c), gaussian_blur(edt_lin, s_c))[0, 0]
    gs, ge = sobel_mag(lightness(sb)), sobel_mag(lightness(eb))
    d_g = 100.0 * gaussian_blur((gs - ge).abs(), s_f)[0, 0]
    tex = gaussian_blur(torch.maximum(gs, ge), 2.0 * s_f)[0, 0]

    bg = erode(valid, max(2, round(2 * s_c)))[0, 0]
    if exclude_hint is not None:
        bg = bg * (1.0 - dilate((exclude_hint > 0.05).float(), round(4 * s_c)))[0, 0]
    k_hi, k_lo = 7.0 / sens, 3.5 / sens
    min_area = max(12, int(min_region_pct * 0.01 * h * w))
    close_r = max(1, round(0.0015 * diag))
    info: dict = {}
    core = torch.zeros(1, 1, h, w, device=src_lin.device)

    for it in range(2):
        sel = tex[bg > 0.5]
        if sel.numel() < 2000:           # nothing trustworthy to measure noise on
            sel = tex.flatten()
            bg = torch.ones_like(bg)
        if sel.numel() > 200_000:
            sel = sel[torch.randint(0, sel.numel(), (200_000,), device=sel.device, generator=_gen(sel.device))]
        edges = torch.quantile(sel, torch.linspace(0, 1, _BINS + 1, device=sel.device)[1:-1])
        z_f = _zscore(d_f, tex, bg, edges, 0.25, cap=3.0)   # dE units; VAE noise is ~1-2
        z_c = _zscore(d_c, tex, bg, edges, 0.15, cap=2.0)
        z_g = _zscore(d_g, tex, bg, edges, 0.15)

        def level(k, scale):
            return (((z_f > k) & (d_f > 2.0 * scale / sens))
                    | ((z_c > k) & (d_c > 1.5 * scale / sens))
                    | ((z_g > 1.5 * k) & (d_f > 1.0 * scale / sens)))

        strong = level(k_hi, 1.0) & (valid[0, 0] > 0.5)
        weak = level(k_lo, 0.5) & (valid[0, 0] > 0.5)
        weak_m = erode(dilate(weak.float()[None, None], close_r), close_r)
        excess = (d_f - 2.0 / sens).clamp_min(0) * (valid[0, 0] > 0.5)
        core = _hysteresis(weak_m, strong, min_area, like=src_lin, excess=excess,
                           rel_energy=min(0.5, 0.16 * mf / sens), weak_frac=0.5 * mf,
                           solid_frac=min(1.0, 1.2 * mf), info=info)
        if it == 0:
            # re-estimate the noise without the detected region (it inflates MAD)
            bg = bg * (1.0 - dilate(core, round(4 * s_c)))[0, 0]
            if float(core.mean()) < 0.002:
                break

    core = _fill_holes(core, max_hole=int(0.02 * h * w))
    n_comp = int(cv2.connectedComponents(mask_to_np(core), connectivity=8)[0]) - 1
    info.update(area_fraction=round(float(core.mean()), 5), components=n_comp,
                noise_delta_e=round(float(d_f[bg > 0.5].median()) if bool((bg > 0.5).any()) else 0.0, 3))
    if f > 1.0:
        core = (F.interpolate(core, size=full_hw, mode="bilinear", align_corners=False) > 0.25).float()
    return core, info


def _hysteresis(weak: torch.Tensor, strong: torch.Tensor, min_area: int, like: torch.Tensor,
                excess: torch.Tensor | None = None, rel_energy: float = 0.0, weak_frac: float = 0.25,
                solid_frac: float = 0.6, info: dict | None = None) -> torch.Tensor:
    """Keep weak components that contain strong evidence and are not tiny.

    Edit models also repaint small background details (text, faces, foliage). Such
    regions are real differences but carry little evidence compared with the actual
    edit, so components whose summed excess difference is below `rel_energy` of the
    strongest component, or that are mostly weak, are dropped (they stay original)."""
    w_np = mask_to_np(weak)
    s_np = strong.detach().cpu().numpy().astype(np.float64)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(w_np, connectivity=8)
    if n <= 1:
        return torch.zeros_like(weak)
    flat = labels.ravel()
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    strong_count = np.bincount(flat, weights=s_np.ravel(), minlength=n)
    keep = (area >= min_area) & (strong_count >= max(4, 0.02 * min_area))
    keep[0] = False
    if excess is not None and keep.any():
        energy = np.bincount(flat, weights=excess.detach().cpu().numpy().astype(np.float64).ravel(), minlength=n)
        energy[0] = 0
        top = energy[keep].max()
        frac = strong_count / np.maximum(area, 1)
        # mostly-weak regions are repaint noise; a solidly changed region is an edit however small
        minor = keep & ((frac < weak_frac) | ((energy < rel_energy * top) & (frac < solid_frac)))
        # the strongest region is the edit itself, however faint: an edit that only changes
        # an expression must not be thrown away together with the repaint noise
        minor[int(np.argmax(np.where(keep, energy, -1.0)))] = False
        if minor.any() and (keep & ~minor).any():
            # a weak region right next to the main edit is part of it (shadow, the far side of an occluder)
            r = max(2, int(round(0.03 * float(np.hypot(*w_np.shape)))))
            near = cv2.distanceTransform(1 - (keep & ~minor)[labels].astype(np.uint8), cv2.DIST_L2, 3) <= r
            touching = np.bincount(flat, weights=near.ravel().astype(np.float64), minlength=n) > 0
            minor &= ~touching
        if info is not None:
            info["dropped_minor_regions"] = int(minor.sum())
            info["dropped_minor_area"] = round(float(area[minor].sum() / flat.size), 5)
        keep &= ~minor
    return np_to_mask(keep[labels], like)


def _fill_holes(mask: torch.Tensor, max_hole: int) -> torch.Tensor:
    """Fill enclosed holes smaller than `max_hole` px (large enclosed areas are real background)."""
    m = mask_to_np(mask)
    inv = (1 - m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n <= 1:
        return mask
    h, w = m.shape
    fill = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        fill[i] = (not touches) and area <= max_hole
    return np_to_mask(np.maximum(m, fill[labels].astype(np.uint8)), mask)


def prepare_user_mask(mask: torch.Tensor | None, hw) -> torch.Tensor | None:
    """(1,1,h,w) user mask -> source size; an empty mask counts as 'not connected'."""
    if mask is None:
        return None
    if mask.shape[-2:] != tuple(hw):
        mask = F.interpolate(mask, size=hw, mode="bilinear", align_corners=False)
    mask = mask.clamp(0, 1)
    return mask if float(mask.max()) > 0.01 else None


def finalize_mask(core: torch.Tensor | None, user: torch.Tensor | None, mode: str, valid: torch.Tensor, expand_px: float):
    """Combine automatic and user masks, add the safety margin. Returns (hard mask, effective mode)."""
    e = int(round(expand_px))
    ub = None if user is None else (user > 0.5).float()
    if mode != "whole_image" and mode != "auto" and ub is None:
        mode = "auto"                      # nothing connected: behave as auto
    if mode == "whole_image":
        hard = torch.ones_like(valid)
    elif mode == "mask_only":
        hard = dilate(ub, e)
    elif mode == "auto_inside_mask":
        hard = dilate(core * ub, e) * dilate(ub, e)
    elif mode == "auto_plus_mask":
        hard = dilate(torch.maximum(core, ub), e)
    else:
        hard = dilate(core, e)
    if mode == "whole_image":
        return hard, mode
    ok = (valid > 0.5).float()
    hard = hard * ok
    if float(ok.min()) == 0:
        # The model zoomed in, so a thin band along the frame has no edit data. Where the mask
        # reaches that band, continue it to the frame edge (the warp replicates edge pixels there)
        # instead of leaving a strip of the original object.
        r = max(2, round(0.02 * math.hypot(*valid.shape[-2:])))
        hard = torch.maximum(hard, dilate(hard, r) * (1.0 - ok))
    return hard, mode
