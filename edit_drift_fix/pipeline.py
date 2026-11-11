"""Full pipeline: align -> colour -> change mask -> composite -> grain -> report."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields

import torch

from . import blend as _blend
from . import color as _color
from . import grain as _grain
from .mask import MASK_MODES, detect_changes, finalize_mask, prepare_user_mask
from .register import MODELS, Registration, is_identity, register, size_matrix, warp
from .util import delta_e, diag_px, dilate, erode, linear_to_srgb, quantile, srgb_to_linear

COLOR_MODES = ("off", "global", "global_plus_local")
BLEND_MODES = ("multiband", "alpha")
GLOBAL_POLICIES = ("aligned_edit", "raw_edit", "force_composite")


@dataclass
class Settings:
    # main controls
    mask_mode: str = "auto"
    sensitivity: float = 1.0        # higher = smaller differences count as an edit
    minor_filter: float = 0.5       # 0..1: drop weak side changes (background repaints); 0 = keep all
    mask_expand: float = 0.6        # safety margin around the edit, % of diagonal
    feather: float = 0.5            # transition width (inside the margin), % of diagonal
    color_fix: str = "global_plus_local"
    color_strength: float = 1.0
    grain_match: float = 0.5
    edit_opacity: float = 1.0       # 0..1: how much of the edit is mixed in inside the mask
    # advanced
    motion_model: str = "auto"
    max_shift: float = 5.0          # limit of the residual shift, % of diagonal
    blend_mode: str = "multiband"
    global_edit_policy: str = "aligned_edit"
    global_threshold: float = 40.0  # % of the frame; a bigger auto mask means "global edit"
    min_region: float = 0.02        # % of the frame; smaller changed spots are ignored
    seed: int = 0

    def validated(self) -> "Settings":
        s = Settings(**asdict(self))
        for name, allowed in (("mask_mode", MASK_MODES), ("color_fix", COLOR_MODES), ("motion_model", MODELS),
                              ("blend_mode", BLEND_MODES), ("global_edit_policy", GLOBAL_POLICIES)):
            if getattr(s, name) not in allowed:
                raise ValueError(f"{name}={getattr(s, name)!r}; expected one of {allowed}")
        s.sensitivity = min(max(float(s.sensitivity), 0.1), 5.0)
        s.minor_filter = min(max(float(s.minor_filter), 0.0), 1.0)
        s.color_strength = min(max(float(s.color_strength), 0.0), 1.0)
        s.edit_opacity = min(max(float(s.edit_opacity), 0.0), 1.0)
        s.grain_match = min(max(float(s.grain_match), 0.0), 1.0)
        return s


# Presets only touch the main controls (the widgets of the main node).
# NOTE: web/edit_drift_fix.js mirrors this table to update the widgets; keep both in sync.
PRESET_KEYS = ("mask_mode", "sensitivity", "minor_filter", "mask_expand", "feather", "color_fix", "grain_match")
PRESETS: dict[str, dict] = {
    # small local change: tight mask, strict protection of everything else
    "spot_edit": dict(mask_mode="auto", sensitivity=1.0, minor_filter=0.5, mask_expand=0.5, feather=0.4,
                      color_fix="global_plus_local", grain_match=0.5),
    # object replaced / added / removed: more room for shadows and reflections
    "object_edit": dict(mask_mode="auto", sensitivity=1.2, minor_filter=0.5, mask_expand=1.2, feather=1.0,
                        color_fix="global_plus_local", grain_match=0.5),
    # large soft change (lighting, background, clothes): wide soft transition
    "large_soft_edit": dict(mask_mode="auto", sensitivity=1.5, minor_filter=0.5, mask_expand=2.5, feather=2.5,
                            color_fix="global", grain_match=0.3),
    # whole picture changed (style, relight): only geometry is restored, no pasting
    "full_image": dict(mask_mode="whole_image", sensitivity=1.0, minor_filter=0.0, mask_expand=0.0, feather=0.0,
                       color_fix="off", grain_match=0.0),
}
PRESET_NAMES = tuple(PRESETS) + ("custom",)


def apply_preset(settings: Settings, preset: str) -> Settings:
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            setattr(settings, k, v)
    return settings


def settings_from(preset: str = "custom", options: dict | None = None, **main) -> Settings:
    names = {f.name for f in fields(Settings)}
    s = Settings(**{k: v for k, v in main.items() if k in names})
    apply_preset(s, preset)
    for k, v in (options or {}).items():
        if k in names:
            setattr(s, k, v)
    return s.validated()


# ----------------------------------------------------------------------------
def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
    return float((x * m).sum() / m.sum().clamp_min(1.0))


def _align(src_lin, edt_lin, reg: Registration):
    """The single resampling of the edit."""
    hw = src_lin.shape[-2:]
    if is_identity(reg, hw, edt_lin.shape[-2:]):
        return edt_lin, torch.ones(1, 1, *hw, device=src_lin.device)
    out, valid = warp(edt_lin, reg.matrix, hw, mode="bicubic")
    return out.clamp(0.0, 1.0), valid


def fix_pair(source: torch.Tensor, edited: torch.Tensor, user_mask: torch.Tensor | None = None,
             settings: Settings | None = None):
    """Process one pair.

    source, edited : (1,3,H,W) / (1,3,He,We) sRGB float32 on the working device
    user_mask      : (1,1,h,w) or None
    Returns (image (1,3,H,W) sRGB, alpha (1,1,H,W), aligned edit (1,3,H,W) sRGB, report dict)."""
    s = (settings or Settings()).validated()
    h, w = source.shape[-2:]
    warnings: list[str] = []
    src_lin = srgb_to_linear(source)
    edt_lin = srgb_to_linear(edited)
    user = prepare_user_mask(user_mask, (h, w))
    mode = s.mask_mode
    if mode in ("auto_inside_mask", "auto_plus_mask", "mask_only") and user is None:
        warnings.append(f"mask_mode={mode} needs edit_mask, but it is empty or not connected: using auto")
        mode = "auto"
    auto = mode in ("auto", "auto_inside_mask", "auto_plus_mask")
    expand_px = diag_px(h, w, s.mask_expand)
    feather_px = diag_px(h, w, s.feather)
    margin = max(3, round(expand_px + diag_px(h, w, 0.3)))

    # ---- 1. registration (pass 1: only the user hint is excluded) ----------
    hint = user if mode != "whole_image" and user is not None and mode != "auto_inside_mask" else None
    reg = register(src_lin, edt_lin, exclude=hint, model=s.motion_model, max_shift_pct=s.max_shift)
    passes = 0 if s.motion_model == "none" else 1
    aligned, valid = _align(src_lin, edt_lin, reg)

    # ---- 2. coarse change mask, then registration without the edit zone ----
    core, mask_info, is_global = None, {}, False
    if auto:
        bg0 = valid if hint is None else valid * (1.0 - dilate((hint > 0.05).float(), margin))
        pre, _ = _color.color_correct(src_lin, aligned, erode(bg0, 2), mode="global" if s.color_fix != "off" else "off")
        core, mask_info = detect_changes(src_lin, pre, valid, s.sensitivity, s.min_region, exclude_hint=hint,
                                           minor_filter=s.minor_filter)
        frac = float(core.mean())
        is_global = frac * 100.0 > s.global_threshold
        if not is_global and frac > 0 and s.motion_model != "none":
            excl = dilate(core, margin) if hint is None else torch.maximum(dilate(core, margin), (hint > 0.05).float())
            reg2 = register(src_lin, edt_lin, exclude=excl, model=s.motion_model, max_shift_pct=s.max_shift, init=reg)
            passes = 2
            moved = float((reg2.matrix - reg.matrix).abs()[:2].max() * max(h, w))  # crude px bound
            reg = reg2
            if moved > 0.03:
                aligned, valid = _align(src_lin, edt_lin, reg)
                bg0 = valid * (1.0 - excl)
                pre, _ = _color.color_correct(src_lin, aligned, erode(bg0, 2), mode="global" if s.color_fix != "off" else "off")
                core, mask_info = detect_changes(src_lin, pre, valid, s.sensitivity, s.min_region, exclude_hint=excl,
                                                   minor_filter=s.minor_filter)
                is_global = float(core.mean()) * 100.0 > s.global_threshold
    warnings += [f"registration: {x}" for x in reg.info.get("warnings", [])]

    report: dict = {
        "registration": {"model": reg.model, "passes": passes,
                         **{k: reg.info.get(k) for k in ("size_fit", "size_scale", "scale", "shift_px", "rotation_deg",
                                                         "max_displacement_px", "residual_shift_px", "background_fraction")}},
        "settings": asdict(s),
    }

    # ---- 3. global edit: nothing trustworthy to paste into ------------------
    if is_global and s.global_edit_policy != "force_composite":
        warnings.append(f"global edit detected ({mask_info.get('area_fraction', 0) * 100:.0f}% of the frame changed): "
                        f"returning the {'aligned' if s.global_edit_policy == 'aligned_edit' else 'raw'} edit, no composite")
        if s.global_edit_policy == "raw_edit":
            raw, v = warp(edt_lin, size_matrix((h, w), edt_lin.shape[-2:]), (h, w)) if edt_lin.shape[-2:] != (h, w) else (edt_lin, torch.ones_like(valid))
            out_lin, valid = raw.clamp(0, 1), v
        else:
            out_lin = aligned
        img = linear_to_srgb(out_lin)        # uncovered frame edges hold replicated edit pixels
        valid = torch.ones_like(valid)
        report.update(mask={**mask_info, "mode": mode, "global_edit": True}, color={"model": "off"},
                      confidence=0.3, warnings=warnings, outside_mask_changed_pixels=0)
        return img, valid.clone(), img, report

    # ---- 4. colour: estimated on eroded background, applied to the whole edit
    if mode == "whole_image":
        bg = valid
    elif mode == "mask_only":
        bg = valid * (1.0 - dilate((user > 0.5).float(), margin))
    else:
        # every detected change is kept out of the statistics, even one the user mask will reject
        known = torch.maximum(core, (user > 0.5).float()) if mode == "auto_plus_mask" else core
        bg = valid * (1.0 - dilate(known, margin))
    bg = erode(bg, 2)
    corrected, color_info = _color.color_correct(src_lin, aligned, bg, mode=s.color_fix, strength=s.color_strength)
    if color_info.get("warning"):
        warnings.append(f"color: {color_info['warning']}")
    de_before = _masked_mean(delta_e(src_lin, aligned), bg)
    de_after = _masked_mean(delta_e(src_lin, corrected), bg)
    for k in ("matrix", "bias"):
        if k in color_info:
            color_info[k] = torch.tensor(color_info[k], dtype=torch.float64).round(decimals=5).tolist()
    for k in ("holdout_error_identity", "holdout_error_model"):
        if isinstance(color_info.get(k), float):
            color_info[k] = round(color_info[k], 5)
    color_info.update(background_delta_e_before=_r(de_before), background_delta_e_after=_r(de_after))

    # ---- 5. final mask ------------------------------------------------------
    hard, mode = finalize_mask(core, user, mode, valid, expand_px)
    hard = (hard > 0.5).float()
    if mode == "whole_image" and float(valid.mean()) < 0.999:
        warnings.append(f"the edit does not cover {100 * (1 - float(valid.mean())):.1f}% of the frame (the model zoomed in): frame edges are extended from the nearest edit pixels")
    alpha = _blend.feather_alpha(hard, feather_px) * s.edit_opacity
    area = float(hard.mean())

    # ---- 6. composite + grain ----------------------------------------------
    if area == 0.0:
        image, alpha = source, torch.zeros_like(hard)
        grain_info = {"applied": False, "reason": "empty mask"}
        if auto:
            warnings.append("no changes detected: output equals the source (raise sensitivity or supply edit_mask)")
    else:
        comp = _blend.composite(src_lin, corrected, alpha, hard, mode=s.blend_mode)
        grain_info = {"applied": False, "reason": "disabled"}
        if s.grain_match > 0 and mode != "whole_image":
            comp, grain_info = _grain.match_grain(src_lin, comp, corrected, alpha, hard, strength=s.grain_match, seed=s.seed)
        # outside the mask the ORIGINAL tensor is returned untouched, no arithmetic
        image = torch.where(hard > 0, linear_to_srgb(comp), source)
    aligned_out = linear_to_srgb(corrected)

    # ---- 7. metrics ---------------------------------------------------------
    ring_w = max(3, round(diag_px(h, w, 0.5)))
    ring = (dilate(hard, ring_w) - hard).clamp(0, 1) * valid
    if float(ring.sum()) >= 64:
        ring_de = delta_e(src_lin, corrected)[ring > 0.5]
        ring_med, ring_p95 = quantile(ring_de, 0.5), quantile(ring_de, 0.95)
    else:
        ring_med = ring_p95 = None
    outside = int((((image != source).any(1, keepdim=True)) & (hard == 0)).sum())

    conf = 1.0
    res = reg.info.get("residual_shift_px")
    if s.motion_model != "none":
        if not reg.info.get("reliable", True):
            conf *= 0.6
        if res is not None and res > 0.5:
            conf *= 0.6
            warnings.append(f"residual shift on the background is {res:.2f} px")
        elif res is not None and res > 0.3:
            conf *= 0.85
    if ring_p95 is not None and mode != "whole_image":
        # judged against this pair's own background noise, not an absolute number
        if ring_p95 > max(6.0, 4.0 * de_after):
            conf *= 0.7
            warnings.append(f"the edit differs from the source right next to the mask (dE p95 {ring_p95:.1f}): "
                            "the mask may be too tight - raise mask_expand or sensitivity")
        elif ring_med > max(2.0, 1.5 * de_after):
            conf *= 0.85
    if is_global:
        conf *= 0.5
        warnings.append("global edit composited because of force_composite")
    if outside:
        warnings.append(f"{outside} pixels differ outside the mask (unexpected)")

    report.update(
        mask={**mask_info, "mode": mode, "final_area_fraction": _r(area, 5), "global_edit": bool(is_global),
              "expand_px": _r(expand_px, 1), "feather_px": _r(feather_px, 1)},
        color=color_info, grain=grain_info, blend_mode=s.blend_mode,
        ring_delta_e={"median": _r(ring_med), "p95": _r(ring_p95)},
        outside_mask_changed_pixels=outside, confidence=_r(conf, 2), warnings=warnings,
    )
    return image, alpha, aligned_out, report


# ----------------------------------------------------------------------------
# batch front-end in ComfyUI tensor layout (kept here so it is testable without ComfyUI)
# ----------------------------------------------------------------------------
def fix_batch(source: torch.Tensor, edited: torch.Tensor, edit_mask: torch.Tensor | None, settings: Settings,
              device: torch.device | str = "cpu", progress=None):
    """source [B,H,W,C], edited [N,He,We,C], edit_mask [M,h,w] | None.

    One source may serve several edits and vice versa (shorter batches repeat their last item).
    Returns (image [K,H,W,3], mask [K,H,W], aligned [K,H,W,3], report json str)."""
    n = max(source.shape[0], edited.shape[0])
    images, masks, aligned, reports = [], [], [], []
    for i in range(n):
        src = source[min(i, source.shape[0] - 1)][..., :3]
        edt = edited[min(i, edited.shape[0] - 1)][..., :3]
        if src.shape[:2] != source[0].shape[:2]:
            raise ValueError("all source images in a batch must have the same size")
        m = None
        if edit_mask is not None:
            m = edit_mask[min(i, edit_mask.shape[0] - 1)]
            m = m.to(device=device, dtype=torch.float32)[None, None]
        src_d = src.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None].contiguous()
        edt_d = edt.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None].contiguous()
        with torch.no_grad():
            img, alpha, ali, rep = fix_pair(src_d, edt_d, m, settings)
        out_dev = source.device
        images.append(img[0].permute(1, 2, 0).to(out_dev))
        masks.append(alpha[0, 0].to(out_dev))
        aligned.append(ali[0].permute(1, 2, 0).to(out_dev))
        reports.append(rep)
        if progress is not None:
            progress(i + 1, n)
    report = reports[0] if n == 1 else {"items": reports}
    return torch.stack(images), torch.stack(masks), torch.stack(aligned), json.dumps(report, indent=2, ensure_ascii=False)
