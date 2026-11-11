"""ComfyUI nodes: thin wrappers around the edit_drift_fix core."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from .edit_drift_fix import blend as _blend
from .edit_drift_fix import color as _color
from .edit_drift_fix import grain as _grain
from .edit_drift_fix.mask import MASK_MODES, detect_changes, prepare_user_mask
from .edit_drift_fix.pipeline import (BLEND_MODES, COLOR_MODES, GLOBAL_POLICIES, PRESET_NAMES, fix_batch, settings_from)
from .edit_drift_fix.register import MODELS, is_identity, register, warp
from .edit_drift_fix.util import diag_px, dilate, erode, linear_to_srgb, srgb_to_linear

CATEGORY = "image/Edit Drift Fix"


def _device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _progress(total):
    try:
        import comfy.utils
        bar = comfy.utils.ProgressBar(total)
        return lambda done, n: bar.update_absolute(done, n)
    except Exception:
        return None


def _run(fn):
    """Run on the GPU; fall back to the CPU if memory runs out."""
    dev = _device()
    try:
        return fn(dev)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fn(torch.device("cpu"))


def _chw(img: torch.Tensor, i: int, dev) -> torch.Tensor:
    return img[min(i, img.shape[0] - 1)][..., :3].to(device=dev, dtype=torch.float32).permute(2, 0, 1)[None].contiguous()


def _m(mask: torch.Tensor | None, i: int, dev, hw) -> torch.Tensor | None:
    if mask is None:
        return None
    return prepare_user_mask(mask[min(i, mask.shape[0] - 1)].to(device=dev, dtype=torch.float32)[None, None], hw)


def _hwc(x: torch.Tensor) -> torch.Tensor:
    return x[0].permute(1, 2, 0).cpu()


def _pairs(*batched):
    return max(b.shape[0] for b in batched if b is not None)


# ============================================================================
# main node
# ============================================================================
class EditDriftFix:
    DESCRIPTION = ("Pastes an AI edit back into the original: aligns geometry, colour and grain, finds the "
                   "changed area and returns the original pixels everywhere outside it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE", {"tooltip": "The original (before the edit). Never resampled."}),
                "edited": ("IMAGE", {"tooltip": "Output of the editing model. May differ in size from the original."}),
                "preset": (list(PRESET_NAMES), {"default": "spot_edit", "tooltip": (
                    "Ready-made settings for the task:\n"
                    "• spot_edit — a small fix (detail, lettering, blemish)\n"
                    "• object_edit — object replaced, added or removed; extra room for shadows\n"
                    "• large_soft_edit — large soft change (lighting, background, clothes)\n"
                    "• full_image — the whole frame changed: geometry is aligned, nothing is pasted\n"
                    "• custom — use the values below\n"
                    "Picking a preset fills the fields below; editing a field by hand switches to custom.")}),
                "mask_mode": (list(MASK_MODES), {"default": "auto", "tooltip": (
                    "Where the paste area comes from:\n"
                    "• auto — detect changes automatically\n"
                    "• auto_inside_mask — detect, but only inside edit_mask (rejects side changes)\n"
                    "• auto_plus_mask — detected changes plus edit_mask\n"
                    "• mask_only — exactly edit_mask\n"
                    "• whole_image — no paste: return the whole aligned edit")}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05, "tooltip": (
                    "Sensitivity of change detection. Higher — weaker differences (shadows, highlights) enter "
                    "the mask; lower — only obvious ones. The threshold is derived from the noise of the pair itself.")}),
                "minor_filter": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                    "Filter for side changes. Models often slightly repaint the background (text, foliage, faces) — "
                    "such weakly changed regions are dropped and stay original.\n"
                    "• 0 — off: every detected region goes into the mask\n"
                    "• 0.5 — default\n"
                    "• 1 — only the main edit is kept\n"
                    "The number of dropped regions is in the report (dropped_minor_regions).")}),
                "mask_expand": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": (
                    "Safety margin around the detected edit, % of the image diagonal.")}),
                "feather": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": (
                    "Width of the soft transition, % of the diagonal. Built inwards, never outside the mask.")}),
                "color_fix": (list(COLOR_MODES), {"default": "global_plus_local", "tooltip": (
                    "Bring the colour of the edit back to the original (estimated on unchanged background only):\n"
                    "• off — leave as is\n• global — overall colour and brightness shift\n"
                    "• global_plus_local — plus a smooth local correction")}),
                "grain_match": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                    "Add the missing grain of the original to the pasted area. 0 — off.")}),
            },
            "optional": {
                "edit_mask": ("MASK", {"tooltip": "Optional hint: where the edit was made. An empty mask is ignored."}),
                "options": ("EDF_OPTIONS", {"tooltip": "Advanced settings from the Edit Drift Fix Options node."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "aligned_edited", "report")
    OUTPUT_TOOLTIPS = ("Result at the size of the original: outside the mask the original pixels, bit for bit.",
                       "Soft paste mask.",
                       "The edit, aligned and colour-corrected, not composited.",
                       "JSON: motion model, residual shift, dE, mask area, confidence, warnings.")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, edited, preset, mask_mode, sensitivity, minor_filter, mask_expand, feather, color_fix, grain_match,
            edit_mask=None, options=None):
        settings = settings_from(preset, options, mask_mode=mask_mode, sensitivity=sensitivity, minor_filter=minor_filter,
                                 mask_expand=mask_expand,
                                 feather=feather, color_fix=color_fix, grain_match=grain_match)
        prog = _progress(max(source.shape[0], edited.shape[0]))
        image, mask, aligned, report = _run(lambda dev: fix_batch(source, edited, edit_mask, settings, dev, prog))
        return {"ui": {"text": [_summary(report)]}, "result": (image, mask, aligned, report)}


def _summary(report_json: str) -> str:
    """One-line digest shown on the node after execution."""
    try:
        rep = json.loads(report_json)
        items = rep.get("items", [rep])
        lines = []
        for r in items:
            reg, m = r.get("registration", {}), r.get("mask", {})
            area = m.get("final_area_fraction", m.get("area_fraction", 0)) or 0
            line = (f"{reg.get('model')} · drift {reg.get('max_displacement_px', 0)} px · "
                    f"mask {100 * area:.1f}% · conf {r.get('confidence')}")
            if m.get("global_edit"):
                line += " · GLOBAL EDIT"
            if r.get("warnings"):
                line += f" · ⚠ {len(r['warnings'])}"
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        return ""


class EditDriftFixOptions:
    DESCRIPTION = "Advanced settings for Edit Drift Fix. Optional: the defaults suit most cases."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "motion_model": (list(MODELS), {"default": "auto", "tooltip": (
                "Geometry drift model:\n• auto — the simplest one that explains the shift\n"
                "• translation — shift only\n• scale_shift — per-axis scale + shift (typical drift)\n"
                "• similarity — scale + rotation + shift\n• none — size fit only")}),
            "max_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 25.0, "step": 0.5,
                                    "tooltip": "Limit of the residual shift, % of the diagonal. Estimates beyond it are rejected."}),
            "blend_mode": (list(BLEND_MODES), {"default": "multiband", "tooltip": (
                "multiband — Laplacian-pyramid blending (the seam is less visible on textures);\nalpha — plain feathering.")}),
            "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Strength of the colour correction."}),
            "edit_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                "How much of the edit is mixed in inside the mask. 1 — the full edit; lower values blend it "
                "with the original, for when the model overdid the change.")}),
            "global_edit_policy": (list(GLOBAL_POLICIES), {"default": "aligned_edit", "tooltip": (
                "What to do when the automatic mask covers most of the frame (a global edit):\n"
                "• aligned_edit — return the aligned edit\n• raw_edit — return the edit as is (at the original size)\n"
                "• force_composite — composite anyway")}),
            "global_threshold": ("FLOAT", {"default": 40.0, "min": 5.0, "max": 100.0, "step": 1.0,
                                           "tooltip": "Share of the frame (%) from which the edit counts as global."}),
            "min_region": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 5.0, "step": 0.01,
                                     "tooltip": "Ignore changed spots smaller than this share of the frame (%)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Noise seed for grain matching."}),
        }}

    RETURN_TYPES = ("EDF_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        return (kw,)


# ============================================================================
# building blocks (same core functions, for custom graphs)
# ============================================================================
class EditDriftAlign:
    DESCRIPTION = "Alignment only: brings the edit to the geometry and size of the original with a single resampling."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",), "edited": ("IMAGE",),
            "motion_model": (list(MODELS), {"default": "auto"}),
            "max_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 25.0, "step": 0.5, "tooltip": "% of the diagonal"}),
        }, "optional": {"exclude_mask": ("MASK", {"tooltip": "Edit area: excluded from the shift estimate."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("aligned", "valid_mask", "transform")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, edited, motion_model, max_shift, exclude_mask=None):
        def go(dev):
            imgs, valids, infos = [], [], []
            for i in range(_pairs(source, edited)):
                s, e = srgb_to_linear(_chw(source, i, dev)), srgb_to_linear(_chw(edited, i, dev))
                hw = s.shape[-2:]
                reg = register(s, e, exclude=_m(exclude_mask, i, dev, hw), model=motion_model, max_shift_pct=max_shift)
                if is_identity(reg, hw, e.shape[-2:]):
                    out, valid = e, torch.ones(1, 1, *hw, device=dev)
                else:
                    out, valid = warp(e, reg.matrix, hw)
                imgs.append(_hwc(linear_to_srgb(out)))
                valids.append(valid[0, 0].cpu())
                infos.append({"model": reg.model, "matrix_source_to_edited": reg.matrix.tolist(), **reg.info})
            return torch.stack(imgs), torch.stack(valids), json.dumps(infos[0] if len(infos) == 1 else infos, indent=2)
        with torch.no_grad():
            return _run(go)


class EditDriftColor:
    DESCRIPTION = "Colour only: brings the colour of the aligned edit back to the original using unchanged background."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",), "aligned": ("IMAGE", {"tooltip": "The edit, already aligned to the original (same size)."}),
            "mode": (list(COLOR_MODES[1:]), {"default": "global_plus_local"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {"edit_mask": ("MASK", {"tooltip": "Edit area: excluded from the colour estimate."})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, aligned, mode, strength, edit_mask=None):
        def go(dev):
            out = []
            for i in range(_pairs(source, aligned)):
                s, e = srgb_to_linear(_chw(source, i, dev)), srgb_to_linear(_same_size(_chw(aligned, i, dev), _chw(source, i, dev)))
                m = _m(edit_mask, i, dev, s.shape[-2:])
                bg = torch.ones(1, 1, *s.shape[-2:], device=dev)
                if m is not None:
                    bg = 1.0 - dilate((m > 0.05).float(), max(3, round(diag_px(*s.shape[-2:], 0.5))))
                c, _ = _color.color_correct(s, e, erode(bg, 2), mode=mode, strength=strength)
                out.append(_hwc(linear_to_srgb(c)))
            return (torch.stack(out),)
        with torch.no_grad():
            return _run(go)


class EditDriftMask:
    DESCRIPTION = "Mask only: finds where the aligned edit really differs from the original."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",), "aligned": ("IMAGE", {"tooltip": "The edit aligned to the original (preferably colour-corrected)."}),
            "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05}),
            "minor_filter": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Drop weakly changed regions (side repaints of the background). 0 — off."}),
            "mask_expand": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "% of the diagonal"}),
            "min_region": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "% of the frame area"}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, aligned, sensitivity, minor_filter, mask_expand, min_region):
        def go(dev):
            out = []
            for i in range(_pairs(source, aligned)):
                s, e = srgb_to_linear(_chw(source, i, dev)), srgb_to_linear(_same_size(_chw(aligned, i, dev), _chw(source, i, dev)))
                core, _ = detect_changes(s, e, torch.ones(1, 1, *s.shape[-2:], device=dev), sensitivity, min_region,
                                         minor_filter=minor_filter)
                out.append(dilate(core, round(diag_px(*s.shape[-2:], mask_expand)))[0, 0].cpu())
            return (torch.stack(out),)
        with torch.no_grad():
            return _run(go)


class EditDriftComposite:
    DESCRIPTION = "Composite only, with a ready mask: outside the mask the original pixels are returned untouched."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",), "aligned": ("IMAGE",), "mask": ("MASK",),
            "feather": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "% of the diagonal, inwards"}),
            "blend_mode": (list(BLEND_MODES), {"default": "multiband"}),
            "grain_match": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "edit_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "How much of the edit is mixed in inside the mask."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, aligned, mask, feather, blend_mode, grain_match, edit_opacity=1.0):
        def go(dev):
            imgs, alphas = [], []
            for i in range(_pairs(source, aligned)):
                src = _chw(source, i, dev)
                s, e = srgb_to_linear(src), srgb_to_linear(_same_size(_chw(aligned, i, dev), src))
                m = _m(mask, i, dev, s.shape[-2:])
                hard = torch.zeros(1, 1, *s.shape[-2:], device=dev) if m is None else (m > 0.5).float()
                if float(hard.max()) == 0:
                    imgs.append(_hwc(src))
                    alphas.append(hard[0, 0].cpu())
                    continue
                alpha = _blend.feather_alpha(hard, diag_px(*s.shape[-2:], feather)) * float(edit_opacity)
                comp = _blend.composite(s, e, alpha, hard, mode=blend_mode)
                if grain_match > 0:
                    comp, _ = _grain.match_grain(s, comp, e, alpha, hard, strength=grain_match)
                imgs.append(_hwc(torch.where(hard > 0, linear_to_srgb(comp), src)))
                alphas.append(alpha[0, 0].cpu())
            return torch.stack(imgs), torch.stack(alphas)
        with torch.no_grad():
            return _run(go)


def _same_size(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if img.shape[-2:] == ref.shape[-2:]:
        return img
    return F.interpolate(img, size=ref.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)


# ============================================================================
# prevention: pad the input to the model grid instead of letting the model resize/crop it
# ============================================================================
class EditDriftPad:
    DESCRIPTION = ("Before generation: pads the image to sides that are multiples of the model grid (no resize), "
                   "so the model does not crop or rescale the frame. After generation use Edit Drift Unpad.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "multiple": ([8, 16, 32, 64], {"default": 32, "tooltip": "Model grid: 16 or 32 for Qwen / FLUX.2."}),
            "fill": (["reflect", "edge"], {"default": "reflect", "tooltip": "How the added margins are filled."}),
        }}

    RETURN_TYPES = ("IMAGE", "EDF_PAD")
    RETURN_NAMES = ("image", "pad")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, multiple, fill):
        h, w = image.shape[1:3]
        ph, pw = (-h) % multiple, (-w) % multiple
        top, left = ph // 2, pw // 2
        pad = {"top": top, "bottom": ph - top, "left": left, "right": pw - left, "height": h, "width": w}
        if ph == 0 and pw == 0:
            return (image, pad)
        x = image.permute(0, 3, 1, 2)
        mode = "reflect" if fill == "reflect" and max(ph, pw) < min(h, w) else "replicate"
        x = F.pad(x, (pad["left"], pad["right"], pad["top"], pad["bottom"]), mode=mode)
        return (x.permute(0, 2, 3, 1).contiguous(), pad)


class EditDriftUnpad:
    DESCRIPTION = "After generation: crops the margins added by Edit Drift Pad (also when the model changed the resolution)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "pad": ("EDF_PAD",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, pad):
        h, w = image.shape[1:3]
        fy = h / (pad["height"] + pad["top"] + pad["bottom"])
        fx = w / (pad["width"] + pad["left"] + pad["right"])
        y0, x0 = round(pad["top"] * fy), round(pad["left"] * fx)
        y1, x1 = h - round(pad["bottom"] * fy), w - round(pad["right"] * fx)
        return (image[:, y0:y1, x0:x1, :],)


NODE_CLASS_MAPPINGS = {
    "EditDriftFix": EditDriftFix,
    "EditDriftFixOptions": EditDriftFixOptions,
    "EditDriftAlign": EditDriftAlign,
    "EditDriftColor": EditDriftColor,
    "EditDriftMask": EditDriftMask,
    "EditDriftComposite": EditDriftComposite,
    "EditDriftPad": EditDriftPad,
    "EditDriftUnpad": EditDriftUnpad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EditDriftFix": "Edit Drift Fix",
    "EditDriftFixOptions": "Edit Drift Fix Options",
    "EditDriftAlign": "Edit Drift Align",
    "EditDriftColor": "Edit Drift Color",
    "EditDriftMask": "Edit Drift Mask",
    "EditDriftComposite": "Edit Drift Composite",
    "EditDriftPad": "Edit Drift Pad",
    "EditDriftUnpad": "Edit Drift Unpad",
}
