"""Shared tensor helpers for the Edit Drift Fix core.

Conventions used by every core module:
  * images  : float32 tensor (1, 3, H, W)
  * masks   : float32 tensor (1, 1, H, W), values in [0, 1]
  * "lin"   : linear-light RGB;  "srgb": gamma-encoded sRGB in [0, 1]
The core has no dependency on ComfyUI.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

try:  # OpenCV ships with ComfyUI; only morphology helpers need it.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------
# colour spaces
# ----------------------------------------------------------------------------
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


_M1 = ((0.4122214708, 0.5363325363, 0.0514459929),
       (0.2119034982, 0.6806995451, 0.1073969566),
       (0.0883024619, 0.2817188376, 0.6299787005))
_M2 = ((0.2104542553, 0.7936177850, -0.0040720468),
       (1.9779984951, -2.4285922050, 0.4505937099),
       (0.0259040371, 0.7827717662, -0.8086757660))


def linear_to_oklab(lin: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) linear RGB -> (1,3,H,W) OKLab (L in 0..1)."""
    m1 = torch.tensor(_M1, dtype=lin.dtype, device=lin.device)
    m2 = torch.tensor(_M2, dtype=lin.dtype, device=lin.device)
    lms = torch.einsum("ij,bjhw->bihw", m1, lin.clamp_min(0.0))
    lms = lms.clamp_min(1e-9) ** (1.0 / 3.0)
    return torch.einsum("ij,bjhw->bihw", m2, lms)


def delta_e(lin_a: torch.Tensor, lin_b: torch.Tensor) -> torch.Tensor:
    """OKLab colour difference scaled x100 (roughly comparable to CIE dE). -> (1,1,H,W)."""
    d = linear_to_oklab(lin_a) - linear_to_oklab(lin_b)
    return 100.0 * torch.sqrt((d * d).sum(1, keepdim=True) + 1e-12)


def luminance(lin: torch.Tensor) -> torch.Tensor:
    """Linear luminance (1,1,H,W)."""
    w = torch.tensor((0.2126, 0.7152, 0.0722), dtype=lin.dtype, device=lin.device).view(1, 3, 1, 1)
    return (lin * w).sum(1, keepdim=True)


def lightness(lin: torch.Tensor) -> torch.Tensor:
    """Perceptual-ish lightness in 0..1 (cube root of luminance). (1,1,H,W)."""
    return luminance(lin).clamp_min(0.0) ** (1.0 / 3.0)


# ----------------------------------------------------------------------------
# filtering
# ----------------------------------------------------------------------------
def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with replicate padding. x: (B,C,H,W)."""
    if sigma <= 0.05:
        return x
    h, w = x.shape[-2:]
    # Large sigmas: blur a downscaled copy (cheap, visually identical for low-pass use).
    if sigma > 6.0 and min(h, w) > 64:
        f = min(int(sigma // 3), max(1, min(h, w) // 32))
        if f >= 2:
            small = F.interpolate(x, size=(max(1, h // f), max(1, w // f)), mode="area")
            small = gaussian_blur(small, sigma / f)
            return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    rh, rw = min(r, h - 1), min(r, w - 1)
    kh = k[r - rh: r + rh + 1]
    kw = k[r - rw: r + rw + 1]
    kh, kw = kh / kh.sum(), kw / kw.sum()
    x = F.conv2d(F.pad(x, (rw, rw, 0, 0), mode="replicate"), kw.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, rh, rh), mode="replicate"), kh.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def masked_blur(x: torch.Tensor, weight: torch.Tensor, sigma: float, eps: float = 1e-4):
    """Normalized convolution: blur x using only pixels where weight > 0.

    Returns (blurred, coverage); values where coverage is ~0 are extrapolated
    by the caller (see `fill_by_diffusion`)."""
    num = gaussian_blur(x * weight, sigma)
    den = gaussian_blur(weight, sigma)
    return num / den.clamp_min(eps), den


def sobel_mag(gray: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of a (1,1,H,W) map."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3) / 8.0
    p = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(p, kx)
    gy = F.conv2d(p, kx.transpose(2, 3))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Approximately round dilation of a (1,1,H,W) mask by `radius` px."""
    radius = int(round(radius))
    if radius <= 0:
        return mask
    out = mask
    left = radius
    while left > 0:  # small square/cross steps alternate -> octagon, cheap on GPU
        s = min(left, 3)
        out = F.max_pool2d(out, 2 * s + 1, stride=1, padding=s)
        left -= s
    return out


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return 1.0 - dilate(1.0 - mask, radius)


def to_rgb01(img: torch.Tensor) -> torch.Tensor:
    """[..., C] image of any channel count -> finite 3-channel values in [0, 1].

    In-range RGB input comes back with identical values, so the source stays bit-exact."""
    c = img.shape[-1]
    if c == 1 or c == 2:
        img = img[..., :1].expand(*img.shape[:-1], 3)
    elif c > 3:
        img = img[..., :3]
    return torch.nan_to_num(img.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def mask_bhw(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Mask given as [H,W], [B,H,W], [B,1,H,W] or [B,H,W,1] -> finite [B,H,W]."""
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask[None]
    elif mask.dim() == 4:
        mask = mask[:, 0] if mask.shape[1] == 1 else mask[..., 0]
    if mask.dim() != 3 or mask.numel() == 0:
        return None
    return torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0)


def diag_px(h: int, w: int, percent: float) -> float:
    """Convert '% of image diagonal' into pixels."""
    return float(percent) * 0.01 * math.hypot(h, w)


# ----------------------------------------------------------------------------
# numpy / OpenCV bridges
# ----------------------------------------------------------------------------
def mask_to_np(mask: torch.Tensor) -> np.ndarray:
    return (mask[0, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)


def np_to_mask(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(like.device)[None, None]


def distance_inside(mask: torch.Tensor) -> torch.Tensor:
    """Euclidean distance (px) from every in-mask pixel to the nearest pixel outside
    the mask; 0 outside. Image borders do NOT count as 'outside'. (1,1,H,W)."""
    m = mask_to_np(mask)
    if m.min() > 0:  # mask covers everything
        return torch.full_like(mask, 1e6)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return np_to_mask(d, mask)


def robust_stats(values: torch.Tensor) -> tuple[float, float]:
    """(median, 1.4826*MAD) of a 1-D tensor; subsamples large inputs."""
    v = values.flatten()
    if v.numel() == 0:
        return 0.0, 0.0
    if v.numel() > 400_000:
        idx = torch.randint(0, v.numel(), (400_000,), device=v.device, generator=_gen(v.device))
        v = v[idx]
    med = v.median()
    mad = (v - med).abs().median()
    return float(med), float(1.4826 * mad)


_GENS: dict = {}


def _gen(device) -> torch.Generator:
    """Deterministic generator per device (results must not change between runs)."""
    key = str(device)
    g = _GENS.get(key)
    if g is None:
        g = torch.Generator(device=device)
        _GENS[key] = g
    g.manual_seed(1234)
    return g


def quantile(values: torch.Tensor, q: float) -> float:
    v = values.flatten()
    if v.numel() == 0:
        return 0.0
    if v.numel() > 400_000:
        idx = torch.randint(0, v.numel(), (400_000,), device=v.device, generator=_gen(v.device))
        v = v[idx]
    return float(torch.quantile(v.float(), q))
