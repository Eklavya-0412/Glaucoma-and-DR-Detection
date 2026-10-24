"""Module B - contrast enhancement: Histogram Equalisation (HE) and CLAHE.

Both are applied to the L* channel of CIELAB so hue is preserved, and to
the green channel (the channel used by the vessel and lesion pipelines).
"""
from __future__ import annotations

import cv2
import numpy as np

CLAHE_CLIP = 2.0
CLAHE_TILES = 8


def he_gray(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Global histogram equalisation; if ``mask`` is given the mapping is
    computed from ROI pixels only (the black surround would otherwise
    dominate the cumulative histogram)."""
    if mask is None:
        return cv2.equalizeHist(gray)
    hist = np.bincount(gray[mask].ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0] if (cdf > 0).any() else 0
    lut = np.clip(np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255), 0, 255).astype(np.uint8)
    out = lut[gray]
    out[~mask] = 0
    return out


def clahe_gray(gray: np.ndarray, clip: float = CLAHE_CLIP, tiles: int = CLAHE_TILES) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def _on_luminance(rgb: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = fn(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def he_rgb(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = _on_luminance(rgb, lambda L: he_gray(L, mask))
    if mask is not None:
        out[~mask] = 0
    return out


def clahe_rgb(rgb: np.ndarray, clip: float = CLAHE_CLIP, tiles: int = CLAHE_TILES,
              mask: np.ndarray | None = None) -> np.ndarray:
    out = _on_luminance(rgb, lambda L: clahe_gray(L, clip, tiles))
    if mask is not None:
        out[~mask] = 0
    return out


def enhance_green(green: np.ndarray, method: str, mask: np.ndarray | None = None,
                  clip: float = CLAHE_CLIP, tiles: int = CLAHE_TILES) -> np.ndarray:
    if method == "none":
        return green
    if method == "he":
        return he_gray(green, mask)
    if method == "clahe":
        return clahe_gray(green, clip, tiles)
    raise ValueError(f"unknown enhancement '{method}'")
