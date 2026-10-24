"""Image loading, validation and normalisation.

Every analysis starts from an *original* file that is never modified. A
normalised 8-bit RGB working copy is derived from it (alpha removed,
grayscale expanded, 16-bit scaled, optionally down-scaled).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MAX_WORKING_SIDE


class ImageError(ValueError):
    """Raised when an input file is not a usable fundus image."""


@dataclass
class WorkingImage:
    rgb: np.ndarray            # HxWx3 uint8, working resolution
    original_shape: tuple[int, ...]
    original_dtype: str
    original_channels: int
    scale: float               # working = original * scale
    notes: list[str]


def read_any(path: Path | str) -> np.ndarray:
    """Read an image file without changing bit depth or channel count."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII paths
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageError(f"Unreadable or unsupported image file: {path.name}")
    return img


def to_rgb8(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Convert grayscale / BGR / BGRA, 8- or 16-bit, to RGB uint8."""
    notes: list[str] = []
    if img.dtype == np.uint16:
        img = (img / 257.0).round().astype(np.uint8)
        notes.append("16-bit image scaled to 8-bit")
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        notes.append("floating-point image scaled to 8-bit")
    elif img.dtype != np.uint8:
        raise ImageError(f"Unsupported pixel type {img.dtype}")

    if img.ndim == 2:
        notes.append("grayscale image replicated to 3 channels (colour-based steps are less informative)")
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), notes
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        notes.append("alpha channel composited onto black")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), notes
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), notes
    raise ImageError(f"Unsupported image layout {img.shape}")


def load_working_image(path: Path | str, max_side: int = MAX_WORKING_SIDE) -> WorkingImage:
    raw = read_any(path)
    h, w = raw.shape[:2]
    if min(h, w) < 128:
        raise ImageError(f"Image too small for fundus analysis ({w}x{h})")
    rgb, notes = to_rgb8(raw)
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        notes.append(f"working copy down-scaled by {scale:.3f} to {rgb.shape[1]}x{rgb.shape[0]}")
    return WorkingImage(rgb, raw.shape, str(raw.dtype), 1 if raw.ndim == 2 else raw.shape[2], scale, notes)


def load_mask(path: Path | str, shape: tuple[int, int], threshold: int = 127) -> np.ndarray:
    """Load a binary reference mask and resize it (nearest) to ``shape``."""
    m = read_any(path)
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > threshold if m.max() > 1 else m > 0


def to_u8(img: np.ndarray) -> np.ndarray:
    """Visualisable uint8 version of a mask / float map / uint8 image."""
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float64)
    lo, hi = np.nanmin(f), np.nanmax(f)
    if hi - lo < 1e-12:
        return np.zeros(f.shape, np.uint8)
    return ((f - lo) / (hi - lo) * 255).astype(np.uint8)


def save_png(path: Path | str, img: np.ndarray) -> None:
    """Save an RGB / grayscale / mask array as PNG (RGB order expected)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = to_u8(img)
    if u8.ndim == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))
