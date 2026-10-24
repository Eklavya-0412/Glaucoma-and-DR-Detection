"""Module H - reusable morphological and connected-component operations.

Each operation documents its structuring element (SE) so that pipelines can
record exactly what was applied.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

_SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}


def se(size: int, shape: str = "ellipse") -> np.ndarray:
    size = max(1, int(size) | 1)  # odd
    return cv2.getStructuringElement(_SHAPES[shape], (size, size))


def line_se(length: int, angle_deg: float) -> np.ndarray:
    """Linear structuring element of given length and orientation."""
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))
    cv2.line(k, (round(c - dx * c), round(c - dy * c)), (round(c + dx * c), round(c + dy * c)), 1, 1)
    return k


def _u8(img: np.ndarray) -> np.ndarray:
    return img.astype(np.uint8) * 255 if img.dtype == bool else img


def erode(img, size=3, shape="ellipse", iterations=1):
    return cv2.erode(_u8(img), se(size, shape), iterations=iterations)


def dilate(img, size=3, shape="ellipse", iterations=1):
    return cv2.dilate(_u8(img), se(size, shape), iterations=iterations)


def opening(img, size=3, shape="ellipse"):
    return cv2.morphologyEx(_u8(img), cv2.MORPH_OPEN, se(size, shape))


def closing(img, size=3, shape="ellipse"):
    return cv2.morphologyEx(_u8(img), cv2.MORPH_CLOSE, se(size, shape))


def gradient(img, size=3, shape="ellipse"):
    return cv2.morphologyEx(_u8(img), cv2.MORPH_GRADIENT, se(size, shape))


def top_hat(img, size=15, shape="ellipse"):
    """White top-hat: img - opening(img). Enhances bright details smaller than the SE."""
    return cv2.morphologyEx(_u8(img), cv2.MORPH_TOPHAT, se(size, shape))


def black_hat(img, size=15, shape="ellipse"):
    """Black-hat: closing(img) - img. Enhances dark details smaller than the SE."""
    return cv2.morphologyEx(_u8(img), cv2.MORPH_BLACKHAT, se(size, shape))


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (n_labels incl. background, labels, stats, centroids) from OpenCV."""
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)


def filter_by_area(mask: np.ndarray, min_area: int = 0, max_area: int | None = None,
                   connectivity: int = 8) -> np.ndarray:
    n, labels, stats, _ = label_components(mask, connectivity)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area) & (areas <= (max_area if max_area is not None else np.inf))
    keep[0] = False
    return keep[labels]


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = label_components(mask)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(mask.astype(bool))


def shape_descriptors(mask: np.ndarray) -> dict[str, Any]:
    """Area, perimeter, bbox, centroid and shape factors of a single binary region."""
    m = mask.astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return {"area": 0}
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    x, y, w, h = cv2.boundingRect(cnt)
    mom = cv2.moments(m, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    # eccentricity from second-order central moments
    mu20, mu02, mu11 = mom["mu20"] / area, mom["mu02"] / area, mom["mu11"] / area
    common = math.sqrt(max(0.0, ((mu20 - mu02) / 2) ** 2 + mu11 ** 2))
    l1, l2 = (mu20 + mu02) / 2 + common, (mu20 + mu02) / 2 - common
    ecc = math.sqrt(1 - l2 / l1) if l1 > 1e-9 and l2 >= 0 else 0.0
    return {
        "area": area,
        "perimeter": round(perim, 2),
        "bbox": [int(x), int(y), int(w), int(h)],
        "width": int(w),
        "height": int(h),
        "centroid": [round(cx, 2), round(cy, 2)],
        "equivalent_diameter": round(math.sqrt(4 * area / math.pi), 2),
        "circularity": round(4 * math.pi * area / perim ** 2, 4) if perim > 0 else None,
        "aspect_ratio": round(w / h, 4) if h > 0 else None,
        "eccentricity": round(ecc, 4),
        "solidity": round(area / hull_area, 4) if hull_area > 0 else None,
        "major_axis": round(4 * math.sqrt(l1), 2) if l1 > 0 else None,
        "minor_axis": round(4 * math.sqrt(max(l2, 0)), 2),
    }


def component_table(mask: np.ndarray, intensity: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    """Per-component descriptors (with optional mean intensity in named channels)."""
    n, labels, stats, cents = label_components(mask)
    rows = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        comp = labels[y:y + h, x:x + w] == i
        d = shape_descriptors(comp)
        d["centroid"] = [round(float(cents[i][0]), 2), round(float(cents[i][1]), 2)]
        d["bbox"] = [int(x), int(y), int(w), int(h)]
        d["label"] = i
        if intensity:
            for name, ch in intensity.items():
                d[f"mean_{name}"] = round(float(ch[y:y + h, x:x + w][comp].mean()), 2)
        rows.append(d)
    return rows


def colorize_labels(labels: np.ndarray, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lut = rng.integers(60, 255, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]
