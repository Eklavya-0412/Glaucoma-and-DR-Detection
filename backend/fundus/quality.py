"""Module A - image-quality measurements.

These are *image* measurements (brightness, contrast, information content,
sharpness, noise). They are not clinical findings. Unless stated otherwise
every metric is computed on the luminance (ITU-R BT.601 grayscale) pixels
inside the valid retinal field.
"""
from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np

from .edges import canny

METRIC_DOCS = {
    "mean_intensity": "Mean gray level of ROI pixels (0-255).",
    "median_intensity": "Median gray level of ROI pixels.",
    "min_intensity": "Minimum gray level in the ROI.",
    "max_intensity": "Maximum gray level in the ROI.",
    "std_intensity": "Standard deviation of ROI gray levels.",
    "intensity_range": "max - min of ROI gray levels.",
    "global_contrast_michelson": "(P99 - P1) / (P99 + P1) using the 1st/99th percentiles (robust Michelson contrast).",
    "rms_contrast": "Standard deviation of ROI gray levels normalised to [0,1] (RMS contrast). Reported as 'Contrast'.",
    "local_contrast": "Mean of the local standard deviation in a 9x9 window, normalised to [0,1].",
    "dark_pixel_pct": "Percentage of ROI pixels with gray level < 25.",
    "bright_pixel_pct": "Percentage of ROI pixels with gray level > 230.",
    "entropy_bits": "Shannon entropy of the 256-bin ROI histogram, in bits.",
    "sharpness_laplacian_var": "Variance of the Laplacian inside the ROI (higher = sharper).",
    "edge_density": "Fraction of ROI pixels marked as edges by Canny (auto thresholds).",
    "noise_sigma": "Immerkaer fast noise-sigma estimate (gray levels), computed inside the ROI.",
}

CORE_QUALITY = ["mean_intensity", "std_intensity", "rms_contrast", "entropy_bits", "edge_density"]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def entropy(values: np.ndarray) -> float:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def immerkaer_sigma(gray: np.ndarray, mask: np.ndarray) -> float:
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, k)
    vals = np.abs(conv[mask])
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.pi / 2) * vals.mean() / 6.0)


def quality_metrics(gray: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    mask = np.ones(gray.shape, bool) if mask is None else mask.astype(bool)
    v = gray[mask]
    if v.size == 0:
        return {k: None for k in METRIC_DOCS}
    vf = v.astype(np.float64)
    p1, p99 = np.percentile(vf, [1, 99])
    g = gray.astype(np.float64) / 255.0
    mu = cv2.blur(g, (9, 9))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mu * mu, 0))
    edges, _ = canny(gray)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # ignore the FOV rim edge
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "mean_intensity": round(float(vf.mean()), 3),
        "median_intensity": round(float(np.median(vf)), 3),
        "min_intensity": int(v.min()),
        "max_intensity": int(v.max()),
        "std_intensity": round(float(vf.std()), 3),
        "intensity_range": int(v.max()) - int(v.min()),
        "global_contrast_michelson": round(float((p99 - p1) / (p99 + p1)), 4) if p99 + p1 > 0 else None,
        "rms_contrast": round(float((vf / 255.0).std()), 4),
        "local_contrast": round(float(sd[mask].mean()), 4),
        "dark_pixel_pct": round(float((v < 25).mean() * 100), 3),
        "bright_pixel_pct": round(float((v > 230).mean() * 100), 3),
        "entropy_bits": round(entropy(v), 4),
        "sharpness_laplacian_var": round(float(lap[inner].var()), 3) if inner.any() else None,
        "edge_density": round(float((edges[inner] > 0).mean()), 5) if inner.any() else None,
        "noise_sigma": round(immerkaer_sigma(gray, inner if inner.any() else mask), 3),
    }


def histogram_plot(channels: dict[str, np.ndarray], mask: np.ndarray, title: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Gray": "#333333", "Red": "#d1495b", "Green": "#2a9d8f", "Blue": "#3d5a80",
              "Original": "#333333", "HE": "#e07a5f", "CLAHE": "#3d5a80"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=120)
    for name, ch in channels.items():
        hist = np.bincount(ch[mask].ravel(), minlength=256)
        ax.plot(hist / max(hist.sum(), 1), color=colors.get(name, None), lw=1.4, label=name)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Gray level")
    ax.set_ylabel("Fraction of ROI pixels")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
