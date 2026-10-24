"""Module J - Gray-Level Co-occurrence Matrix (GLCM) texture features.

Protocol
  * Input: ITU-R BT.601 grayscale of the working image.
  * Quantisation: ROI pixels are mapped linearly to levels 1..(LEVELS-1);
    pixels outside the valid retinal field get level 0.
  * GLCM: distances {1, 2, 4} px, angles {0, 45, 90, 135} degrees,
    symmetric. Row/column 0 (pairs involving background) are removed and the
    matrix is renormalised, so the black surround never contributes.
  * Features per (distance, angle): contrast, homogeneity, energy,
    correlation (skimage definitions) and entropy (-sum p log2 p).
  * Aggregation: mean over all 12 (distance, angle) pairs; per-distance means
    are also reported.

These are texture descriptors of the image; they do not by themselves
indicate disease.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from .artifacts import ArtifactRecorder

LEVELS = 32
DISTANCES = [1, 2, 4]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PROPS = ["contrast", "homogeneity", "energy", "correlation"]


def quantize(gray: np.ndarray, mask: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    q = np.zeros(gray.shape, np.uint8)
    v = gray[mask].astype(np.float64)
    if v.size == 0:
        return q
    lo, hi = np.percentile(v, [0.5, 99.5])
    scaled = np.clip((gray.astype(np.float64) - lo) / max(hi - lo, 1e-9), 0, 1)
    q[mask] = (1 + np.round(scaled[mask] * (levels - 2))).astype(np.uint8)
    return q


def glcm_features(gray: np.ndarray, mask: np.ndarray, rec: ArtifactRecorder | None = None) -> dict[str, Any]:
    q = quantize(gray, mask)
    P = graycomatrix(q, DISTANCES, ANGLES, levels=LEVELS, symmetric=True, normed=False).astype(np.float64)
    P = P[1:, 1:, :, :]
    sums = P.sum(axis=(0, 1), keepdims=True)
    P = P / np.where(sums == 0, 1, sums)
    per = {}
    for prop in PROPS:
        per[prop] = graycoprops(P, prop)          # shape (distances, angles)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(P > 0, P * np.log2(P), 0), axis=(0, 1))
    per["entropy"] = ent
    agg = {k: round(float(v.mean()), 5) for k, v in per.items()}
    by_distance = {str(d): {k: round(float(v[i].mean()), 5) for k, v in per.items()} for i, d in enumerate(DISTANCES)}
    result = {
        "features": agg, "by_distance": by_distance,
        "protocol": {"grayscale": "ITU-R BT.601 luma", "levels": LEVELS,
                     "quantisation": "linear between ROI 0.5/99.5 percentiles to levels 1..31; background = 0 (excluded)",
                     "distances_px": DISTANCES, "angles_deg": [0, 45, 90, 135], "symmetric": True,
                     "aggregation": "mean over all distance/angle pairs", "region": "valid retinal field"},
    }
    if rec is not None and rec.enabled:
        rec.stage("texture", "glcm_input", "GLCM Input — Quantised Grayscale (32 levels)", (q.astype(np.float32) * (255 / (LEVELS - 1))).astype(np.uint8),
                  "Grayscale retina quantised to 31 gray levels (black = excluded background). Co-occurrences of "
                  "these levels between pixel pairs form the GLCM.", result["protocol"])
        rec.stage("texture", "glcm_matrix", "GLCM Heat-map (d = 1 px, mean of 4 angles)", _glcm_plot(P[:, :, 0, :].mean(axis=2)),
                  "Normalised co-occurrence probabilities p(i, j) for neighbouring pixels. Mass near the diagonal "
                  "means smooth texture (high homogeneity, low contrast).", {"distance": 1}, kind="plot")
    return result


def _glcm_plot(P: np.ndarray) -> np.ndarray:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=120)
    im = ax.imshow(np.log10(P + 1e-6), cmap="viridis", origin="upper")
    ax.set_xlabel("gray level j"); ax.set_ylabel("gray level i")
    fig.colorbar(im, ax=ax, label="log10 p(i,j)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
