"""Module C - retinal region-of-interest (field-of-view) extraction.

Pipeline: red channel (brightest channel of a fundus, best separates the
retina from the dark surround) -> Gaussian smoothing -> global threshold
(Otsu, scaled down and bounded) -> morphological opening -> largest
connected component -> hole filling -> erosion to a safe "valid" field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from . import morphology as M
from .artifacts import ArtifactRecorder, overlay_mask


@dataclass
class RoiResult:
    mask: np.ndarray          # full retinal field (bool)
    valid: np.ndarray         # field eroded away from the rim (bool) - used for metrics
    metrics: dict[str, Any]
    params: dict[str, Any]


def extract_roi(rgb: np.ndarray, rec: ArtifactRecorder | None = None, otsu_factor: float = 0.5,
                min_threshold: int = 12, open_size: int | None = None, erode_px: int | None = None) -> RoiResult:
    h, w = rgb.shape[:2]
    scale = max(h, w) / 700.0               # parameters were chosen at STARE resolution
    open_size = open_size or max(5, int(round(9 * scale)))
    erode_px = erode_px or max(3, int(round(6 * scale)))

    red = rgb[:, :, 0]
    smooth = cv2.GaussianBlur(red, (0, 0), 2 * scale)
    otsu, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(max(min_threshold, otsu * otsu_factor))
    binary = smooth > thr
    opened = M.opening(binary, open_size) > 0
    largest = M.largest_component(opened)
    filled = M.fill_holes(largest)
    mask = M.closing(filled, open_size) > 0
    valid = M.erode(mask, 2 * erode_px + 1) > 0

    params = {"channel": "red", "gaussian_sigma": round(2 * scale, 2), "otsu_threshold": float(otsu),
              "otsu_factor": otsu_factor, "min_threshold": min_threshold, "applied_threshold": thr,
              "opening_se": f"ellipse {open_size}x{open_size}", "valid_field_erosion_px": erode_px}

    metrics = roi_metrics(mask)
    metrics["valid_area_px"] = int(valid.sum())
    if metrics["area_px"] < 0.15 * h * w:
        metrics["warning"] = "retinal field smaller than 15% of the image - ROI extraction may have failed"

    if rec is not None and rec.enabled:
        rec.stage("roi", "red_channel", "Red Channel Extraction", red,
                  "The red channel is the brightest fundus channel, giving the strongest contrast between the "
                  "retinal field and the dark camera surround.", {"channel": "R of RGB"})
        rec.stage("roi", "threshold", "Global Thresholding (scaled Otsu)", binary,
                  "Pixels brighter than the threshold are retina candidates. Otsu's threshold is scaled down "
                  "so that dark peripheral retina is not lost.", {k: params[k] for k in
                  ("gaussian_sigma", "otsu_threshold", "otsu_factor", "applied_threshold")}, kind="mask")
        rec.stage("roi", "morph_cleanup", "Opening + Largest Component + Hole Filling", mask,
                  "Opening removes thin bright artefacts (e.g. text, reflections); the largest connected "
                  "component is kept and interior holes (dark lesions, vessels) are filled.",
                  {"opening_se": params["opening_se"], "connectivity": 8}, kind="mask", filename="retinal_mask.png")
        ov = overlay_mask(rgb, mask, (0, 220, 255), outline=True)
        ov = overlay_mask(ov, valid, (255, 200, 0), outline=True)
        cx, cy = metrics["centroid"]
        cv2.drawMarker(ov, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        rec.stage("roi", "roi_overlay", "Retinal ROI Overlay", ov,
                  "Cyan: retinal field boundary. Yellow: eroded valid field used for metric normalisation "
                  "and evaluation (avoids the bright/dark rim). Cross: field centroid.", kind="overlay")
        x, y, bw, bh = metrics["bbox"]
        crop = (rgb * mask[:, :, None])[y:y + bh, x:x + bw]
        rec.stage("roi", "retinal_roi", "Cropped Retinal Field", crop,
                  "The image masked by the retinal field and cropped to its bounding box. Pixels outside the "
                  "retina are set to zero and are never counted as retinal tissue.", kind="image")
    return RoiResult(mask, valid, metrics, params)


def roi_metrics(mask: np.ndarray) -> dict[str, Any]:
    d = M.shape_descriptors(mask)
    if d["area"] == 0:
        return {"area_px": 0}
    return {
        "area_px": d["area"],
        "diameter_px": round(max(d["width"], d["height"]), 2),
        "equivalent_diameter_px": d["equivalent_diameter"],
        "centroid": d["centroid"],
        "perimeter_px": d["perimeter"],
        "circularity": d["circularity"],
        "bbox": d["bbox"],
        "fraction_of_image": round(d["area"] / mask.size, 4),
    }


def quadrant_masks(roi: np.ndarray, center: tuple[float, float]) -> dict[str, np.ndarray]:
    """Split the ROI into four image quadrants about ``center`` (x, y)."""
    h, w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = center
    return {
        "upper_left": roi & (xx < cx) & (yy < cy),
        "upper_right": roi & (xx >= cx) & (yy < cy),
        "lower_left": roi & (xx < cx) & (yy >= cy),
        "lower_right": roi & (xx >= cx) & (yy >= cy),
    }


def field_diameter(roi_metrics_: dict[str, Any]) -> float:
    return float(roi_metrics_.get("equivalent_diameter_px") or 0.0) or math.nan
