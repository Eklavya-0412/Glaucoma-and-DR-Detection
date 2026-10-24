"""Module D - classical blood-vessel segmentation and vessel morphometry.

Pipeline (all steps recorded as artifacts):
  1. Green channel extraction (vessels have the highest contrast in green).
  2. Contrast enhancement (CLAHE by default; HE or none for comparison).
  3. Gaussian smoothing (noise reduction).
  4. Multi-scale morphological black-hat (closing - image) with elliptical
     SEs of 7, 11, 15 px (scaled to image size). Vessels are dark, narrow
     structures, so the black-hat responds strongly to them while the slowly
     varying background is removed.
  5. Otsu threshold of the response computed inside the valid retinal field.
  6. Morphological cleanup: removal of connected components smaller than
     ``min_component_area`` pixels.
  7. Final vessel mask; skeletonisation for length/branching measurements.

Parameter choice: SE sizes / smoothing / min-area were selected by comparing
three classical enhancement operators (multi-scale black-hat, line-opening
top-hat, Frangi) on the 20 STARE images; see README for the numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import morphology as M
from .artifacts import ArtifactRecorder, overlay_mask
from .enhancement import enhance_green


@dataclass
class VesselResult:
    mask: np.ndarray
    response: np.ndarray            # continuous enhancement response (for ROC)
    skeleton: np.ndarray
    metrics: dict[str, Any]
    params: dict[str, Any]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)


def vessel_response(green: np.ndarray, valid: np.ndarray, enhancement: str = "clahe",
                    scale: float = 1.0) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    enhanced = enhance_green(green, enhancement, valid)
    sigma = 1.0 * scale
    smooth = cv2.GaussianBlur(enhanced, (0, 0), sigma)
    sizes = [max(3, int(round(s * scale)) | 1) for s in (7, 11, 15)]
    resp = np.mean([M.black_hat(smooth, s).astype(np.float32) for s in sizes], axis=0)
    resp[~valid] = 0
    params = {"enhancement": enhancement, "gaussian_sigma": round(sigma, 2),
              "black_hat_se": [f"ellipse {s}x{s}" for s in sizes], "combination": "mean of black-hat responses"}
    return resp, {"enhanced": enhanced, "smooth": smooth}, params


def segment_vessels(rgb: np.ndarray, valid: np.ndarray, roi_area: int, rec: ArtifactRecorder | None = None,
                    enhancement: str = "clahe", min_component_area: int | None = None) -> VesselResult:
    h, w = rgb.shape[:2]
    scale = max(h, w) / 700.0
    min_component_area = min_component_area or int(round(60 * scale * scale))
    green = rgb[:, :, 1]
    resp, inter, params = vessel_response(green, valid, enhancement, scale)

    vals = resp[valid]
    peak = float(vals.max()) if vals.size else 0.0
    u8 = np.clip(resp / (peak + 1e-9) * 255, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(u8[valid], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) if vals.size else (255, None)
    thresh = (u8 > otsu) & valid
    n_before = M.label_components(thresh)[0] - 1
    cleaned = M.filter_by_area(thresh, min_component_area)
    mask = cleaned
    skel = skeletonize(mask)
    params.update({"threshold_method": "Otsu on normalised response inside valid field",
                   "otsu_threshold_u8": float(otsu), "min_component_area_px": min_component_area})

    metrics = vessel_metrics(mask, skel, roi_area)
    metrics["components_before_cleanup"] = int(n_before)
    if metrics["vessel_density"] is not None and metrics["vessel_density"] < 0.02:
        metrics["warning"] = "very low vessel density - segmentation may have failed (poor contrast?)"

    if rec is not None and rec.enabled:
        rec.stage("vessels", "green_channel", "Green Channel Extraction", green,
                  "Blood vessels absorb green light strongly, so the green channel has the highest "
                  "vessel-to-background contrast.", {"channel": "G of RGB"}, group="vessels")
        rec.stage("vessels", "clahe", {"clahe": "CLAHE — Contrast Limited Adaptive Histogram Equalization",
                                       "he": "Histogram Equalization", "none": "No Enhancement"}[enhancement],
                  inter["enhanced"], "Local contrast enhancement makes thin, low-contrast vessels visible.",
                  {"method": enhancement, "clip_limit": 2.0, "tile_grid": "8x8"} if enhancement == "clahe" else {"method": enhancement},
                  group="vessels")
        rec.stage("vessels", "gaussian", "Gaussian Blur — Noise Reduction", inter["smooth"],
                  "Suppresses pixel noise before the morphological operator.", {"sigma": params["gaussian_sigma"]},
                  group="vessels")
        rec.stage("vessels", "black_hat", "Morphological Black-Hat Transformation (multi-scale)", resp,
                  "closing(I) − I with three elliptical structuring elements. Dark structures narrower than the "
                  "SE (vessels) become bright; smooth background becomes ~0.",
                  {"structuring_elements": params["black_hat_se"]}, group="vessels", filename="enhanced.png")
        rec.stage("vessels", "threshold", "Otsu Thresholding", thresh,
                  "Otsu's method chooses the threshold that best separates vessel and background responses "
                  "inside the retinal field.", {"otsu_threshold": float(otsu), "domain": "valid retinal field"},
                  kind="mask", group="vessels", extra_stats={"components": int(n_before)})
        n, labels, _, _ = M.label_components(cleaned)
        rec.stage("vessels", "connected_components", "Connected-Component Analysis", M.colorize_labels(labels),
                  f"8-connected components after removing those smaller than {min_component_area} px "
                  "(isolated noise). Each colour is one component.",
                  {"connectivity": 8, "min_area_px": min_component_area}, group="vessels",
                  extra_stats={"components_kept": int(n - 1), "components_removed": int(n_before - (n - 1))})
        rec.stage("vessels", "vessel_mask", "Final Vessel Mask", mask,
                  "Binary vessel segmentation used for all vessel measurements.", kind="mask", group="vessels")
        rec.stage("vessels", "skeleton", "Skeletonisation (vessel centrelines)",
                  _skeleton_vis(skel), "One-pixel-wide centrelines (Zhang-Suen style thinning). Red: branch "
                  "points, cyan: end points. Used for length, width and branching measurements.",
                  {"method": "skimage.morphology.skeletonize"}, group="vessels")
        rec.stage("vessels", "vessel_overlay", "Vessel Mask Overlay", overlay_mask(rgb, mask, (40, 220, 120), 0.6),
                  "Predicted vessels (green) drawn on the original image.", kind="overlay", group="vessels")
    return VesselResult(mask, resp, skel, metrics, params, inter)


def _neighbors(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)


def _skeleton_vis(skel: np.ndarray) -> np.ndarray:
    nb = _neighbors(skel)
    vis = np.zeros(skel.shape + (3,), np.uint8)
    vis[skel] = (230, 230, 230)
    bp = skel & (nb >= 3)
    ep = skel & (nb == 1)
    vis[cv2.dilate(bp.astype(np.uint8), np.ones((3, 3))) > 0] = (255, 60, 60)
    vis[cv2.dilate(ep.astype(np.uint8), np.ones((3, 3))) > 0] = (60, 220, 255)
    return vis


def vessel_metrics(mask: np.ndarray, skel: np.ndarray, roi_area: int) -> dict[str, Any]:
    area = int(mask.sum())
    nb = _neighbors(skel)
    # branch points: skeleton pixels with >=3 neighbours, clustered so a junction counts once
    bp_mask = skel & (nb >= 3)
    n_bp = cv2.connectedComponents(bp_mask.astype(np.uint8), connectivity=8)[0] - 1
    n_ep = int((skel & (nb == 1)).sum())
    segments = skel & ~(cv2.dilate(bp_mask.astype(np.uint8), np.ones((3, 3))) > 0)
    n_seg, seg_lab = cv2.connectedComponents(segments.astype(np.uint8), connectivity=8)
    # length: orthogonal steps count 1, diagonal steps sqrt(2) (approximated per pixel)
    s = skel.astype(np.uint8)
    ortho = (s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum()
    diag = (s[1:, 1:] & s[:-1, :-1]).sum() + (s[1:, :-1] & s[:-1, 1:]).sum()
    length = float(ortho + diag * math.sqrt(2)) if skel.any() else 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    widths = 2 * dist[skel] if skel.any() else np.array([])
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in cnts))
    tort = _tortuosity(seg_lab, n_seg)
    return {
        "vessel_area_px": area,
        "vessel_density": round(area / roi_area, 5) if roi_area else None,
        "vessel_length_px": round(length, 1),
        "mean_vessel_width_px": round(area / length, 3) if length > 0 else None,
        "median_centerline_width_px": round(float(np.median(widths)), 3) if widths.size else None,
        "max_vessel_width_px": round(float(widths.max()), 3) if widths.size else None,
        "vessel_perimeter_px": round(perimeter, 1),
        "branch_points": int(n_bp),
        "end_points": n_ep,
        "segment_count": int(n_seg - 1),
        "branching_density_per_10k_px": round(n_bp / roi_area * 1e4, 4) if roi_area else None,
        "mean_tortuosity": tort,
    }


def _tortuosity(seg_lab: np.ndarray, n_seg: int, min_len: int = 20) -> float | None:
    """Mean arc/chord ratio over skeleton segments of at least ``min_len`` pixels."""
    from scipy.ndimage import find_objects
    vals = []
    for i, sl in enumerate(find_objects(seg_lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(seg_lab[sl] == i)
        if ys.size < min_len:
            continue
        # chord = distance between the two farthest-apart endpoints (approx: extreme points along PCA axis)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        c = pts - pts.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        proj = c @ u
        chord = np.linalg.norm(pts[proj.argmax()] - pts[proj.argmin()])
        if chord > 0:
            vals.append(ys.size / chord)
    return round(float(np.mean(vals)), 4) if vals else None
