"""Module G - diabetic-retinopathy lesion *candidate* detection.

Three independent classical pipelines produce candidate masks for
microaneurysms (MA), hemorrhages (HE) and hard exudates (EX). A thresholded
connected component is **not** a confirmed lesion: these are algorithmic
candidates, and the category assignment is an experimental rule based on
size, brightness, colour, shape and context (vessel / optic disc / fovea).

Shared preprocessing: green channel -> CLAHE -> 3x3 median filter (noise)
-> large median filter (background estimate).

G1 MA : background-corrected green -> black-hat (small dark structures) ->
        vessel suppression -> threshold mean + k*std -> size/shape filter.
G2 HE : dark-region enhancement (background - image, larger scale) ->
        threshold -> opening -> components larger than MAs, rejecting
        vessel-like (overlap / elongated) components and the fovea.
G3 EX : CIELAB conversion -> bright-region enhancement (image - background)
        -> threshold -> opening -> components outside the optic disc with
        yellowish colour (b* above the retinal median).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import morphology as M
from .artifacts import ArtifactRecorder, overlay_mask
from .enhancement import clahe_gray

COLORS = {"microaneurysm": (255, 60, 60), "hemorrhage": (170, 0, 255), "exudate": (255, 230, 0)}


@dataclass
class LesionConfig:
    ma_k_std: float = 6.0
    ma_min_area: int = 3
    ma_max_area: int = 60
    ma_min_circularity: float = 0.45
    ma_max_elongation: float = 2.0
    ma_vessel_dilation: int = 7
    rim_margin_px: int = 12
    he_k_std: float = 3.0
    he_max_vessel_overlap: float = 0.4
    he_max_elongation: float = 3.5
    ex_k_std: float = 6.0
    ex_min_area: int = 4


@dataclass
class LesionResult:
    masks: dict[str, np.ndarray]
    components: dict[str, list[dict[str, Any]]]
    summary: dict[str, dict[str, Any]]
    fovea: tuple[float, float] | None
    params: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, np.ndarray] = field(default_factory=dict)


def _thresh(resp: np.ndarray, domain: np.ndarray, k: float) -> float:
    v = resp[domain].astype(np.float64)
    return float(v.mean() + k * v.std()) if v.size else float("inf")


def locate_fovea(gray: np.ndarray, valid: np.ndarray, disc_center, disc_r: float) -> tuple[float, float] | None:
    """Darkest smoothed point 1.8-3.4 disc diameters from the disc, within
    +-1 disc diameter vertically (the macula lies roughly level with the disc)."""
    if disc_center is None:
        return None
    sm = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(3.0, disc_r / 2))
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dd = 2 * disc_r
    d = np.hypot(xx - disc_center[0], yy - disc_center[1])
    region = valid & (d > 1.8 * dd) & (d < 3.4 * dd) & (np.abs(yy - disc_center[1]) < dd)
    if region.sum() < 50:
        return None
    sm[~region] = np.inf
    y, x = np.unravel_index(np.argmin(sm), sm.shape)
    return float(x), float(y)


def detect_lesions(rgb: np.ndarray, valid: np.ndarray, vessel_mask: np.ndarray, disc_mask: np.ndarray | None,
                   disc_center, field_diameter: float, rec: ArtifactRecorder | None = None,
                   cfg: LesionConfig | None = None) -> LesionResult:
    cfg = cfg or LesionConfig()
    h, w = rgb.shape[:2]
    s = max(h, w) / 700.0
    s2 = s * s
    disc_r = field_diameter / 13.0
    green = rgb[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = clahe_gray(green)
    den = cv2.medianBlur(clahe, 3)
    k_bg = int(round(41 * s)) | 1
    bg = cv2.medianBlur(den, k_bg)

    # exclusion regions
    disc_excl = np.zeros((h, w), bool)
    if disc_mask is not None:
        disc_excl = M.dilate(disc_mask, int(disc_r * 0.6) | 1) > 0
    elif disc_center is not None:
        cv2.circle(disc_excl.view(np.uint8), tuple(int(v) for v in disc_center), int(1.5 * disc_r), 1, -1)
    fovea = locate_fovea(gray, valid, disc_center, disc_r)
    fovea_excl = np.zeros((h, w), np.uint8)
    if fovea is not None:
        cv2.circle(fovea_excl, (int(fovea[0]), int(fovea[1])), int(0.8 * disc_r), 1, -1)
    fovea_excl = fovea_excl > 0
    vessel_d = M.dilate(vessel_mask, 3) > 0
    vessel_wide = M.dilate(vessel_mask, int(cfg.ma_vessel_dilation * s) | 1) > 0
    inner = M.erode(valid, 2 * int(cfg.rim_margin_px * s) + 1) > 0   # away from the bright/dark FOV rim
    domain = inner & ~disc_excl

    # ------------------------------------------------------------- G1 MA
    bh_size = int(round(11 * s)) | 1
    ma_resp = M.black_hat(den, bh_size).astype(np.float32)
    ma_resp[vessel_wide | ~domain] = 0
    t_ma = _thresh(ma_resp, domain & ~vessel_wide, cfg.ma_k_std)
    ma_bin = ma_resp > t_ma
    ma_size = M.filter_by_area(ma_bin, max(2, int(cfg.ma_min_area * s2)), int(cfg.ma_max_area * s2))

    def ma_rule(c):
        elong = (c["major_axis"] / c["minor_axis"]) if c.get("minor_axis") else 99
        round_enough = (c.get("circularity") or 0) >= cfg.ma_min_circularity or c["area"] <= 6 * s2
        return round_enough and elong <= cfg.ma_max_elongation
    ma_mask, ma_rejected = _shape_filter(ma_size, ma_rule)

    # ------------------------------------------------------------- G2 HE
    k_bg2 = int(round(71 * s)) | 1
    bg2 = cv2.medianBlur(den, k_bg2)
    he_resp = np.clip(bg2.astype(np.float32) - den.astype(np.float32), 0, None)
    he_resp[~domain] = 0
    t_he = _thresh(he_resp, domain & ~vessel_d, cfg.he_k_std)
    he_bin = M.opening(he_resp > t_he, 3) > 0
    he_size = M.filter_by_area(he_bin, int(cfg.ma_max_area * s2) + 1)

    def he_rule(c, region):
        ys, xs, sub = region
        overlap = float(vessel_d[ys, xs][sub].mean())
        c["vessel_overlap"] = round(overlap, 3)
        elong = (c["major_axis"] / c["minor_axis"]) if c.get("minor_axis") else 99
        c["elongation"] = round(elong, 2)
        in_fovea = bool(fovea_excl[int(c["centroid"][1]), int(c["centroid"][0])])
        c["in_fovea_region"] = in_fovea
        return overlap <= cfg.he_max_vessel_overlap and elong <= cfg.he_max_elongation and not in_fovea
    he_mask, he_rejected = _shape_filter(he_size, he_rule, pass_mask=True)

    # ------------------------------------------------------------- G3 EX
    ex_resp = np.clip(den.astype(np.float32) - bg2.astype(np.float32), 0, None)
    ex_resp[~domain] = 0
    t_ex = _thresh(ex_resp, domain, cfg.ex_k_std)
    ex_bin = M.opening(ex_resp > t_ex, 3) > 0
    ex_size = M.filter_by_area(ex_bin, max(2, int(cfg.ex_min_area * s2)))
    b_med = float(np.median(lab[:, :, 2][valid]))
    L_med = float(np.median(lab[:, :, 0][valid]))
    ex_mask, ex_rejected = _shape_filter(
        ex_size, lambda c, r: bool(lab[r[0], r[1], 2][r[2]].mean() >= b_med and lab[r[0], r[1], 0][r[2]].mean() > L_med),
        pass_mask=True)

    masks = {"microaneurysm": ma_mask, "hemorrhage": he_mask, "exudate": ex_mask}
    intensity = {"gray": gray, "L": lab[:, :, 0], "a": lab[:, :, 1], "b": lab[:, :, 2],
                 "R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}
    comps = {k: M.component_table(m, intensity) for k, m in masks.items()}
    params = {
        "shared": {"channel": "green", "clahe": "clip 2.0, 8x8 tiles", "median_denoise": "3x3",
                   "background_estimate": f"median filter {k_bg}x{k_bg}",
                   "exclusions": f"optic disc (dilated), fovea region (HE only), {int(cfg.rim_margin_px * s)}px "
                                 "margin inside the retinal field rim"},
        "microaneurysm": {"black_hat_se": f"ellipse {bh_size}x{bh_size}",
                          "vessel_suppression": f"vessel mask dilated {int(cfg.ma_vessel_dilation * s) | 1}px",
                          "max_elongation": cfg.ma_max_elongation,
                          "threshold": f"mean + {cfg.ma_k_std} std = {t_ma:.2f}",
                          "area_range_px": [max(2, int(cfg.ma_min_area * s2)), int(cfg.ma_max_area * s2)],
                          "min_circularity": cfg.ma_min_circularity},
        "hemorrhage": {"dark_enhancement": f"median background {k_bg2}x{k_bg2} - image",
                       "threshold": f"mean + {cfg.he_k_std} std = {t_he:.2f}", "opening": "ellipse 3x3",
                       "min_area_px": int(cfg.ma_max_area * s2) + 1, "max_vessel_overlap": cfg.he_max_vessel_overlap,
                       "max_elongation": cfg.he_max_elongation},
        "exudate": {"colour_space": "CIELAB", "bright_enhancement": f"image - median background {k_bg2}x{k_bg2}",
                    "threshold": f"mean + {cfg.ex_k_std} std = {t_ex:.2f}", "opening": "ellipse 3x3",
                    "min_area_px": max(2, int(cfg.ex_min_area * s2)),
                    "colour_rule": f"mean b* >= retinal median ({b_med:.1f}) and mean L* > retinal median ({L_med:.1f})"},
        "rejected_components": {"microaneurysm": ma_rejected, "hemorrhage": he_rejected, "exudate": ex_rejected},
    }
    roi_area = int(valid.sum())
    summary = {k: summarize(comps[k], roi_area, disc_center) for k in masks}
    result = LesionResult(masks, comps, summary, fovea, params,
                          {"microaneurysm": ma_resp, "hemorrhage": he_resp, "exudate": ex_resp})

    if rec is not None and rec.enabled:
        sh = "Shared"
        rec.stage("lesions", "green_clahe", "Green Channel + CLAHE", clahe,
                  "Lesions are most visible in the green channel; CLAHE increases local contrast.",
                  {"clip_limit": 2.0, "tile_grid": "8x8"}, group="shared")
        rec.stage("lesions", "median_denoise", "Median Filter — Noise Reduction", den,
                  "A 3x3 median filter removes impulse noise while preserving small lesion edges.", {"kernel": "3x3"},
                  group="shared")
        rec.stage("lesions", "background", "Background Estimation (large median filter)", bg,
                  "A large median filter removes all structures smaller than the kernel, leaving the slowly "
                  "varying illumination/background.", {"kernel": f"{k_bg}x{k_bg}"}, group="shared")
        ctx = rgb.copy()
        for m, col in ((disc_excl, (40, 255, 120)), (fovea_excl, (0, 200, 255))):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ctx, cnts, -1, col, 2)
        rec.stage("lesions", "context", "Anatomical Context: Optic Disc & Fovea Exclusion", ctx,
                  "Green: dilated optic disc (excluded from all lesion searches). Blue: estimated fovea "
                  "(darkest region 1.8-3.4 disc diameters from the disc); excluded from hemorrhage candidates.",
                  {"fovea_xy": [round(v, 1) for v in fovea] if fovea else "not found"}, kind="overlay", group="shared")
        # G1
        rec.stage("lesions", "ma_black_hat", "Morphological Black-Hat (vessels suppressed)", ma_resp,
                  "closing − image with a small SE highlights small dark spots; pixels on/near detected vessels are "
                  "zeroed so vessel segments are not taken as microaneurysms.", params["microaneurysm"],
                  group="microaneurysm")
        rec.stage("lesions", "ma_threshold", "Thresholding (mean + k·std)", ma_bin,
                  "Statistical threshold on the black-hat response within the retina.",
                  {"threshold": params["microaneurysm"]["threshold"]}, kind="mask", group="microaneurysm")
        rec.stage("lesions", "microaneurysm_mask", "Microaneurysm Candidates (size + circularity filter)", ma_mask,
                  "Connected components within the MA size range, sufficiently round and not elongated.",
                  {"area_range_px": params["microaneurysm"]["area_range_px"],
                   "min_circularity": cfg.ma_min_circularity, "max_elongation": cfg.ma_max_elongation}, kind="mask", group="microaneurysm",
                  extra_stats={"candidates": len(comps["microaneurysm"])})
        # G2
        rec.stage("lesions", "he_dark_enhancement", "Dark-Region Enhancement (background − image)", he_resp,
                  "Subtracting the image from a large-scale background makes regions darker than their "
                  "surroundings (blot/dot hemorrhages, vessels, fovea) bright.",
                  {"background": params["hemorrhage"]["dark_enhancement"]}, group="hemorrhage")
        rec.stage("lesions", "he_threshold", "Thresholding + Morphological Opening", he_bin,
                  "Threshold mean + k·std, then opening (3x3 ellipse) to detach thin vessel bridges.",
                  {"threshold": params["hemorrhage"]["threshold"]}, kind="mask", group="hemorrhage")
        rec.stage("lesions", "hemorrhage_mask", "Hemorrhage Candidates (vessel / fovea / shape rules)", he_mask,
                  "Components larger than MAs, overlapping vessels by at most 40%, not strongly elongated, and not "
                  "in the fovea region.", {k: params["hemorrhage"][k] for k in ("min_area_px", "max_vessel_overlap",
                                                                                "max_elongation")},
                  kind="mask", group="hemorrhage", extra_stats={"candidates": len(comps["hemorrhage"])})
        # G3
        rec.stage("lesions", "ex_lab_b", "CIELAB b* Channel (yellowness)", lab[:, :, 2],
                  "Colour-space conversion: b* separates yellow (exudates) from the orange-red retina.",
                  {"colour_space": "CIELAB"}, group="exudate")
        rec.stage("lesions", "ex_bright_enhancement", "Bright-Region Enhancement (image − background)", ex_resp,
                  "Structures brighter than the local background (exudates, reflections, disc rim) become bright.",
                  {"background": params["exudate"]["bright_enhancement"]}, group="exudate")
        rec.stage("lesions", "ex_threshold", "Thresholding + Morphological Opening", ex_bin,
                  "Threshold mean + k·std then opening (3x3 ellipse).", {"threshold": params["exudate"]["threshold"]},
                  kind="mask", group="exudate")
        rec.stage("lesions", "exudate_mask", "Hard-Exudate Candidates (colour rule)", ex_mask,
                  "Components outside the disc whose mean colour is yellowish (b*) and bright (L*).",
                  {"colour_rule": params["exudate"]["colour_rule"]}, kind="mask", group="exudate",
                  extra_stats={"candidates": len(comps["exudate"])})
        ov = rgb.copy()
        for k, m in masks.items():
            ov = overlay_mask(ov, M.dilate(m, 3) > 0 if k == "microaneurysm" else m, COLORS[k], 0.75)
            for c in comps[k]:
                if k == "microaneurysm":
                    cv2.circle(ov, (int(c["centroid"][0]), int(c["centroid"][1])), int(6 * s), COLORS[k], 1)
        rec.stage("lesions", "lesion_overlay", "All Lesion Candidates Overlay", ov,
                  "Red circles: microaneurysm candidates. Purple: hemorrhage candidates. Yellow: hard-exudate "
                  "candidates.", kind="overlay", group="summary")
    return result


def _shape_filter(mask: np.ndarray, rule, pass_mask: bool = False) -> tuple[np.ndarray, int]:
    n, labels, stats, cents = M.label_components(mask)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        sub = labels[y:y + h, x:x + w] == i
        d = M.shape_descriptors(sub)
        d["centroid"] = [float(cents[i][0]), float(cents[i][1])]
        if pass_mask:
            # region accessor: (row slice, col slice, local mask) - avoids full-image masks per component
            keep[i] = rule(d, (slice(y, y + h), slice(x, x + w), sub))
        else:
            keep[i] = rule(d)
    return keep[labels], int((n - 1) - keep[1:].sum())


def summarize(comps: list[dict[str, Any]], roi_area: int, disc_center) -> dict[str, Any]:
    if not comps:
        return {"count": 0, "total_area_px": 0, "density": 0.0 if roi_area else None,
                "mean_area_px": None, "max_area_px": None}
    areas = np.array([c["area"] for c in comps], float)

    def mean_of(key):
        vals = [c[key] for c in comps if c.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    out = {
        "count": len(comps), "total_area_px": int(areas.sum()),
        "density": round(float(areas.sum() / roi_area), 6) if roi_area else None,
        "mean_area_px": round(float(areas.mean()), 2), "max_area_px": int(areas.max()),
        "total_perimeter_px": round(float(sum(c.get("perimeter", 0) for c in comps)), 1),
        "mean_perimeter_px": mean_of("perimeter"), "mean_equivalent_diameter_px": mean_of("equivalent_diameter"),
        "mean_circularity": mean_of("circularity"), "mean_aspect_ratio": mean_of("aspect_ratio"),
        "mean_eccentricity": mean_of("eccentricity"), "mean_solidity": mean_of("solidity"),
        "mean_brightness_L": mean_of("mean_L"), "mean_rgb": [mean_of("mean_R"), mean_of("mean_G"), mean_of("mean_B")],
        "mean_b_star": mean_of("mean_b"),
    }
    if disc_center is not None:
        dists = [math.dist(c["centroid"], disc_center) for c in comps]
        for c, d in zip(comps, dists):
            c["distance_to_disc_px"] = round(d, 1)
        out.update({"mean_distance_to_disc_px": round(float(np.mean(dists)), 1),
                    "min_distance_to_disc_px": round(float(np.min(dists)), 1),
                    "max_distance_to_disc_px": round(float(np.max(dists)), 1)})
    return out
