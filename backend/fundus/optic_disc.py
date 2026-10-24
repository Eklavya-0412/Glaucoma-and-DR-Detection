"""Modules E & F - optic-disc localisation/segmentation, optic-cup
segmentation and cup-to-disc geometry.

Disc localisation
  Gray image -> morphological closing (removes dark vessels crossing the
  disc) -> Gaussian blur at disc scale -> brightest point inside the valid
  field. The five strongest separated peaks are kept as candidates.

Disc segmentation (inside a square window around the chosen candidate)
  Luminance L* (the red channel is often saturated around the disc) ->
  closing to remove vessels -> Gaussian smoothing -> recursive Otsu
  threshold within the window ->
  component containing / nearest to the candidate -> hole filling ->
  ellipse fit (boundary refinement). The result is rejected if its area is
  implausible relative to the retinal field.

Cup segmentation (independent threshold, inside the disc only)
  Green channel (cup pallor contrast is highest in green) -> closing to
  remove vessels -> smoothing -> threshold (Otsu inside the disc, or a
  configurable percentile) -> component nearest the disc centre -> hole
  filling -> ellipse fit -> clipped to the disc.

Limitations: purely intensity-based. Peripapillary atrophy, bright lesions
near the disc, and low-contrast / blurred images can produce wrong
boundaries. The cup is particularly hard to delineate from colour alone
(the clinical cup boundary is defined by depth / vessel bending).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import morphology as M
from .artifacts import ArtifactRecorder, labelled_panel
from .edges import canny, sobel


@dataclass
class DiscCupResult:
    disc_center: tuple[float, float] | None
    disc_mask: np.ndarray | None
    cup_mask: np.ndarray | None
    candidates: list[dict[str, float]]
    disc: dict[str, Any] = field(default_factory=dict)
    cup: dict[str, Any] = field(default_factory=dict)
    ratios: dict[str, Any] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


def _z(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = a[m]
    return (a - v.mean()) / (v.std() + 1e-9)


def localize_disc(rgb: np.ndarray, valid: np.ndarray, field_diameter: float, n_candidates: int = 5):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(5, int(field_diameter / 43) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, M.se(k))
    sigma = field_diameter / 40
    bright = cv2.GaussianBlur(closed, (0, 0), sigma)
    inner = cv2.erode(valid.astype(np.uint8), M.se(k)) > 0
    score = _z(bright, inner)
    score[~inner] = -np.inf
    s = score.copy()
    cands = []
    for _ in range(n_candidates):
        y, x = np.unravel_index(np.argmax(s), s.shape)
        if not np.isfinite(s[y, x]):
            break
        cands.append({"x": float(x), "y": float(y), "score": round(float(s[y, x]), 3)})
        cv2.circle(s, (int(x), int(y)), int(field_diameter / 6), -np.inf, -1)
    params = {"closing_se": f"ellipse {k}x{k}", "gaussian_sigma": round(sigma, 2),
              "rule": "maximum of vessel-free, disc-scale smoothed brightness"}
    return cands, bright, score, params


def _fit_ellipse_mask(region: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(region.shape, np.uint8)
    if not cnts:
        return out > 0
    c = max(cnts, key=cv2.contourArea)
    if len(c) >= 5:
        cv2.ellipse(out, cv2.fitEllipse(c), 1, -1)
    else:
        cv2.drawContours(out, [c], -1, 1, -1)
    return out > 0


def _pick_component(binary: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    n, labels, stats, cents = M.label_components(binary)
    if n <= 1:
        return np.zeros(binary.shape, bool)
    x, y = point
    lab = labels[y, x] if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] else 0
    if lab == 0:
        d = [math.dist(point, cents[i]) - math.sqrt(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
        lab = 1 + int(np.argmin(d))
    return labels == lab


def _segment_disc_at(rgb, valid, cx, cy, expected_r):
    """Segment a disc-like bright region in a window centred on (cx, cy)."""
    h, w = rgb.shape[:2]
    half = int(round(2.4 * expected_r))
    x0, y0, x1, y1 = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
    win_valid = valid[y0:y1, x0:x1]
    vk = max(5, int(expected_r / 3.5) | 1)
    # Luminance (L* of CIELAB): the red channel is frequently saturated around
    # the disc in STARE, which destroys the disc boundary.
    lum = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2LAB)[:, :, 0]
    disc_ch = cv2.GaussianBlur(cv2.morphologyEx(lum, cv2.MORPH_CLOSE, M.se(vk)), (0, 0), 2)
    expected_area = math.pi * expected_r ** 2
    # Recursive Otsu: threshold the window; while the selected region is far
    # larger than a plausible disc, re-apply Otsu to the bright class only.
    pool = disc_ch[win_valid]
    thresholds: list[float] = []
    comp = np.zeros(disc_ch.shape, bool)
    disc_bin = comp
    for _ in range(4):
        if pool.size < 50:
            break
        t, _ = cv2.threshold(pool.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.append(float(t))
        disc_bin = M.opening((disc_ch > t) & win_valid, vk) > 0
        comp = M.fill_holes(_pick_component(disc_bin, (cx - x0, cy - y0)))
        if comp.sum() <= 2.0 * expected_area:
            break
        pool = pool[pool > t]
    disc_local = _fit_ellipse_mask(comp) & win_valid
    area_ratio = float(disc_local.sum() / expected_area)
    d = M.shape_descriptors(disc_local)
    elong = (d["major_axis"] / d["minor_axis"]) if d.get("minor_axis") else float("inf")
    ok = 0.25 <= area_ratio <= 3.0 and elong <= 1.7
    reason = "" if ok else (f"region is {area_ratio:.2f}x the expected disc area and has elongation "
                            f"{elong:.2f} (accepted: 0.25-3.0x, elongation <= 1.7)")
    return {"window": (x0, y0, x1, y1), "vk": vk, "disc_ch": disc_ch, "disc_bin": disc_bin,
            "disc_local": disc_local, "thresholds": thresholds, "area_ratio": area_ratio,
            "elongation": elong, "ok": ok, "reason": reason}


def segment_disc_cup(rgb: np.ndarray, valid: np.ndarray, field_diameter: float,
                     rec: ArtifactRecorder | None = None, cup_method: str = "otsu",
                     cup_percentile: float = 80.0) -> DiscCupResult:
    h, w = rgb.shape[:2]
    cands, bright, score, loc_params = localize_disc(rgb, valid, field_diameter)
    res = DiscCupResult(None, None, None, cands, params={"localisation": loc_params})
    if not cands:
        res.status = {"disc": "failed: no candidate inside retinal field", "cup": "unavailable: no disc"}
        return res
    expected_r = field_diameter / 13.0        # typical disc radius relative to a 45-50 deg field
    # Candidate verification: candidates are tried in brightness order and the
    # first one whose segmented region is disc-shaped (plausible area, not
    # elongated) is accepted. Bright lesions usually fail this shape test.
    seg, chosen = None, 0
    for i, c in enumerate(cands):
        trial = _segment_disc_at(rgb, valid, int(c["x"]), int(c["y"]), expected_r)
        c["disc_shape_ok"] = trial["ok"]
        if seg is None:
            seg = trial
        if trial["ok"]:
            seg, chosen = trial, i
            break
    res.params["localisation"]["selected_candidate"] = chosen + 1
    res.params["localisation"]["verification"] = ("first brightness candidate whose segmentation is disc-shaped "
                                                  "(area 0.25-3x expected, elongation <= 1.7)")
    cx, cy = int(cands[chosen]["x"]), int(cands[chosen]["y"])
    res.disc_center = (float(cx), float(cy))
    x0, y0, x1, y1 = seg["window"]
    vk, disc_bin, disc_local, red_c = seg["vk"], seg["disc_bin"], seg["disc_local"], seg["disc_ch"]
    thresholds = seg["thresholds"]
    res.params["disc"] = {"window": [x0, y0, x1 - x0, y1 - y0], "channel": "luminance L*",
                          "vessel_removal_closing_se": f"ellipse {vk}x{vk}",
                          "threshold": "recursive Otsu in window (re-thresholded while region > 2x expected disc area)",
                          "otsu_thresholds": thresholds,
                          "boundary_refinement": "cv2.fitEllipse",
                          "plausibility": "area 0.25-3.0 x expected disc area, elongation <= 1.7",
                          "expected_disc_radius_px": round(expected_r, 1),
                          "area_ratio": round(seg["area_ratio"], 3), "elongation": round(seg["elongation"], 3)}
    disc_ok = seg["ok"]
    disc_mask = np.zeros((h, w), bool)
    if disc_ok:
        disc_mask[y0:y1, x0:x1] = disc_local
        res.disc_mask = disc_mask
        res.disc = M.shape_descriptors(disc_mask)
        res.disc["radius"] = round(res.disc["equivalent_diameter"] / 2, 2)
        res.disc["diameter"] = res.disc["equivalent_diameter"]
        res.disc_center = tuple(res.disc["centroid"])
        res.status["disc"] = "segmented"
    else:
        res.status["disc"] = (f"failed: no candidate produced a disc-shaped region ({seg['reason']}); "
                              "disc centre from brightness localisation only")

    # ------------------------------------------------------------------ cup
    green = rgb[y0:y1, x0:x1, 1]
    green_c = cv2.GaussianBlur(cv2.morphologyEx(green, cv2.MORPH_CLOSE, M.se(vk)), (0, 0), 1.5)
    cup_local = np.zeros_like(disc_local)
    if disc_ok:
        vals = green_c[disc_local]
        if cup_method == "percentile":
            t_cup = float(np.percentile(vals, cup_percentile))
        else:
            t_cup, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cup_bin = (green_c > t_cup) & disc_local
        dcx, dcy = int(res.disc["centroid"][0] - x0), int(res.disc["centroid"][1] - y0)
        cup_comp = M.fill_holes(_pick_component(cup_bin, (dcx, dcy)))
        cup_local = _fit_ellipse_mask(cup_comp) & disc_local
        res.params["cup"] = {"channel": "green", "vessel_removal_closing_se": f"ellipse {vk}x{vk}",
                             "method": cup_method, "threshold": round(float(t_cup), 2),
                             **({"percentile": cup_percentile} if cup_method == "percentile" else {}),
                             "constraint": "cup clipped to disc"}
        if cup_local.sum() < 0.03 * disc_local.sum():
            res.status["cup"] = "failed: cup region < 3% of disc area"
        else:
            cup_mask = np.zeros((h, w), bool)
            cup_mask[y0:y1, x0:x1] = cup_local
            res.cup_mask = cup_mask
            res.cup = M.shape_descriptors(cup_mask)
            res.status["cup"] = "segmented"
    else:
        res.status["cup"] = "unavailable: disc segmentation failed"

    res.ratios = cdr_features(res.disc if res.disc_mask is not None else None,
                              res.cup if res.cup_mask is not None else None)
    res.ratios.update(rim_thickness(res.disc_mask, res.cup_mask, res.disc_center, field_center=None))

    if rec is not None and rec.enabled:
        _record(rec, rgb, bright, cands, (x0, y0, x1, y1), red_c, disc_bin, disc_local, green_c,
                cup_local, res)
    return res


def cdr_features(disc: dict[str, Any] | None, cup: dict[str, Any] | None) -> dict[str, Any]:
    """VCDR = cup height / disc height, HCDR = cup width / disc width,
    ACDR = cup area / disc area, rim area = disc area - cup area."""
    out: dict[str, Any] = {k: None for k in ("vcdr", "hcdr", "acdr", "rim_area", "rim_to_disc_area_ratio")}
    if not disc or not disc.get("area"):
        return out
    if not cup or not cup.get("area"):
        return out
    out["vcdr"] = round(cup["height"] / disc["height"], 4) if disc["height"] else None
    out["hcdr"] = round(cup["width"] / disc["width"], 4) if disc["width"] else None
    out["acdr"] = round(cup["area"] / disc["area"], 4)
    out["rim_area"] = int(disc["area"] - cup["area"])
    out["rim_to_disc_area_ratio"] = round(out["rim_area"] / disc["area"], 4)
    return out


def rim_thickness(disc_mask, cup_mask, center, field_center=None) -> dict[str, Any]:
    """Rim thickness along the four image directions through the disc centroid.

    Superior/inferior are image up/down. Nasal/temporal require knowing
    whether this is a left or right eye, which is not available in the
    datasets, so only image left/right thickness is reported and the ISNT
    nasal/temporal values are marked unavailable.
    """
    keys = ["rim_superior_px", "rim_inferior_px", "rim_left_px", "rim_right_px"]
    out: dict[str, Any] = {k: None for k in keys}
    out["rim_nasal_px"] = out["rim_temporal_px"] = None
    out["isnt_note"] = "nasal/temporal unavailable: eye laterality not provided"
    if disc_mask is None or cup_mask is None or center is None:
        return out
    cx, cy = int(round(center[0])), int(round(center[1]))
    col_d, col_c = disc_mask[:, cx], cup_mask[:, cx]
    row_d, row_c = disc_mask[cy, :], cup_mask[cy, :]

    def span(v):
        idx = np.nonzero(v)[0]
        return (idx.min(), idx.max()) if idx.size else (None, None)

    (dt, db), (ct, cb) = span(col_d), span(col_c)
    (dl, dr), (cl, cr) = span(row_d), span(row_c)
    if None not in (dt, ct):
        out["rim_superior_px"], out["rim_inferior_px"] = int(ct - dt), int(db - cb)
    if None not in (dl, cl):
        out["rim_left_px"], out["rim_right_px"] = int(cl - dl), int(dr - cr)
    return out


def _record(rec, rgb, bright, cands, win, red_c, disc_bin, disc_local, green_c, cup_local, res):
    x0, y0, x1, y1 = win
    cand_vis = rgb.copy()
    sel = res.params["localisation"].get("selected_candidate", 1) - 1
    for i, c in enumerate(cands):
        col = (255, 230, 0) if i == sel else (255, 255, 255)
        cv2.circle(cand_vis, (int(c["x"]), int(c["y"])), 10 if i == sel else 6, col, 2)
        cv2.putText(cand_vis, str(i + 1), (int(c["x"]) + 10, int(c["y"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.rectangle(cand_vis, (x0, y0), (x1, y1), (255, 230, 0), 1)
    rec.stage("glaucoma", "brightness_map", "Vessel-Free Brightness Map (closing + Gaussian)", bright,
              "Morphological closing removes dark vessels; heavy Gaussian blur at disc scale turns the bright "
              "optic disc into a smooth peak.", res.params["localisation"], group="disc")
    rec.stage("glaucoma", "disc_candidate", "Optic-Disc Candidate Localisation", cand_vis,
              "Strongest separated brightness peaks, numbered by brightness. Yellow = selected (first candidate "
              "whose segmentation is disc-shaped). The square is the window used for disc and cup segmentation.",
              {"candidates": len(cands), "selected": sel + 1}, kind="overlay", group="disc")
    rec.stage("glaucoma", "disc_channel", f"Disc Window — {res.params['disc']['channel'].split(' ')[0].title()} "
              "Channel after Vessel Removal", red_c,
              "The disc boundary has the strongest contrast in the red channel; closing removes the vessels "
              "that cross the disc.", {"closing_se": res.params["disc"]["vessel_removal_closing_se"]}, group="disc")
    rec.stage("glaucoma", "disc_threshold", "Recursive Otsu Thresholding (disc window)", disc_bin,
              "Bright pixels in the window; the component containing the candidate becomes the disc. Otsu is "
              "re-applied to the bright class while the region is implausibly large.",
              {"otsu_thresholds": res.params["disc"]["otsu_thresholds"]}, kind="mask", group="disc")
    e, ep = canny(red_c)
    s, sp = sobel(red_c)
    rec.stage("glaucoma", "disc_edges", "Edge Maps for Boundary Inspection (Sobel | Canny)",
              labelled_panel([("Sobel gradient magnitude", s), ("Canny edges", e)], tile_width=260),
              "Edge detectors applied to the vessel-free disc window, used to visually check that the fitted "
              "disc boundary follows the strongest intensity transition.", {"sobel": sp, "canny": ep},
              kind="comparison", group="disc")
    if res.disc_mask is not None:
        rec.stage("glaucoma", "disc_mask", "Predicted Optic-Disc Mask (ellipse-refined)", res.disc_mask,
                  "Selected component after hole filling and least-squares ellipse fitting.", kind="mask",
                  group="disc")
        rec.stage("glaucoma", "cup_channel", "Green Channel inside Disc (vessel-removed)", green_c,
                  "The cup (central pallor) is brightest in the green channel.",
                  {"closing_se": res.params["cup"]["vessel_removal_closing_se"]}, group="cup")
        cup_cand = np.zeros(green_c.shape + (3,), np.uint8)
        cup_cand[disc_local] = (90, 90, 90)
        t = res.params["cup"]["threshold"]
        cup_cand[(green_c > t) & disc_local] = (255, 210, 0)
        rec.stage("glaucoma", "cup_candidate", "Candidate Cup Region (thresholded pallor)", cup_cand,
                  "Pixels inside the disc brighter than the cup threshold (yellow); grey is the remaining disc.",
                  {"method": res.params["cup"]["method"], "threshold": t}, kind="mask", group="cup")
        if res.cup_mask is not None:
            rec.stage("glaucoma", "cup_mask", "Predicted Optic-Cup Mask", res.cup_mask,
                      "Component nearest the disc centre, hole-filled, ellipse-fitted and clipped to the disc.",
                      kind="mask", group="cup")
    ov = rgb.copy()
    for mask, col in ((res.disc_mask, (40, 255, 120)), (res.cup_mask, (255, 220, 0))):
        if mask is not None:
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, col, 2)
    if res.disc_center:
        cv2.drawMarker(ov, tuple(int(v) for v in res.disc_center), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    zoom = ov[y0:y1, x0:x1]
    rec.stage("glaucoma", "disc_cup_overlay", "Final Disc / Cup Boundary Overlay",
              cv2.resize(zoom, (zoom.shape[1] * 2, zoom.shape[0] * 2), interpolation=cv2.INTER_CUBIC),
              "Green: optic-disc boundary. Yellow: optic-cup boundary (enlarged 2x).",
              {k: res.ratios.get(k) for k in ("vcdr", "hcdr", "acdr")}, kind="overlay", group="cup")
