"""Module I - lesion density and spatial distribution.

D_all = (A_MA + A_HE + A_EX) / A_retina, and per-category densities, where
A_retina is the valid retinal-field area. The retina is split into four
image quadrants about the retinal-field centroid; counts, areas and
densities are reported per quadrant. Euclidean distances from each lesion
centroid to the optic-disc centroid are summarised when the disc was found.
"""
from __future__ import annotations

import io
import math
from typing import Any

import cv2
import numpy as np

from .artifacts import ArtifactRecorder
from .lesions import COLORS
from .roi import quadrant_masks


def spatial_analysis(rgb: np.ndarray, valid: np.ndarray, roi_centroid, masks: dict[str, np.ndarray],
                     comps: dict[str, list[dict[str, Any]]], disc_center, rec: ArtifactRecorder | None = None) -> dict[str, Any]:
    a_ret = int(valid.sum())
    areas = {k: int((m & valid).sum()) for k, m in masks.items()}
    dens = {k: (round(v / a_ret, 6) if a_ret else None) for k, v in areas.items()}
    total = sum(areas.values())
    quads = quadrant_masks(valid, tuple(roi_centroid))
    per_q: dict[str, dict[str, Any]] = {}
    for qname, qm in quads.items():
        q_area = int(qm.sum())
        row: dict[str, Any] = {"retina_area_px": q_area}
        for k, m in masks.items():
            cnt = sum(1 for c in comps[k] if qm[int(round(c["centroid"][1])), int(round(c["centroid"][0]))])
            a = int((m & qm).sum())
            row[k] = {"count": cnt, "area_px": a, "density": round(a / q_area, 6) if q_area else None}
        row["total_count"] = sum(row[k]["count"] for k in masks)
        row["total_density"] = round(sum(row[k]["area_px"] for k in masks) / q_area, 6) if q_area else None
        per_q[qname] = row
    dist: dict[str, Any] = {"available": disc_center is not None}
    if disc_center is not None:
        all_d = [math.dist(c["centroid"], disc_center) for k in comps for c in comps[k]]
        if all_d:
            dist.update({"n": len(all_d), "mean_px": round(float(np.mean(all_d)), 1),
                         "median_px": round(float(np.median(all_d)), 1), "min_px": round(float(np.min(all_d)), 1),
                         "max_px": round(float(np.max(all_d)), 1)})
    else:
        dist["note"] = "optic-disc location unavailable - distances not computed"
    quadrants_with = sum(1 for q in per_q.values() if q["total_count"] > 0)
    out = {
        "retina_area_px": a_ret,
        "lesion_area_px": {**areas, "total": total},
        "density": {**dens, "total": round(total / a_ret, 6) if a_ret else None},
        "formula": "D = lesion area / valid retinal-field area",
        "quadrant_center": [round(float(v), 1) for v in roi_centroid],
        "quadrants": per_q,
        "quadrants_with_lesions": quadrants_with,
        "distance_to_disc": dist,
    }
    if rec is not None and rec.enabled:
        rec.stage("lesions", "spatial_map", "Spatial Distribution Map", _spatial_vis(rgb, valid, roi_centroid, comps, disc_center),
                  "Lesion candidate centroids by category, quadrant boundaries through the retinal-field centroid, "
                  "and the optic-disc centre (white). Rings mark 1x, 2x, 3x disc-to-field-centre distance.",
                  {"quadrant_center": out["quadrant_center"]}, kind="overlay", group="summary")
        rec.stage("lesions", "quadrant_chart", "Lesion Counts per Quadrant", _quadrant_chart(per_q, list(masks)),
                  "Number of candidates of each category in each image quadrant.", kind="plot", group="summary")
    return out


def _spatial_vis(rgb, valid, center, comps, disc_center):
    vis = (rgb * 0.45).astype(np.uint8)
    h, w = valid.shape
    cx, cy = int(center[0]), int(center[1])
    cv2.line(vis, (cx, 0), (cx, h), (200, 200, 200), 1)
    cv2.line(vis, (0, cy), (w, cy), (200, 200, 200), 1)
    for k, cl in comps.items():
        for c in cl:
            cv2.circle(vis, (int(c["centroid"][0]), int(c["centroid"][1])), 4, COLORS[k], -1)
    if disc_center is not None:
        dc = (int(disc_center[0]), int(disc_center[1]))
        cv2.drawMarker(vis, dc, (255, 255, 255), cv2.MARKER_STAR, 18, 2)
        base = math.dist(disc_center, center)
        for m in (1, 2, 3):
            if base > 5:
                cv2.circle(vis, dc, int(m * base), (120, 120, 120), 1)
    for i, (k, col) in enumerate(COLORS.items()):
        cv2.circle(vis, (14, 18 + 20 * i), 5, col, -1)
        cv2.putText(vis, k, (26, 23 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def _quadrant_chart(per_q, kinds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(per_q)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    for i, k in enumerate(kinds):
        ax.bar(x + (i - 1) * 0.26, [per_q[q][k]["count"] for q in names], 0.26, label=k,
               color=np.array(COLORS[k]) / 255.0)
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylabel("candidates")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
