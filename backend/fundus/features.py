"""Module L - consistent feature table (32-feature core set).

Every feature has a value, unit, group and an explicit status:
``available`` | ``unavailable`` (module could not measure it, e.g. disc not
found) | ``failed``. Missing values are ``None`` - never replaced by 0.
"""
from __future__ import annotations

import csv
import io
from typing import Any

# (key, label, group, unit)
CORE_FEATURES: list[tuple[str, str, str, str]] = [
    ("mean_intensity", "Mean intensity", "Image quality", "gray level"),
    ("std_intensity", "Intensity standard deviation", "Image quality", "gray level"),
    ("contrast", "Contrast (RMS)", "Image quality", "ratio"),
    ("entropy", "Entropy", "Image quality", "bits"),
    ("edge_density", "Edge density", "Image quality", "fraction"),
    ("disc_area", "Disc area", "Glaucoma", "px²"),
    ("cup_area", "Cup area", "Glaucoma", "px²"),
    ("disc_width", "Disc width", "Glaucoma", "px"),
    ("disc_height", "Disc height", "Glaucoma", "px"),
    ("cup_width", "Cup width", "Glaucoma", "px"),
    ("cup_height", "Cup height", "Glaucoma", "px"),
    ("vcdr", "Vertical CDR", "Glaucoma", "ratio"),
    ("hcdr", "Horizontal CDR", "Glaucoma", "ratio"),
    ("acdr", "Area CDR", "Glaucoma", "ratio"),
    ("rim_area", "Neuroretinal rim area", "Glaucoma", "px²"),
    ("vessel_area", "Vessel area", "Blood vessels", "px²"),
    ("vessel_density", "Vessel density", "Blood vessels", "fraction of retina"),
    ("mean_vessel_width", "Mean vessel width", "Blood vessels", "px"),
    ("branching_density", "Branching density", "Blood vessels", "branch points / 10⁴ px"),
    ("ma_count", "Microaneurysm candidate count", "DR lesions", "count"),
    ("ma_area", "Microaneurysm candidate area", "DR lesions", "px²"),
    ("he_count", "Hemorrhage candidate count", "DR lesions", "count"),
    ("he_area", "Hemorrhage candidate area", "DR lesions", "px²"),
    ("ex_count", "Exudate candidate count", "DR lesions", "count"),
    ("ex_area", "Exudate candidate area", "DR lesions", "px²"),
    ("total_lesion_area", "Total lesion area", "DR lesions", "px²"),
    ("total_lesion_density", "Total lesion density", "DR lesions", "fraction of retina"),
    ("lesion_spatial_distribution", "Lesion spatial distribution", "DR lesions", "quadrants with candidates (0-4)"),
    ("glcm_contrast", "GLCM contrast", "Texture", "-"),
    ("glcm_homogeneity", "GLCM homogeneity", "Texture", "-"),
    ("glcm_energy", "GLCM energy", "Texture", "-"),
    ("glcm_correlation", "GLCM correlation", "Texture", "-"),
]


def build_feature_table(r: dict[str, Any]) -> list[dict[str, Any]]:
    """``r`` is the per-image results dict assembled by the pipeline."""
    q = r.get("quality", {}).get("roi", {}) or {}
    g = r.get("glaucoma", {}) or {}
    disc, cup, ratios = g.get("disc") or {}, g.get("cup") or {}, g.get("ratios") or {}
    v = r.get("vessels", {}).get("metrics", {}) or {}
    les = r.get("lesions", {}).get("summary", {}) or {}
    sp = r.get("lesions", {}).get("spatial", {}) or {}
    tx = (r.get("texture") or {}).get("features", {}) or {}
    status = r.get("module_status", {})

    def les_get(kind, key):
        return les.get(kind, {}).get(key) if kind in les else None

    values = {
        "mean_intensity": q.get("mean_intensity"), "std_intensity": q.get("std_intensity"),
        "contrast": q.get("rms_contrast"), "entropy": q.get("entropy_bits"), "edge_density": q.get("edge_density"),
        "disc_area": disc.get("area"), "cup_area": cup.get("area"),
        "disc_width": disc.get("width"), "disc_height": disc.get("height"),
        "cup_width": cup.get("width"), "cup_height": cup.get("height"),
        "vcdr": ratios.get("vcdr"), "hcdr": ratios.get("hcdr"), "acdr": ratios.get("acdr"),
        "rim_area": ratios.get("rim_area"),
        "vessel_area": v.get("vessel_area_px"), "vessel_density": v.get("vessel_density"),
        "mean_vessel_width": v.get("mean_vessel_width_px"),
        "branching_density": v.get("branching_density_per_10k_px"),
        "ma_count": les_get("microaneurysm", "count"), "ma_area": les_get("microaneurysm", "total_area_px"),
        "he_count": les_get("hemorrhage", "count"), "he_area": les_get("hemorrhage", "total_area_px"),
        "ex_count": les_get("exudate", "count"), "ex_area": les_get("exudate", "total_area_px"),
        "total_lesion_area": (sp.get("lesion_area_px") or {}).get("total"),
        "total_lesion_density": (sp.get("density") or {}).get("total"),
        "lesion_spatial_distribution": sp.get("quadrants_with_lesions"),
        "glcm_contrast": tx.get("contrast"), "glcm_homogeneity": tx.get("homogeneity"),
        "glcm_energy": tx.get("energy"), "glcm_correlation": tx.get("correlation"),
    }
    group_module = {"Image quality": "quality", "Glaucoma": "glaucoma", "Blood vessels": "vessels",
                    "DR lesions": "lesions", "Texture": "texture"}
    rows = []
    for key, label, group, unit in CORE_FEATURES:
        val = values[key]
        mod_status = status.get(group_module[group], "ok")
        if val is not None:
            st = "available"
            note = "algorithmic candidate count/area, not confirmed lesions" if group == "DR lesions" else ""
        else:
            st = "failed" if str(mod_status).startswith("error") else "unavailable"
            note = _reason(key, g, mod_status)
        rows.append({"key": key, "feature": label, "group": group, "value": val, "unit": unit,
                     "status": st, "note": note})
    return rows


def _reason(key: str, g: dict[str, Any], mod_status: str) -> str:
    st = g.get("status", {})
    if key.startswith(("disc", "vcdr", "hcdr", "acdr", "rim")) or key.startswith("cup"):
        if key.startswith("cup") or key in ("vcdr", "hcdr", "acdr", "rim_area"):
            return st.get("cup") or st.get("disc") or "glaucoma module did not run"
        return st.get("disc") or "glaucoma module did not run"
    return str(mod_status)


def feature_csv(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["dataset", "image_id", "group", "feature_key", "feature", "value", "unit", "status", "note"])
    for r in rows:
        w.writerow([meta["dataset"], meta["image_id"], r["group"], r["key"], r["feature"],
                    "" if r["value"] is None else r["value"], r["unit"], r["status"], r["note"]])
    return buf.getvalue()
