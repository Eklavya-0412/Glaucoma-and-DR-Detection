"""End-to-end analysis of one fundus image.

Execution order (also the order of stages in the processing manifest):
original -> retinal ROI -> image quality -> enhancement comparison ->
vessels -> optic disc / cup -> DR lesions + spatial analysis -> GLCM texture
-> morphology & edge-detector demonstrations -> evaluation -> features ->
screening flags -> report.

Every module is isolated: a failure is recorded in ``module_status`` and
the remaining independent modules still run. Nothing is reported as
successful unless it produced a result.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import edges as E
from . import morphology as M
from .artifacts import (ArtifactRecorder, _json_default, labelled_panel, overlay_mask,
                        prediction_vs_reference)
from .config import OUTPUT_DIR
from .datasets import ImageRecord
from .enhancement import CLAHE_CLIP, CLAHE_TILES, clahe_gray, clahe_rgb, he_rgb
from .evaluation import component_metrics, localization_error, pixel_metrics, roc_curve_points
from .features import build_feature_table, feature_csv
from .imageio import load_mask, load_working_image, read_any
from .lesions import detect_lesions
from .optic_disc import segment_disc_cup
from .quality import CORE_QUALITY, METRIC_DOCS, histogram_plot, quality_metrics, to_gray
from .roi import extract_roi
from .screening import evaluate_rules
from .spatial import spatial_analysis
from .texture import glcm_features
from .vessels import segment_vessels

log = logging.getLogger("fundus.pipeline")


def analysis_dir(dataset: str, image_id: str) -> Path:
    return OUTPUT_DIR / dataset / image_id


def run_analysis(record: ImageRecord, out_dir: Path | None = None, save_artifacts: bool = True,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    t0 = time.time()
    progress = progress or (lambda s: None)
    out_dir = out_dir or analysis_dir(record.dataset, record.image_id)
    rec = ArtifactRecorder(out_dir if save_artifacts else None)
    R: dict[str, Any] = {"dataset": record.dataset, "image_id": record.image_id, "module_status": {},
                         "annotations": record.annotation_summary(), "reference_labels": record.labels,
                         "evaluation": {}}
    status = R["module_status"]

    def module(name: str, fn: Callable[[], None]) -> None:
        progress(name)
        try:
            fn()
            status.setdefault(name, "ok")
        except Exception as exc:  # keep going: other modules are independent
            log.exception("module %s failed", name)
            status[name] = f"error: {exc}"
            R.setdefault("errors", {})[name] = traceback.format_exc(limit=3)

    # ---------------------------------------------------------------- original
    progress("loading")
    wi = load_working_image(record.path)
    rgb = wi.rgb
    h, w = rgb.shape[:2]
    R["source"] = {"file": record.path.name, "original_shape": list(wi.original_shape),
                   "original_dtype": wi.original_dtype, "original_channels": wi.original_channels,
                   "working_size": [w, h], "scale": wi.scale, "normalisation_notes": wi.notes}
    if rec.enabled:
        orig_copy = rec.path("original", "source" + record.path.suffix.lower())
        shutil.copyfile(record.path, orig_copy)          # byte-identical original
        rec.files.append(rec.rel(orig_copy))
        rec.stage("original", "original", "Original Fundus Image", rgb,
                  "Unmodified input image (8-bit RGB working copy; the source file is preserved byte-for-byte "
                  f"as {orig_copy.name}).", {"source_file": record.path.name, **({"working_scale": wi.scale} if wi.scale != 1 else {})},
                  filename="original.png")
    scale = max(h, w) / 700.0

    # ---------------------------------------------------------------- ROI
    progress("roi")
    roi = extract_roi(rgb, rec)
    R["roi"] = {"metrics": roi.metrics, "params": roi.params}
    status["roi"] = "ok" if "warning" not in roi.metrics else f"warning: {roi.metrics['warning']}"
    valid = roi.valid
    roi_area = int(valid.sum())
    D = float(roi.metrics.get("equivalent_diameter_px") or min(h, w))
    gray = to_gray(rgb)

    # ---------------------------------------------------------------- quality
    def do_quality():
        R["quality"] = {"roi": quality_metrics(gray, valid), "full_image": quality_metrics(gray),
                        "definitions": METRIC_DOCS, "core": CORE_QUALITY,
                        "region": "valid retinal field (ROI); full_image computed for comparison"}
        if rec.enabled:
            rec.stage("quality", "grayscale", "Grayscale Conversion (ITU-R BT.601)", gray,
                      "Y = 0.299R + 0.587G + 0.114B. All quality metrics use these gray levels inside the ROI.",
                      {"formula": "0.299R + 0.587G + 0.114B"})
            rec.stage("quality", "intensity_histogram", "Intensity Histogram (ROI)",
                      _png_to_rgb(histogram_plot({"Gray": gray, "Red": rgb[:, :, 0], "Green": rgb[:, :, 1],
                                                  "Blue": rgb[:, :, 2]}, valid, "ROI intensity distribution")),
                      "Distribution of gray and per-channel levels inside the retinal field.", kind="plot")
            rec.write_json("quality", "quality_metrics.json", R["quality"])
    module("quality", do_quality)

    # ---------------------------------------------------------------- enhancement
    def do_enhancement():
        he = he_rgb(rgb, roi.mask)
        cl = clahe_rgb(rgb, mask=roi.mask)
        versions = {"Original": rgb, "Histogram Equalization": he, "CLAHE": cl}
        rows = []
        for name, img in versions.items():
            m = quality_metrics(to_gray(img), valid)
            rows.append({"version": name, **{k: m[k] for k in ("mean_intensity", "std_intensity", "rms_contrast",
                                                               "local_contrast", "entropy_bits", "edge_density",
                                                               "sharpness_laplacian_var", "noise_sigma")}})
        R["enhancement"] = {"metrics": rows, "params": {
            "HE": "global histogram equalisation of CIELAB L*, mapping computed from retinal-field pixels only",
            "CLAHE": f"CIELAB L*, clip limit {CLAHE_CLIP}, {CLAHE_TILES}x{CLAHE_TILES} tiles"}}
        if rec.enabled:
            rec.stage("enhancement", "histogram_equalization", "Histogram Equalization (HE)", he,
                      "Global HE spreads the cumulative histogram of luminance over 0-255. Applied to L* so colours "
                      "are preserved; the mapping uses retinal pixels only.", {"channel": "CIELAB L*", "domain": "retinal field"})
            rec.stage("enhancement", "clahe", "CLAHE — Contrast Limited Adaptive Histogram Equalization", cl,
                      "Equalises each tile separately with a clipped histogram (limits noise amplification), then "
                      "interpolates between tiles.", {"channel": "CIELAB L*", "clip_limit": CLAHE_CLIP,
                                                       "tile_grid": f"{CLAHE_TILES}x{CLAHE_TILES}"})
            rec.stage("enhancement", "original_comparison", "Original vs HE vs CLAHE",
                      labelled_panel([("Original", rgb), ("Histogram Equalization", he),
                                      (f"CLAHE (clip {CLAHE_CLIP}, {CLAHE_TILES}x{CLAHE_TILES})", cl)], 420),
                      "Side-by-side comparison of the three versions.", kind="comparison")
            g = rgb[:, :, 1]
            rec.stage("enhancement", "green_comparison", "Green Channel: Original vs HE vs CLAHE",
                      labelled_panel([("Green channel", g), ("Green + HE", _he_masked(g, roi.mask)),
                                      ("Green + CLAHE", clahe_gray(g))], 420),
                      "The same enhancements on the green channel used by the vessel and lesion pipelines.",
                      kind="comparison")
            rec.stage("enhancement", "histogram_comparison", "Luminance Histograms: Original vs HE vs CLAHE",
                      _png_to_rgb(histogram_plot({"Original": gray, "HE": to_gray(he), "CLAHE": to_gray(cl)}, valid,
                                                 "Gray-level distribution inside the ROI")),
                      "HE flattens the histogram globally; CLAHE redistributes contrast locally.", kind="plot")
            import pandas as pd
            rec.write_bytes("enhancement", "enhancement_metrics.csv", pd.DataFrame(rows).to_csv(index=False).encode())
    module("enhancement", do_enhancement)

    # ---------------------------------------------------------------- vessels
    ves_holder: dict[str, Any] = {}

    def do_vessels():
        v = segment_vessels(rgb, valid, roi_area, rec)
        ves_holder["v"] = v
        R["vessels"] = {"metrics": v.metrics, "params": v.params}
        ref_path = record.annotations.get("vessel_ah")
        if isinstance(ref_path, Path):
            ref = load_mask(ref_path, (h, w))
            ev = {"reference": f"STARE manual vessel mask, observer AH ({ref_path.name})",
                  "protocol": "pixel-level, inside the valid retinal field; ROC-AUC from the black-hat response",
                  "metrics": pixel_metrics(v.mask, ref, valid, v.response),
                  "roc": roc_curve_points(ref, v.response, valid)}
            vk = record.annotations.get("vessel_vk")
            if isinstance(vk, Path):
                ref2 = load_mask(vk, (h, w))
                ev["second_observer_vs_prediction"] = pixel_metrics(v.mask, ref2, valid)
                ev["inter_observer_vk_vs_ah"] = pixel_metrics(ref2, ref, valid)
            hb = record.annotations.get("vessel_hoover_result")
            if isinstance(hb, Path):
                ev["published_baseline_hoover2000_vs_ah"] = pixel_metrics(load_mask(hb, (h, w)), ref, valid)
            # enhancement -> downstream segmentation effect (same GT, same threshold rule)
            down = []
            for method in ("none", "he", "clahe"):
                vr = segment_vessels(rgb, valid, roi_area, None, enhancement=method)
                m = pixel_metrics(vr.mask, ref, valid, vr.response)
                down.append({"enhancement": method, **{k: m[k] for k in ("sensitivity", "specificity", "precision",
                                                                         "f1", "roc_auc")}})
            ev["enhancement_effect"] = down
            R["evaluation"]["vessels"] = ev
            if rec.enabled:
                rec.stage("vessels", "ground_truth", "Ground-Truth Annotation (STARE observer AH)", ref,
                          "Manual vessel tracing by A. Hoover. Shown for comparison only; never used by the algorithm.",
                          {"file": ref_path.name}, kind="reference", group="evaluation")
                rec.stage("vessels", "prediction_vs_gt", "Prediction vs Ground Truth",
                          prediction_vs_reference(v.mask, ref, valid),
                          "White: true positive, red: false positive, green: false negative (missed vessel), "
                          "grey: outside evaluated field.", {k: ev["metrics"][k] for k in ("f1", "sensitivity",
                                                                                            "specificity", "roc_auc")},
                          kind="comparison", group="evaluation")
                rec.stage("vessels", "side_by_side", "Predicted Mask | Ground Truth",
                          labelled_panel([("Predicted vessel mask", v.mask), ("Ground truth (AH)", ref)], 420),
                          "Side-by-side comparison of prediction and reference.", kind="comparison", group="evaluation")
        else:
            R["evaluation"]["vessels"] = {"status": "unavailable: no vessel reference mask for this image"}
    module("vessels", do_vessels)
    vmask = ves_holder["v"].mask if "v" in ves_holder else np.zeros((h, w), bool)

    # ---------------------------------------------------------------- glaucoma
    disc_holder: dict[str, Any] = {}

    def do_glaucoma():
        dc = segment_disc_cup(rgb, valid, D, rec)
        disc_holder["dc"] = dc
        R["glaucoma"] = {"disc_center": dc.disc_center, "candidates": dc.candidates, "disc": dc.disc, "cup": dc.cup,
                         "ratios": dc.ratios, "status": dc.status, "params": dc.params,
                         "limitations": "Intensity-based segmentation; the cup boundary is defined clinically by "
                                        "depth/vessel bending and is hard to see in colour. Wrong localisation on "
                                        "images with bright lesions. VCDR alone is not a glaucoma diagnosis."}
        ev: dict[str, Any] = {}
        nerve = record.annotations.get("optic_nerve_center")
        if isinstance(nerve, dict):
            ref_xy = (nerve["x"] * wi.scale, nerve["y"] * wi.scale)
            tol = 60 * scale
            ev["localization"] = {"reference": "STARE GT_NERVES.txt optic-nerve centre",
                                  "criterion": "success if distance <= 60 px at 700x605 (Hoover & Goldbaum 2003)",
                                  **localization_error(dc.disc_center, ref_xy, tol)}
        elif nerve == "not visible":
            ev["localization"] = {"status": "reference says optic nerve not visible in this image"}
        for key in ("disc_cup_mask", "mask_od"):
            mp = record.annotations.get(key)
            if isinstance(mp, Path):
                ev.update(_evaluate_disc_masks(mp, key, dc, (h, w), valid, rec, rgb))
        if not ev:
            ev["status"] = "unavailable: no optic disc/cup reference for this image"
        R["evaluation"]["glaucoma"] = ev
        if rec.enabled:
            rec.write_json("glaucoma", "glaucoma_features.json", R["glaucoma"])
        if dc.disc_mask is None:
            status["glaucoma"] = f"partial: {dc.status.get('disc')}"
        elif dc.cup_mask is None:
            status["glaucoma"] = f"partial: {dc.status.get('cup')}"
    module("glaucoma", do_glaucoma)
    dc = disc_holder.get("dc")

    # ---------------------------------------------------------------- lesions
    def do_lesions():
        L = detect_lesions(rgb, valid, vmask, dc.disc_mask if dc else None, dc.disc_center if dc else None, D, rec)
        sp = spatial_analysis(rgb, valid, roi.metrics["centroid"], L.masks, L.components,
                              dc.disc_center if dc else None, rec)
        R["lesions"] = {"summary": L.summary, "spatial": sp, "params": L.params,
                        "fovea": L.fovea, "components": {k: v[:300] for k, v in L.components.items()},
                        "note": "Algorithmic candidates from thresholded connected components - not confirmed lesions."}
        ev: dict[str, Any] = {}
        for kind, key, label in (("microaneurysm", "mask_ma", "MA"), ("hemorrhage", "mask_he", "HE"),
                                 ("exudate", "mask_ex", "EX")):
            mp = record.annotations.get(key)
            if isinstance(mp, Path):
                ref = load_mask(mp, (h, w), threshold=0)
                ev[kind] = {"reference": f"IDRiD {label} mask ({mp.name})",
                            "pixel": pixel_metrics(L.masks[kind], ref, valid, L.scores[kind]),
                            "component": component_metrics(L.masks[kind], ref, valid, tolerance_px=max(2, int(3 * scale)))}
                if rec.enabled:
                    rec.stage("lesions", f"{kind}_vs_gt", f"{kind.title()} Prediction vs Ground Truth (IDRiD)",
                              prediction_vs_reference(L.masks[kind], ref, valid),
                              "White: TP, red: FP, green: FN.", {"f1": ev[kind]["pixel"]["f1"]},
                              kind="comparison", group=kind)
        man = record.labels.get("manifestations")
        if man:
            image_level = {}
            for kind, nums in (("microaneurysm", ["9"]), ("hemorrhage", ["11", "12"]), ("exudate", ["13", "14"])):
                grades = {n: man[n] for n in nums if n in man}
                known = [g["state"] for g in grades.values() if g["state"] != 0]
                image_level[kind] = {
                    "reference_grades": grades,
                    "reference_present": (any(s >= 2 for s in known) if known else None),
                    "predicted_candidate_count": L.summary[kind]["count"],
                }
            ev["stare_image_level"] = {"note": "STARE provides image-level manifestation grades (no lesion masks); "
                                               "pixel-level lesion evaluation is not possible on STARE.",
                                       **image_level}
        if not ev:
            ev["status"] = "unavailable: no lesion reference for this image"
        R["evaluation"]["lesions"] = ev
        if rec.enabled:
            rec.write_json("lesions", "lesion_features.json", {k: R["lesions"][k] for k in ("summary", "spatial", "params", "fovea")})
    module("lesions", do_lesions)

    # ---------------------------------------------------------------- texture
    def do_texture():
        R["texture"] = glcm_features(gray, valid, rec)
        if rec.enabled:
            rec.write_json("texture", "glcm_features.json", R["texture"])
    module("texture", do_texture)

    # ---------------------------------------------------------------- morphology & edges demonstrations
    def do_morph_demo():
        base = clahe_gray(rgb[:, :, 1])
        size = max(3, int(round(7 * scale)) | 1)
        ops = [("erosion", "Erosion", M.erode(base, size), "Minimum filter: shrinks bright regions, widens dark vessels."),
               ("dilation", "Dilation", M.dilate(base, size), "Maximum filter: grows bright regions, removes thin dark vessels."),
               ("opening", "Opening (erosion → dilation)", M.opening(base, size), "Removes bright details smaller than the SE."),
               ("closing", "Closing (dilation → erosion)", M.closing(base, size), "Removes dark details smaller than the SE (e.g. vessels)."),
               ("gradient", "Morphological Gradient", M.gradient(base, size), "Dilation − erosion: highlights boundaries."),
               ("top_hat", "White Top-Hat", M.top_hat(base, 4 * size + 1), "Image − opening: isolates small bright structures (exudate-like)."),
               ("black_hat", "Black-Hat", M.black_hat(base, 2 * size + 1), "Closing − image: isolates small dark structures (vessels, MAs).")]
        R["morphology_demo"] = {"input": "CLAHE green channel", "structuring_element": "ellipse",
                                "sizes": {k: (4 * size + 1 if k == "top_hat" else 2 * size + 1 if k == "black_hat" else size) for k, *_ in ops}}
        if rec.enabled:
            for key, title, img, expl in ops:
                rec.stage("morphology", key, title, img, expl,
                          {"input": "CLAHE green", "se": f"ellipse {R['morphology_demo']['sizes'][key]}px", "iterations": 1})
        if rec.enabled:
            for name, fn in E.DETECTORS.items():
                img, p = fn(base)
                rec.stage("edges", name.lower(), f"{name} Edge Detection", img,
                          {"Sobel": "First-derivative gradient magnitude with 3x3 Sobel kernels (smoothing built in).",
                           "Prewitt": "First-derivative gradient magnitude with 3x3 Prewitt (box-averaging) kernels.",
                           "Canny": "Gaussian smoothing, gradient, non-maximum suppression and hysteresis thresholding: thin, connected edges.",
                           "Laplacian": "Second derivative (Laplacian of Gaussian magnitude): responds to both edges and fine texture."}[name],
                          {"input": "CLAHE green", **p})
    module("morphology", do_morph_demo)

    # ---------------------------------------------------------------- evaluation summary artifact
    if rec.enabled:
        rec.write_json("evaluation", "evaluation_metrics.json", R["evaluation"])
        panels = [(s.title, cv2.cvtColor(read_any(rec.root / s.file), cv2.COLOR_BGR2RGB)) for s in rec.stages
                  if s.key == "prediction_vs_gt" or s.key.endswith("_vs_gt")]
        if panels:
            rec.stage("evaluation", "evaluation_comparison", "Prediction vs Reference — All Evaluated Tasks",
                      labelled_panel(panels, 360, cols=min(3, len(panels))),
                      "All prediction-vs-ground-truth maps for this image (white TP, red FP, green FN).",
                      kind="comparison")

    # ---------------------------------------------------------------- features / screening / report
    progress("features")
    R["features"] = build_feature_table(R)
    fdict = {f["key"]: f["value"] for f in R["features"]}
    fdict["rim_to_disc_area_ratio"] = (R.get("glaucoma") or {}).get("ratios", {}).get("rim_to_disc_area_ratio")
    R["screening"] = evaluate_rules(fdict)
    R["screening"]["reference"] = {k: record.labels[k] for k in ("diagnosis_codes", "diagnosis_text", "dr_reference",
                                                                  "dr_grade", "glaucoma_reference") if k in record.labels}
    R["elapsed_s"] = round(time.time() - t0, 2)
    R["stages"] = rec.all_stages()
    if rec.enabled:
        progress("report")
        meta = {"dataset": record.dataset, "image_id": record.image_id}
        rec.write_bytes("report", "feature_table.csv", feature_csv(R["features"], meta).encode("utf-8"))
        rec.write_json("report", "analysis_metadata.json", {k: R[k] for k in ("dataset", "image_id", "source",
                                                                              "annotations", "reference_labels",
                                                                              "module_status", "elapsed_s")})
        try:
            from .report import build_pdf
            rec.write_bytes("report", "screening_report.pdf", build_pdf(R, out_dir))
        except Exception as exc:
            log.exception("PDF report failed")
            status["report"] = f"error: {exc}"
        R["files"] = rec.files
        (out_dir / "processing_manifest.json").write_text(json.dumps(R, indent=2, default=_json_default))
    return R


def _evaluate_disc_masks(mp: Path, key: str, dc, shape, valid, rec, rgb) -> dict[str, Any]:
    raw = read_any(mp)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    out: dict[str, Any] = {}
    if key == "disc_cup_mask":       # REFUGE: 0 cup, 128 rim, 255 background
        ref_disc, ref_cup = raw < 200, raw < 64
        src = "REFUGE disc/cup mask"
    else:                            # IDRiD: OD mask (non-zero = disc)
        ref_disc, ref_cup = raw > 0, None
        src = "IDRiD optic disc mask"
    pred_disc = dc.disc_mask if dc.disc_mask is not None else np.zeros(shape, bool)
    out["disc_segmentation"] = {"reference": f"{src} ({mp.name})", **pixel_metrics(pred_disc, ref_disc, valid)}
    if ref_cup is not None:
        pred_cup = dc.cup_mask if dc.cup_mask is not None else np.zeros(shape, bool)
        out["cup_segmentation"] = {"reference": f"{src} ({mp.name})", **pixel_metrics(pred_cup, ref_cup, valid)}
        rd, rc = M.shape_descriptors(ref_disc), M.shape_descriptors(ref_cup)
        if rd.get("area") and rc.get("area"):
            ref_vcdr = rc["height"] / rd["height"]
            pv = dc.ratios.get("vcdr")
            out["vcdr"] = {"reference": round(ref_vcdr, 4), "predicted": pv,
                           "absolute_error": round(abs(pv - ref_vcdr), 4) if pv is not None else None}
    if rec.enabled:
        rec.stage("glaucoma", "disc_vs_gt", "Optic Disc: Prediction vs Ground Truth", prediction_vs_reference(pred_disc, ref_disc, valid),
                  "White TP, red FP, green FN.", {"dice": out["disc_segmentation"]["dice"]}, kind="comparison", group="evaluation")
        if ref_cup is not None:
            rec.stage("glaucoma", "cup_vs_gt", "Optic Cup: Prediction vs Ground Truth", prediction_vs_reference(pred_cup, ref_cup, valid),
                      "White TP, red FP, green FN.", {"dice": out["cup_segmentation"]["dice"]}, kind="comparison", group="evaluation")
        ov = rgb.copy()
        for m, col in ((ref_disc, (0, 140, 255)), (ref_cup, (255, 0, 200))):
            if m is not None:
                ov = overlay_mask(ov, m, col, outline=True)
        rec.stage("glaucoma", "reference_disc_cup", "Reference Disc / Cup Annotation", ov,
                  "Blue: reference disc, magenta: reference cup.", kind="reference", group="evaluation")
    return out


def _he_masked(g, mask):
    from .enhancement import he_gray
    return he_gray(g, mask)


def _png_to_rgb(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
