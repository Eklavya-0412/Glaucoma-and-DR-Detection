"""PDF screening report (ReportLab).

Sections: input & provenance, image quality, processing pipeline thumbnails,
segmentation summaries, glaucoma geometry, vessel measurements, DR lesion
candidates, texture, ground-truth evaluation, screening flags, limitations.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

LIMITATIONS = [
    "All results come from classical image-processing algorithms. They are measurements and algorithmic "
    "candidates, not clinical findings.",
    "Screening flags are experimental rules with thresholds chosen on a small tuning split; they are not "
    "validated clinical criteria and must not be used for diagnosis.",
    "Optic-cup segmentation from colour intensity is unreliable; CDR values should be read with caution.",
    "STARE has no pixel-level lesion masks and no glaucoma labels; lesion evaluation on STARE is image-level only.",
    "Pixel measurements are in pixels of the working image; no physical (mm) calibration is available.",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def build_pdf(R: dict[str, Any], out_dir: Path) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"Fundus analysis {R['dataset']}/{R['image_id']}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=6, textColor=colors.HexColor("#14213d"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                        textColor=colors.HexColor("#14213d"))
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=11)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=colors.HexColor("#555555"))
    story: list = []

    def table(rows, widths=None, header=True):
        t = Table([[Paragraph(_fmt(c), body) for c in r] for r in rows], colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9ced6")),
                 ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")))
        t.setStyle(TableStyle(style))
        return t

    def img(rel: str, width_mm: float):
        p = out_dir / rel
        if not p.exists():
            return Paragraph("(image missing)", small)
        from PIL import Image as PImage
        with PImage.open(p) as im:
            w, h = im.size
        return Image(str(p), width=width_mm * mm, height=width_mm * mm * h / w)

    story.append(Paragraph("Fundus Image Analysis — Screening Report", h1))
    story.append(Paragraph(f"<b>Image:</b> {R['dataset'].upper()} / {R['image_id']} &nbsp; "
                           f"<b>Processing time:</b> {R.get('elapsed_s')} s", body))
    src = R.get("source", {})
    story.append(Paragraph(f"<b>Source file:</b> {src.get('file')} — original shape {src.get('original_shape')}, "
                           f"working size {src.get('working_size')}. {' '.join(src.get('normalisation_notes', []))}", body))
    ann = R.get("annotations", {})
    story.append(Paragraph("<b>Available annotations:</b> " + (", ".join(f"{k}" for k in ann) or "none"), body))
    ref = R.get("screening", {}).get("reference", {})
    if ref:
        story.append(Paragraph("<b>Dataset reference labels:</b> " + "; ".join(f"{k} = {v}" for k, v in ref.items()), body))
    story.append(Paragraph(R["screening"]["disclaimer"], small))

    stages = {s["key"]: s for s in R.get("stages", [])}
    thumbs = [k for k in ("original", "retinal_mask", "clahe", "vessel_mask", "disc_cup_overlay", "lesion_overlay")
              if k in stages]
    if thumbs:
        story.append(Paragraph("Key processing outputs", h2))
        cells = [[img(stages[k]["file"], 55) for k in thumbs[i:i + 3]] for i in range(0, len(thumbs), 3)]
        caps = [[Paragraph(stages[k]["title"], small) for k in thumbs[i:i + 3]] for i in range(0, len(thumbs), 3)]
        rows = []
        for c, cap in zip(cells, caps):
            rows += [c, cap]
        t = Table(rows)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)

    story.append(Paragraph("Measured features (32-feature core set)", h2))
    rows = [["Group", "Feature", "Value", "Unit", "Status"]]
    for f in R.get("features", []):
        rows.append([f["group"], f["feature"], f["value"], f["unit"], f["status"] + (f" — {f['note']}" if f["note"] and f["status"] != "available" else "")])
    story.append(table(rows, [28 * mm, 52 * mm, 24 * mm, 30 * mm, 44 * mm]))

    story.append(PageBreak())
    story.append(Paragraph("Image quality (inside retinal field)", h2))
    q = (R.get("quality") or {}).get("roi") or {}
    story.append(table([["Metric", "Value", "Definition"]] +
                       [[k, q.get(k), R["quality"]["definitions"].get(k, "")] for k in q], [45 * mm, 22 * mm, 111 * mm]))
    enh = (R.get("enhancement") or {}).get("metrics")
    if enh:
        story.append(Paragraph("Enhancement comparison", h2))
        keys = list(enh[0].keys())
        story.append(table([keys] + [[r[k] for k in keys] for r in enh]))

    g = R.get("glaucoma") or {}
    story.append(Paragraph("Optic disc / cup", h2))
    story.append(Paragraph(f"Disc: {g.get('status', {}).get('disc', 'not run')}. Cup: {g.get('status', {}).get('cup', 'not run')}.", body))
    rat = g.get("ratios") or {}
    story.append(table([["VCDR", "HCDR", "ACDR", "Rim area", "Rim/disc"],
                        [rat.get("vcdr"), rat.get("hcdr"), rat.get("acdr"), rat.get("rim_area"),
                         rat.get("rim_to_disc_area_ratio")]]))

    vm = (R.get("vessels") or {}).get("metrics") or {}
    if vm:
        story.append(Paragraph("Vessel measurements", h2))
        story.append(table([["Measurement", "Value"]] + [[k, v] for k, v in vm.items()], [70 * mm, 40 * mm]))

    les = (R.get("lesions") or {}).get("summary") or {}
    if les:
        story.append(Paragraph("DR lesion candidates (algorithmic, unconfirmed)", h2))
        cols = ["count", "total_area_px", "mean_area_px", "max_area_px", "density", "mean_circularity",
                "mean_distance_to_disc_px"]
        story.append(table([["Category"] + cols] + [[k] + [v.get(c) for c in cols] for k, v in les.items()]))

    tx = (R.get("texture") or {}).get("features")
    if tx:
        story.append(Paragraph("GLCM texture", h2))
        story.append(table([list(tx.keys()), list(tx.values())]))

    story.append(Paragraph("Ground-truth evaluation", h2))
    ev = R.get("evaluation", {})
    any_eval = False
    v = ev.get("vessels", {})
    if "metrics" in v:
        any_eval = True
        m = v["metrics"]
        rows = [["Comparison", "Sens.", "Spec.", "Prec.", "F1/Dice", "IoU", "ROC-AUC"],
                ["Prediction vs AH", m["sensitivity"], m["specificity"], m["precision"], m["f1"], m["iou"], m["roc_auc"]]]
        for key, lab in (("second_observer_vs_prediction", "Prediction vs VK"),
                         ("published_baseline_hoover2000_vs_ah", "Hoover 2000 result vs AH"),
                         ("inter_observer_vk_vs_ah", "Observer VK vs AH")):
            if key in v:
                x = v[key]
                rows.append([lab, x["sensitivity"], x["specificity"], x["precision"], x["f1"], x["iou"], x.get("roc_auc")])
        story.append(Paragraph("Vessel segmentation (pixel level, inside retinal field):", body))
        story.append(table(rows))
    loc = (ev.get("glaucoma") or {}).get("localization")
    if loc and "distance_px" in loc:
        any_eval = True
        story.append(Paragraph(f"Optic-disc localisation: distance {loc['distance_px']} px "
                               f"({'success' if loc['success'] else 'failure'}; tolerance {loc['tolerance_px']} px).", body))
    for k in ("disc_segmentation", "cup_segmentation"):
        x = (ev.get("glaucoma") or {}).get(k)
        if x:
            any_eval = True
            story.append(Paragraph(f"{k.replace('_', ' ').title()}: Dice {x['dice']}, IoU {x['iou']}, "
                                   f"sensitivity {x['sensitivity']}, specificity {x['specificity']}.", body))
    sl = (ev.get("lesions") or {}).get("stare_image_level")
    if sl:
        any_eval = True
        rows = [["Category", "Reference grades (STARE)", "Reference present", "Predicted candidates"]]
        for kind in ("microaneurysm", "hemorrhage", "exudate"):
            x = sl[kind]
            rows.append([kind, "; ".join(f"{g['name']}: {g['meaning']}" for g in x["reference_grades"].values()),
                         x["reference_present"], x["predicted_candidate_count"]])
        story.append(Paragraph("DR lesions vs STARE image-level grades:", body))
        story.append(table(rows, [28 * mm, 85 * mm, 28 * mm, 35 * mm]))
    for kind in ("microaneurysm", "hemorrhage", "exudate"):
        x = (ev.get("lesions") or {}).get(kind)
        if x:
            any_eval = True
            p, c = x["pixel"], x["component"]
            story.append(Paragraph(f"{kind} vs {x['reference']}: pixel F1 {p['f1']}, sensitivity {p['sensitivity']}; "
                                   f"component sensitivity {c['component_sensitivity']}, precision {c['component_precision']}.", body))
    if not any_eval:
        story.append(Paragraph("No suitable ground truth is available for this image.", body))

    story.append(Paragraph("Experimental screening flags", h2))
    rows = [["Flag", "Result", "Conditions"]]
    for f in R["screening"]["flags"].values():
        conds = "; ".join(f"{c['feature']} {c['op']} {c['threshold']} (value {_fmt(c['value'])})" for c in f["conditions"])
        rows.append([f["label"], f["result"], f"{f['combine']} of: {conds}"])
    story.append(table(rows, [45 * mm, 55 * mm, 78 * mm]))

    story.append(KeepTogether([Paragraph("Limitations", h2)] + [Paragraph("• " + t, body) for t in LIMITATIONS]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Module status: " + "; ".join(f"{k}: {v}" for k, v in R.get("module_status", {}).items()), small))
    doc.build(story)
    return buf.getvalue()
