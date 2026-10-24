"""Dataset-level evaluation on STARE.

1. Vessel segmentation - 20 images with manual masks (observer AH as
   reference). Our method with three enhancement variants, Hoover et al.
   (2000) published result, and second observer (VK) for context.
2. Optic-disc localisation - GT_NERVES.txt; the 81-image Hoover &
   Goldbaum (2003) set and all images where the nerve is visible.
3. DR lesion candidates vs image-level manifestation grades.
4. Experimental DR flag vs diagnosis codes 7/8.

Protocol for 3-4: parameters/thresholds are chosen on the *tuning split*
(im0001-im0100) and reported separately on the *held-out split*
(im0101-im0402). Vessel parameters were chosen by comparing three methods
on all 20 vessel images (no separate held-out set exists for vessels -
documented limitation).

Usage (from backend/):  python -m scripts.evaluate_stare [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fundus.config import OUTPUT_DIR
from fundus.datasets import DATASETS
from fundus.evaluation import binary_classification, pixel_metrics
from fundus.imageio import load_mask, load_working_image
from fundus.lesions import detect_lesions
from fundus.optic_disc import segment_disc_cup
from fundus.roi import extract_roi
from fundus.screening import evaluate_rules, load_rules
from fundus.vessels import segment_vessels

OUT = OUTPUT_DIR / "_evaluation" / "stare"
TUNING_MAX_ID = 100


def split_of(image_id: str) -> str:
    return "tuning" if int(image_id[2:]) <= TUNING_MAX_ID else "held_out"


def mean_metrics(rows: list[dict], keys=("sensitivity", "specificity", "precision", "f1", "iou", "roc_auc", "accuracy")):
    df = pd.DataFrame(rows)
    return {k: (round(float(df[k].dropna().mean()), 4) if k in df and df[k].notna().any() else None) for k in keys}


def vessel_eval(records) -> dict:
    rows = []
    for r in records:
        if "vessel_ah" not in r.annotations:
            continue
        w = load_working_image(r.path)
        roi = extract_roi(w.rgb)
        ref = load_mask(r.annotations["vessel_ah"], w.rgb.shape[:2])
        for method in ("none", "he", "clahe"):
            v = segment_vessels(w.rgb, roi.valid, int(roi.mask.sum()), enhancement=method)
            rows.append({"image_id": r.image_id, "method": f"ours ({method})", **pixel_metrics(v.mask, ref, roi.valid, v.response)})
        if "vessel_hoover_result" in r.annotations:
            hb = load_mask(r.annotations["vessel_hoover_result"], ref.shape)
            rows.append({"image_id": r.image_id, "method": "Hoover et al. 2000 (published result)", **pixel_metrics(hb, ref, roi.valid)})
        if "vessel_vk" in r.annotations:
            vk = load_mask(r.annotations["vessel_vk"], ref.shape)
            rows.append({"image_id": r.image_id, "method": "second observer (VK)", **pixel_metrics(vk, ref, roi.valid)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "vessel_per_image.csv", index=False)
    summary = {m: {"n": int((df.method == m).sum()), **mean_metrics(g.to_dict("records"))} for m, g in df.groupby("method")}
    return {"reference": "observer AH", "protocol": "pixel-level inside valid retinal field; mean over images", "methods": summary}


def per_image_light(records, need_lesions: bool):
    """ROI + vessels + disc (+ lesions) without artifacts, for batch evaluation."""
    out = []
    for i, r in enumerate(records):
        w = load_working_image(r.path)
        roi = extract_roi(w.rgb)
        D = float(roi.metrics["equivalent_diameter_px"])
        v = segment_vessels(w.rgb, roi.valid, int(roi.mask.sum()))
        dc = segment_disc_cup(w.rgb, roi.valid, D)
        item = {"record": r, "disc_center": dc.disc_center, "disc_segmented": dc.disc_mask is not None,
                "vcdr": dc.ratios.get("vcdr")}
        if need_lesions:
            L = detect_lesions(w.rgb, roi.valid, v.mask, dc.disc_mask, dc.disc_center, D)
            item["counts"] = {k: L.summary[k]["count"] for k in L.summary}
        out.append(item)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(records)}")
    return out


def disc_eval(items) -> dict:
    rows = []
    for it in items:
        r = it["record"]
        gt = r.annotations.get("optic_nerve_center")
        if not isinstance(gt, dict):
            continue
        d = math.dist(it["disc_center"], (gt["x"], gt["y"])) if it["disc_center"] else None
        rows.append({"image_id": r.image_id, "distance_px": d, "success": d is not None and d <= 60,
                     "in_81_set": bool(r.labels.get("in_nerve_experiment_set")), "disc_segmented": it["disc_segmented"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "disc_localization.csv", index=False)

    def summ(sub):
        return {"n": int(len(sub)), "success_rate": round(float(sub.success.mean()), 4),
                "median_distance_px": round(float(sub.distance_px.median()), 1),
                "mean_distance_px": round(float(sub.distance_px.mean()), 1),
                "disc_segmented_rate": round(float(sub.disc_segmented.mean()), 4)}
    hoover = None
    nl = DATASETS["stare"].root / "nervelist.txt"
    if nl.exists():
        recs = DATASETS["stare"].records()
        ok = []
        for line in nl.read_text().split("\n"):
            p = line.split()
            if len(p) == 3 and p[0] in recs and recs[p[0]].labels.get("in_nerve_experiment_set"):
                gt = recs[p[0]].annotations.get("optic_nerve_center")
                if isinstance(gt, dict):
                    row, col = int(p[1]), int(p[2])
                    ok.append(row >= 0 and math.dist((col, row), (gt["x"], gt["y"])) <= 60)
        hoover = {"n": len(ok), "success_rate": round(float(np.mean(ok)), 4) if ok else None,
                  "source": "nervelist.txt (Hoover & Goldbaum 2003 fuzzy-convergence result)"}
    return {"criterion": "success if centre within 60 px of GT_NERVES (Hoover & Goldbaum 2003)",
            "hoover_81_image_set": summ(df[df.in_81_set]), "all_visible_nerve_images": summ(df),
            "published_baseline_81_set": hoover}


def _ref_present(r, nums):
    man = r.labels.get("manifestations", {})
    st = [man[str(n)]["state"] for n in nums if str(n) in man and man[str(n)]["state"] != 0]
    return None if not st else any(s >= 2 for s in st)


def youden_threshold(y, scores):
    best, thr = -1, None
    for t in sorted(set(scores)):
        pred = np.array(scores) >= t
        y_ = np.array(y)
        sens = (pred & y_).sum() / max(y_.sum(), 1)
        spec = (~pred & ~y_).sum() / max((~y_).sum(), 1)
        if sens + spec - 1 > best:
            best, thr = sens + spec - 1, t
    return thr


def lesion_eval(items) -> dict:
    cats = {"microaneurysm": [9], "hemorrhage": [11, 12], "exudate": [13, 14]}
    out = {}
    rows = []
    for it in items:
        r = it["record"]
        row = {"image_id": r.image_id, "split": split_of(r.image_id), **{f"{k}_count": it["counts"][k] for k in cats},
               "dr_reference": r.labels.get("dr_reference")}
        for k, nums in cats.items():
            row[f"{k}_ref"] = _ref_present(r, nums)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lesion_image_level.csv", index=False)
    for k in cats:
        sub = df[df[f"{k}_ref"].notna()]
        tune, held = sub[sub.split == "tuning"], sub[sub.split == "held_out"]
        thr = youden_threshold(tune[f"{k}_ref"].astype(bool).tolist(), tune[f"{k}_count"].tolist()) if len(tune) else None
        res = {"reference": f"STARE manifestation grades {cats[k]} (present = grade >= 2, unknown excluded)",
               "count_threshold_from_tuning": thr}
        for name, s in (("tuning", tune), ("held_out", held)):
            if len(s) and s[f"{k}_ref"].nunique() == 2:
                y = s[f"{k}_ref"].astype(bool)
                res[name] = {"n": int(len(s)), "positives": int(y.sum()),
                             "auc_of_candidate_count": round(float(roc_auc_score(y, s[f"{k}_count"])), 4),
                             "at_threshold": binary_classification(y.tolist(), (s[f"{k}_count"] >= thr).tolist())}
        out[k] = res
    return out, df


def dr_flag_eval(df: pd.DataFrame) -> dict:
    rules = load_rules()
    res = {"reference": "STARE diagnosis codes: DR = code 7 (background) or 8 (proliferative)",
           "rule": rules["flags"]["dr_lesions"]["conditions"]}
    sub = df[df.dr_reference.notna()].copy()
    flags = []
    for _, row in sub.iterrows():
        f = evaluate_rules({"ma_count": row.microaneurysm_count, "he_count": row.hemorrhage_count,
                            "ex_count": row.exudate_count}, rules)["flags"]["dr_lesions"]["flag"]
        flags.append(bool(f))
    sub["flag"] = flags
    for name in ("tuning", "held_out"):
        s = sub[sub.split == name]
        if len(s):
            res[name] = binary_classification(s.dr_reference.astype(bool).tolist(), s.flag.tolist())
    return res


def charts(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = summary["vessels"]["methods"]
    names = list(ms)
    metrics = ["sensitivity", "specificity", "precision", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=120)
    x = np.arange(len(metrics))
    for i, n in enumerate(names):
        ax.bar(x + (i - len(names) / 2) * 0.15 + 0.075, [ms[n][m] or 0 for m in metrics], 0.15, label=n)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1)
    ax.set_title("STARE vessel segmentation vs observer AH (mean of 20 images)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "vessel_methods.png"); plt.close(fig)


def redraw_charts() -> None:
    """Regenerate charts from an existing summary.json (no re-processing)."""
    charts(json.loads((OUT / "summary.json").read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images (lesions/disc)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recs = list(DATASETS["stare"].records().values())
    if args.limit:
        recs = recs[: args.limit]
    print("[1/4] vessels")
    summary = {"dataset": "STARE", "split": {"tuning": f"im0001-im{TUNING_MAX_ID:04d}", "held_out": f"im{TUNING_MAX_ID + 1:04d}+"}}
    summary["vessels"] = vessel_eval(recs)
    print("[2/4] per-image disc + lesions")
    items = per_image_light(recs, need_lesions=True)
    summary["disc_localization"] = disc_eval(items)
    print("[3/4] lesion image-level")
    summary["lesions_image_level"], df = lesion_eval(items)
    print("[4/4] DR flag")
    summary["dr_flag"] = dr_flag_eval(df)
    summary["vcdr_distribution"] = {"n_with_vcdr": int(sum(1 for i in items if i["vcdr"] is not None)),
                                    "median": float(np.median([i["vcdr"] for i in items if i["vcdr"] is not None]))
                                    if any(i["vcdr"] is not None for i in items) else None,
                                    "note": "STARE has no glaucoma labels or cup annotations; VCDR cannot be evaluated on STARE."}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    charts(summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
