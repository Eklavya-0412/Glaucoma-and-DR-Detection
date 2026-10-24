"""Module M - quantitative evaluation against reference annotations.

Protocols
---------
* Pixel-level segmentation: confusion matrix computed **only inside the
  valid retinal field** (pixels outside the retina are not true negatives).
* ROC-AUC: computed from the continuous enhancement response (before
  thresholding) inside the same field, when such a response exists.
* Localisation: Euclidean distance between predicted and reference centres;
  success if the distance is below a stated tolerance.
* Component-level lesion matching: a predicted component is a true positive
  if it overlaps any reference component (dilated by ``tolerance_px``);
  a reference component is detected if any prediction overlaps it.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _safe(num: float, den: float) -> float | None:
    return None if den == 0 else round(float(num) / float(den), 4)


def pixel_metrics(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None,
                  score: np.ndarray | None = None) -> dict[str, Any]:
    pred, ref = pred.astype(bool), ref.astype(bool)
    valid = np.ones(pred.shape, bool) if valid is None else valid.astype(bool)
    p, r = pred[valid], ref[valid]
    tp = int((p & r).sum()); fp = int((p & ~r).sum())
    fn = int((~p & r).sum()); tn = int((~p & ~r).sum())
    out: dict[str, Any] = {
        "evaluated_pixels": int(valid.sum()), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": _safe(tp, tp + fn), "specificity": _safe(tn, tn + fp),
        "precision": _safe(tp, tp + fp), "recall": _safe(tp, tp + fn),
        "accuracy": _safe(tp + tn, tp + tn + fp + fn),
        "f1": _safe(2 * tp, 2 * tp + fp + fn), "dice": _safe(2 * tp, 2 * tp + fp + fn),
        "iou": _safe(tp, tp + fp + fn), "roc_auc": None,
    }
    if score is not None and 0 < r.sum() < r.size:
        from sklearn.metrics import roc_auc_score
        out["roc_auc"] = round(float(roc_auc_score(r, score[valid].astype(np.float64))), 4)
    return out


def roc_curve_points(ref: np.ndarray, score: np.ndarray, valid: np.ndarray, n: int = 60) -> dict[str, list[float]]:
    from sklearn.metrics import roc_curve
    r = ref[valid].astype(bool)
    if not (0 < r.sum() < r.size):
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(r, score[valid].astype(np.float64))
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return {"fpr": [round(float(x), 4) for x in fpr[idx]], "tpr": [round(float(x), 4) for x in tpr[idx]]}


def localization_error(pred_xy: tuple[float, float] | None, ref_xy: tuple[float, float] | None,
                       tolerance_px: float) -> dict[str, Any]:
    if pred_xy is None or ref_xy is None:
        return {"distance_px": None, "success": None, "tolerance_px": tolerance_px}
    d = math.dist(pred_xy, ref_xy)
    return {"distance_px": round(d, 2), "success": bool(d <= tolerance_px), "tolerance_px": round(tolerance_px, 1),
            "predicted": [round(pred_xy[0], 1), round(pred_xy[1], 1)], "reference": [round(ref_xy[0], 1), round(ref_xy[1], 1)]}


def component_metrics(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None,
                      tolerance_px: int = 3) -> dict[str, Any]:
    pred, ref = pred.astype(bool), ref.astype(bool)
    if valid is not None:
        pred, ref = pred & valid, ref & valid
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    ref_d = cv2.dilate(ref.astype(np.uint8), k) > 0
    pred_d = cv2.dilate(pred.astype(np.uint8), k) > 0
    n_p, lab_p = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    n_r, lab_r = cv2.connectedComponents(ref.astype(np.uint8), connectivity=8)
    hit_p = set(np.unique(lab_p[ref_d & pred])) - {0}
    hit_r = set(np.unique(lab_r[pred_d & ref])) - {0}
    tp_p, fp = len(hit_p), (n_p - 1) - len(hit_p)
    tp_r, fn = len(hit_r), (n_r - 1) - len(hit_r)
    return {
        "matching_rule": f"overlap after {tolerance_px}px dilation",
        "predicted_components": n_p - 1, "reference_components": n_r - 1,
        "matched_predictions": tp_p, "false_positive_components": fp,
        "detected_references": tp_r, "missed_references": fn,
        "component_precision": _safe(tp_p, n_p - 1), "component_sensitivity": _safe(tp_r, n_r - 1),
    }


def binary_classification(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    t = np.array(y_true, bool); p = np.array(y_pred, bool)
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum()); tn = int((~t & ~p).sum())
    return {"n": int(t.size), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": _safe(tp, tp + fn), "specificity": _safe(tn, tn + fp),
            "precision": _safe(tp, tp + fp), "accuracy": _safe(tp + tn, t.size),
            "f1": _safe(2 * tp, 2 * tp + fp + fn)}
