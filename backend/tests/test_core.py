"""Unit tests for core image-processing and feature functions (synthetic images,
so they run without any dataset)."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from fundus import morphology as M
from fundus.edges import DETECTORS
from fundus.enhancement import clahe_gray, he_gray
from fundus.evaluation import binary_classification, component_metrics, localization_error, pixel_metrics
from fundus.features import CORE_FEATURES, build_feature_table
from fundus.imageio import ImageError, to_rgb8
from fundus.optic_disc import cdr_features
from fundus.quality import entropy, quality_metrics
from fundus.roi import extract_roi
from fundus.screening import evaluate_rules
from fundus.texture import glcm_features
from fundus.vessels import segment_vessels, vessel_metrics


def synthetic_fundus(h=605, w=700, seed=0) -> np.ndarray:
    """Orange disc-shaped retina on black, a bright optic disc and dark 'vessels'."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), 280, (200, 110, 60), -1)
    cv2.circle(img, (w // 2 - 170, h // 2), 40, (250, 220, 170), -1)          # optic disc
    for k in range(6):
        ang = k * math.pi / 3
        p2 = (int(w // 2 - 170 + 260 * math.cos(ang)), int(h // 2 + 260 * math.sin(ang)))
        cv2.line(img, (w // 2 - 170, h // 2), p2, (120, 40, 20), 5)            # vessels
    field = np.zeros((h, w), np.uint8)
    cv2.circle(field, (w // 2, h // 2), 280, 1, -1)
    img[field == 0] = 0                                                         # keep vessels inside the retina
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img


# ------------------------------------------------------------------ image io
def test_to_rgb8_handles_gray_rgba_16bit():
    g = np.full((200, 200), 100, np.uint8)
    rgb, notes = to_rgb8(g)
    assert rgb.shape == (200, 200, 3) and notes
    rgba = np.zeros((200, 200, 4), np.uint8); rgba[..., 2] = 200; rgba[..., 3] = 255
    rgb, _ = to_rgb8(rgba)
    assert rgb[0, 0, 0] == 200                                   # BGR(A) -> RGB
    rgb, notes = to_rgb8(np.full((200, 200, 3), 65535, np.uint16))
    assert rgb.dtype == np.uint8 and rgb.max() == 255 and "16-bit" in notes[0]
    with pytest.raises(ImageError):
        to_rgb8(np.zeros((10, 10, 2), np.uint8))


# ------------------------------------------------------------------ morphology
def test_black_hat_highlights_dark_line():
    img = np.full((60, 60), 200, np.uint8)
    img[30, 10:50] = 50
    bh = M.black_hat(img, 7)
    assert bh[30, 30] > 100 and bh[5, 5] == 0


def test_filter_by_area_and_largest_component():
    m = np.zeros((50, 50), bool)
    m[2:4, 2:4] = True             # 4 px
    m[10:30, 10:30] = True         # 400 px
    assert M.filter_by_area(m, 10).sum() == 400
    assert M.largest_component(m).sum() == 400


def test_shape_descriptors_circle():
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 50, 1, -1)
    d = M.shape_descriptors(m.astype(bool))
    assert abs(d["equivalent_diameter"] - 100) < 2
    assert d["circularity"] > 0.85 and d["eccentricity"] < 0.2
    assert d["centroid"] == pytest.approx([100, 100], abs=0.5)


# ------------------------------------------------------------------ enhancement / quality
def test_he_respects_mask_and_clahe_shape():
    g = np.tile(np.arange(50, 150, dtype=np.uint8), (100, 1))
    mask = np.zeros(g.shape, bool); mask[:, :] = True; mask[:5] = False
    out = he_gray(g, mask)
    assert out[mask].max() == 255 and out[~mask].max() == 0
    assert clahe_gray(g).shape == g.shape


def test_entropy_and_quality_metrics():
    assert entropy(np.zeros(100, np.uint8)) == 0
    assert entropy(np.arange(256, dtype=np.uint8)) == pytest.approx(8.0)
    gray = cv2.cvtColor(synthetic_fundus(), cv2.COLOR_RGB2GRAY)
    q = quality_metrics(gray, gray > 20)
    assert 0 < q["mean_intensity"] < 255 and q["min_intensity"] >= 20
    assert q["rms_contrast"] == pytest.approx(q["std_intensity"] / 255, rel=1e-3)
    assert quality_metrics(gray, np.zeros(gray.shape, bool))["mean_intensity"] is None


def test_edge_detectors_return_uint8_maps():
    gray = cv2.cvtColor(synthetic_fundus(), cv2.COLOR_RGB2GRAY)
    for name, fn in DETECTORS.items():
        out, params = fn(gray)
        assert out.dtype == np.uint8 and out.shape == gray.shape and out.max() > 0, name
        assert isinstance(params, dict)


# ------------------------------------------------------------------ ROI / vessels
def test_roi_extracts_retinal_disc():
    rgb = synthetic_fundus()
    roi = extract_roi(rgb)
    expected = math.pi * 280 ** 2
    assert abs(roi.metrics["area_px"] - expected) / expected < 0.03
    assert roi.metrics["circularity"] > 0.85
    assert roi.valid.sum() < roi.mask.sum()
    assert not roi.mask[0, 0]                                    # black surround excluded


def test_vessels_found_and_metrics_consistent():
    rgb = synthetic_fundus()
    roi = extract_roi(rgb)
    v = segment_vessels(rgb, roi.valid, int(roi.mask.sum()))
    truth = np.zeros(rgb.shape[:2], np.uint8)
    for k in range(6):
        ang = k * math.pi / 3
        cv2.line(truth, (180, 302), (int(180 + 260 * math.cos(ang)), int(302 + 260 * math.sin(ang))), 1, 5)
    m = pixel_metrics(v.mask, truth > 0, roi.valid, v.response)
    assert m["sensitivity"] > 0.5 and m["roc_auc"] > 0.9
    assert v.metrics["vessel_area_px"] == int(v.mask.sum())
    assert v.metrics["vessel_density"] == pytest.approx(v.mask.sum() / roi.mask.sum(), rel=1e-3)


def test_vessel_metrics_branch_and_end_points():
    from skimage.morphology import skeletonize
    m = np.zeros((101, 101), bool)
    m[50, 10:91] = True
    m[10:91, 50] = True                                          # a plus sign: 1 junction, 4 ends
    mask = cv2.dilate(m.astype(np.uint8), np.ones((3, 3))) > 0
    vm = vessel_metrics(mask, skeletonize(mask), roi_area=101 * 101)
    assert vm["branch_points"] == 1 and vm["end_points"] == 4


# ------------------------------------------------------------------ glaucoma geometry
def test_cdr_formulas_and_missing_data():
    disc = {"area": 1000, "width": 40, "height": 50}
    cup = {"area": 300, "width": 20, "height": 30}
    r = cdr_features(disc, cup)
    assert r["vcdr"] == 0.6 and r["hcdr"] == 0.5 and r["acdr"] == 0.3
    assert r["rim_area"] == 700 and r["rim_to_disc_area_ratio"] == 0.7
    assert cdr_features(disc, None)["vcdr"] is None               # never fabricate
    assert cdr_features({"area": 0, "width": 0, "height": 0}, cup)["vcdr"] is None


# ------------------------------------------------------------------ evaluation
def test_pixel_metrics_restricted_to_valid_field():
    pred = np.array([[1, 1, 0, 0]], bool); ref = np.array([[1, 0, 1, 0]], bool)
    m = pixel_metrics(pred, ref)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 1)
    assert m["dice"] == 0.5 and m["iou"] == pytest.approx(1 / 3, abs=1e-3)
    valid = np.array([[1, 1, 1, 0]], bool)
    assert pixel_metrics(pred, ref, valid)["TN"] == 0           # outside FOV is not a TN


def test_component_and_localization_metrics():
    pred = np.zeros((40, 40), bool); ref = np.zeros((40, 40), bool)
    pred[5:8, 5:8] = True; pred[30:32, 30:32] = True              # 1 hit, 1 false positive
    ref[6:9, 6:9] = True; ref[20:22, 5:7] = True                  # 1 detected, 1 missed
    c = component_metrics(pred, ref, tolerance_px=1)
    assert c["matched_predictions"] == 1 and c["false_positive_components"] == 1
    assert c["detected_references"] == 1 and c["missed_references"] == 1
    loc = localization_error((0, 0), (30, 40), 60)
    assert loc["distance_px"] == 50 and loc["success"]
    assert localization_error(None, (1, 1), 60)["success"] is None
    b = binary_classification([True, True, False, False], [True, False, True, False])
    assert b["sensitivity"] == 0.5 and b["specificity"] == 0.5


# ------------------------------------------------------------------ texture / features / screening
def test_glcm_features_ranges_and_background_excluded():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 255, (120, 120)).astype(np.uint8)
    mask = np.zeros(gray.shape, bool); mask[20:100, 20:100] = True
    f = glcm_features(gray, mask)["features"]
    assert 0 < f["energy"] <= 1 and 0 < f["homogeneity"] <= 1 and -1 <= f["correlation"] <= 1
    smooth = glcm_features(np.full((120, 120), 128, np.uint8), mask)["features"]
    assert smooth["contrast"] == 0


def test_feature_table_has_32_features_and_nulls_not_zeros():
    assert len(CORE_FEATURES) == 32
    rows = build_feature_table({"quality": {"roi": {"mean_intensity": 100.0}}, "glaucoma": {"status": {"disc": "failed: x"}}})
    assert len(rows) == 32
    by = {r["key"]: r for r in rows}
    assert by["mean_intensity"]["status"] == "available"
    assert by["vcdr"]["value"] is None and by["vcdr"]["status"] == "unavailable"
    assert by["disc_area"]["note"] == "failed: x"


def test_screening_rules_handle_missing_values():
    out = evaluate_rules({"vcdr": None, "rim_to_disc_area_ratio": None, "ma_count": 30, "he_count": 9, "ex_count": 0})
    assert out["flags"]["glaucoma_suspect"]["flag"] is None
    assert out["flags"]["dr_lesions"]["flag"] is True               # 2 of 3 conditions met
    out = evaluate_rules({"vcdr": 0.3, "rim_to_disc_area_ratio": 0.8, "ma_count": 1, "he_count": 0, "ex_count": 0})
    assert out["flags"]["glaucoma_suspect"]["flag"] is False and out["flags"]["dr_lesions"]["flag"] is False
