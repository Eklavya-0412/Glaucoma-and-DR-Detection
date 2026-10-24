"""End-to-end tests through the REST API. Uses a real STARE image when the
dataset has been prepared, otherwise a synthetic upload."""
from __future__ import annotations

import hashlib
import io
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from fundus.datasets import DATASETS
from tests.test_core import synthetic_fundus

client = TestClient(app)
HAS_STARE = DATASETS["stare"].available()


def wait_done(aid: str, timeout: float = 120) -> dict:
    t = time.time()
    while time.time() - t < timeout:
        s = client.get(f"/api/analysis/{aid}").json()
        if s["status"] in ("completed", "failed"):
            return s
        time.sleep(0.5)
    raise TimeoutError(aid)


def test_health_and_datasets():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    ids = {d["id"] for d in client.get("/api/datasets").json()}
    assert {"stare", "refuge", "idrid", "upload"} <= ids


def test_invalid_requests():
    assert client.post("/api/analyze", json={"dataset": "stare", "image_id": "../x"}).status_code == 400
    assert client.post("/api/analyze", json={"dataset": "nope", "image_id": "a"}).status_code == 404
    assert client.get("/api/analysis/nonsense").status_code == 404
    assert client.post("/api/upload", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/upload", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 422
    ok, small = cv2.imencode(".png", np.zeros((40, 40, 3), np.uint8))
    assert client.post("/api/upload", files={"file": ("s.png", small.tobytes(), "image/png")}).status_code == 422


def test_upload_analysis_without_annotations():
    ok, buf = cv2.imencode(".png", cv2.cvtColor(synthetic_fundus(), cv2.COLOR_RGB2BGR))
    up = client.post("/api/upload", files={"file": ("synthetic.png", buf.tobytes(), "image/png")}).json()
    aid = client.post("/api/analyze", json={"dataset": "upload", "image_id": up["image_id"]}).json()["analysis_id"]
    assert wait_done(aid)["status"] == "completed"
    ev = client.get(f"/api/analysis/{aid}/evaluation").json()["evaluation"]
    assert ev["vessels"]["status"].startswith("unavailable")            # missing GT handled
    feats = client.get(f"/api/analysis/{aid}/features").json()["features"]
    assert len(feats) == 32


@pytest.mark.skipif(not HAS_STARE, reason="STARE not prepared (run scripts.prepare_stare)")
def test_stare_end_to_end_artifacts_and_evaluation():
    aid = client.post("/api/analyze", json={"dataset": "stare", "image_id": "im0077"}).json()["analysis_id"]
    assert wait_done(aid)["status"] == "completed"
    stages = client.get(f"/api/analysis/{aid}/artifacts").json()["stages"]
    keys = {(s["module"], s["key"]) for s in stages}
    for expected in [("original", "original"), ("enhancement", "clahe"), ("roi", "morph_cleanup"),
                     ("vessels", "green_channel"), ("vessels", "black_hat"), ("vessels", "threshold"),
                     ("vessels", "connected_components"), ("vessels", "vessel_mask"), ("vessels", "ground_truth"),
                     ("vessels", "prediction_vs_gt"), ("glaucoma", "disc_cup_overlay"), ("lesions", "lesion_overlay")]:
        assert expected in keys, expected
    assert [s["step"] for s in stages] == list(range(1, len(stages) + 1))   # ordered, no gaps
    for s in stages[:3]:
        assert client.get(s["url"]).status_code == 200                       # real files are served

    # original preserved byte-for-byte
    rec = DATASETS["stare"].records()["im0077"]
    src = client.get("/files/stare/im0077/00_original/source.ppm").content
    assert hashlib.sha256(src).hexdigest() == hashlib.sha256(rec.path.read_bytes()).hexdigest()

    ev = client.get(f"/api/analysis/{aid}/evaluation").json()["evaluation"]
    assert "im0077.ah.ppm" in ev["vessels"]["reference"]                       # matching annotation
    assert ev["vessels"]["metrics"]["f1"] > 0.5
    assert ev["glaucoma"]["localization"]["distance_px"] is not None

    # feature table == calculated measurements
    res = client.get(f"/api/analysis/{aid}/results").json()
    feats = {f["key"]: f["value"] for f in res["features"]}
    assert feats["vessel_area"] == res["vessels"]["metrics"]["vessel_area_px"]
    assert feats["vcdr"] == res["glaucoma"]["ratios"]["vcdr"]
    assert feats["mean_intensity"] == res["quality"]["roi"]["mean_intensity"]

    csv = client.get(f"/api/analysis/{aid}/features.csv")
    assert csv.status_code == 200 and "vessel_density" in csv.text
    pdf = client.get(f"/api/analysis/{aid}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
