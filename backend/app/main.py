"""FastAPI REST API for the fundus analysis system.

Image processing runs in a background thread pool; clients poll
``GET /api/analysis/{id}`` for status. Generated files are served from the
outputs directory through ``/files`` (StaticFiles prevents path traversal);
no other filesystem access is exposed.
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fundus.config import ALLOWED_UPLOAD_EXT, MAX_UPLOAD_BYTES, OUTPUT_DIR, UPLOAD_DIR
from fundus.datasets import DATASETS, get_record, image_size
from fundus.imageio import ImageError, load_working_image, read_any, to_rgb8
from fundus.pipeline import analysis_dir, run_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fundus.api")

app = FastAPI(title="Fundus Image Analysis API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _check_id(value: str, what: str) -> str:
    if not ID_RE.match(value):
        raise HTTPException(400, f"Invalid {what}")
    return value


def _analysis_id(dataset: str, image_id: str) -> str:
    return f"{dataset}--{image_id}"


def _split_analysis_id(analysis_id: str) -> tuple[str, str]:
    if "--" not in analysis_id:
        raise HTTPException(404, "Unknown analysis id")
    ds, iid = analysis_id.split("--", 1)
    return _check_id(ds, "dataset"), _check_id(iid, "image id")


def _manifest(analysis_id: str) -> dict[str, Any]:
    ds, iid = _split_analysis_id(analysis_id)
    p = analysis_dir(ds, iid) / "processing_manifest.json"
    if not p.exists():
        with _lock:
            job = _jobs.get(analysis_id)
        if job and job["status"] in ("queued", "running"):
            raise HTTPException(409, f"Analysis is {job['status']}")
        raise HTTPException(404, "Analysis not found - run POST /api/analyze first")
    return json.loads(p.read_text())


def _file_url(ds: str, iid: str, rel: str) -> str:
    return f"/files/{ds}/{iid}/{rel}"


# ------------------------------------------------------------------ health / datasets
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "opencv": cv2.__version__, "output_dir_writable": OUTPUT_DIR.exists(),
            "datasets": {k: d.available() for k, d in DATASETS.items()}}


@app.get("/api/datasets")
def datasets() -> list[dict[str, Any]]:
    return [d.info() for d in DATASETS.values()]


@app.get("/api/datasets/{dataset_id}/images")
def dataset_images(dataset_id: str) -> list[dict[str, Any]]:
    ds = DATASETS.get(_check_id(dataset_id, "dataset"))
    if ds is None:
        raise HTTPException(404, "Unknown dataset")
    out = []
    for r in ds.records().values():
        out.append({"image_id": r.image_id, "file": r.path.name, "annotations": sorted(r.annotations),
                    "labels": {k: v for k, v in r.labels.items() if k in ("diagnosis_text", "dr_reference",
                                                                         "glaucoma_reference", "dr_grade",
                                                                         "original_filename", "in_nerve_experiment_set")},
                    "analyzed": (analysis_dir(dataset_id, r.image_id) / "processing_manifest.json").exists()})
    return out


@app.get("/api/images/{dataset_id}/{image_id}/metadata")
def image_metadata(dataset_id: str, image_id: str) -> dict[str, Any]:
    try:
        r = get_record(_check_id(dataset_id, "dataset"), _check_id(image_id, "image id"))
    except KeyError as e:
        raise HTTPException(404, str(e))
    size = image_size(r.path)
    return {"dataset": r.dataset, "image_id": r.image_id, "file": r.path.name, "format": r.path.suffix.lstrip("."),
            "width": size[0] if size else None, "height": size[1] if size else None,
            "annotations": r.annotation_summary(), "labels": r.labels, "warnings": r.warnings}


@app.get("/api/images/{dataset_id}/{image_id}/preview")
def image_preview(dataset_id: str, image_id: str, max_side: int = 640) -> Response:
    try:
        r = get_record(_check_id(dataset_id, "dataset"), _check_id(image_id, "image id"))
    except KeyError as e:
        raise HTTPException(404, str(e))
    try:
        rgb = load_working_image(r.path, max_side=max(64, min(max_side, 1600))).rgb
    except ImageError as e:
        raise HTTPException(422, str(e))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# ------------------------------------------------------------------ upload
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_UPLOAD_EXT)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    import numpy as np
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(422, "File could not be decoded as an image")
    try:
        to_rgb8(img)
    except ImageError as e:
        raise HTTPException(422, str(e))
    if min(img.shape[:2]) < 128:
        raise HTTPException(422, f"Image too small ({img.shape[1]}x{img.shape[0]}); at least 128 px required")
    iid = "u" + uuid.uuid4().hex[:10]
    d = UPLOAD_DIR / iid
    d.mkdir(parents=True)
    stored = "original" + ext
    (d / stored).write_bytes(data)                       # stored byte-for-byte
    (d / "upload.json").write_text(json.dumps({"original_filename": name, "stored_name": stored,
                                               "uploaded_at": time.time(), "bytes": len(data)}))
    return {"dataset": "upload", "image_id": iid, "original_filename": name,
            "width": int(img.shape[1]), "height": int(img.shape[0])}


# ------------------------------------------------------------------ analysis
class AnalyzeRequest(BaseModel):
    dataset: str = Field(..., examples=["stare"])
    image_id: str = Field(..., examples=["im0077"])


def _run_job(analysis_id: str, record) -> None:
    def progress(stage: str) -> None:
        with _lock:
            _jobs[analysis_id]["stage"] = stage
    with _lock:
        _jobs[analysis_id].update(status="running", started=time.time())
    try:
        R = run_analysis(record, progress=progress)
        with _lock:
            _jobs[analysis_id].update(status="completed", finished=time.time(), elapsed_s=R["elapsed_s"],
                                      module_status=R["module_status"])
    except Exception as exc:
        log.exception("analysis %s failed", analysis_id)
        with _lock:
            _jobs[analysis_id].update(status="failed", error=str(exc), finished=time.time())


@app.post("/api/analyze", status_code=202)
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    ds, iid = _check_id(req.dataset, "dataset"), _check_id(req.image_id, "image id")
    try:
        record = get_record(ds, iid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    aid = _analysis_id(ds, iid)
    with _lock:
        if _jobs.get(aid, {}).get("status") in ("queued", "running"):
            return {"analysis_id": aid, **_jobs[aid]}
        _jobs[aid] = {"status": "queued", "stage": None, "queued_at": time.time()}
    executor.submit(_run_job, aid, record)
    return {"analysis_id": aid, "status": "queued"}


@app.get("/api/analysis/{analysis_id}")
def analysis_status(analysis_id: str) -> dict[str, Any]:
    ds, iid = _split_analysis_id(analysis_id)
    with _lock:
        job = dict(_jobs.get(analysis_id, {}))
    done = (analysis_dir(ds, iid) / "processing_manifest.json").exists()
    if not job:
        if not done:
            raise HTTPException(404, "Analysis not found")
        job = {"status": "completed"}
    out = {"analysis_id": analysis_id, "dataset": ds, "image_id": iid, **job}
    if job.get("status") == "completed" and done:
        m = _manifest(analysis_id)
        out.update(module_status=m["module_status"], elapsed_s=m.get("elapsed_s"), source=m.get("source"))
    return out


@app.get("/api/analysis/{analysis_id}/artifacts")
def analysis_artifacts(analysis_id: str) -> dict[str, Any]:
    m = _manifest(analysis_id)
    ds, iid = m["dataset"], m["image_id"]
    stages = [{**s, "url": _file_url(ds, iid, s["file"])} for s in m["stages"]]
    return {"analysis_id": analysis_id, "stages": stages,
            "files": [{"path": f, "url": _file_url(ds, iid, f)} for f in m.get("files", [])]}


@app.get("/api/analysis/{analysis_id}/results")
def analysis_results(analysis_id: str) -> dict[str, Any]:
    """Full numeric results (everything except the stage list)."""
    m = _manifest(analysis_id)
    m.pop("stages", None)
    m.pop("errors", None)
    if "lesions" in m:
        m["lesions"].pop("components", None)
    return m


@app.get("/api/analysis/{analysis_id}/features")
def analysis_features(analysis_id: str) -> dict[str, Any]:
    m = _manifest(analysis_id)
    return {"analysis_id": analysis_id, "features": m["features"],
            "csv_url": _file_url(m["dataset"], m["image_id"], "09_report/feature_table.csv")}


@app.get("/api/analysis/{analysis_id}/features.csv")
def analysis_features_csv(analysis_id: str) -> FileResponse:
    ds, iid = _split_analysis_id(analysis_id)
    p = analysis_dir(ds, iid) / "09_report" / "feature_table.csv"
    if not p.exists():
        raise HTTPException(404, "Feature table not generated")
    return FileResponse(p, media_type="text/csv", filename=f"{ds}_{iid}_features.csv")


@app.get("/api/analysis/{analysis_id}/evaluation")
def analysis_evaluation(analysis_id: str) -> dict[str, Any]:
    m = _manifest(analysis_id)
    return {"analysis_id": analysis_id, "evaluation": m["evaluation"], "annotations": m["annotations"]}


@app.get("/api/analysis/{analysis_id}/report")
def analysis_report(analysis_id: str) -> dict[str, Any]:
    m = _manifest(analysis_id)
    ds, iid = m["dataset"], m["image_id"]
    return {
        "analysis_id": analysis_id, "dataset": ds, "image_id": iid, "source": m.get("source"),
        "annotations": m.get("annotations"), "reference_labels": m.get("reference_labels"),
        "quality": (m.get("quality") or {}).get("roi"), "roi": (m.get("roi") or {}).get("metrics"),
        "glaucoma": {k: (m.get("glaucoma") or {}).get(k) for k in ("disc", "cup", "ratios", "status", "limitations")},
        "vessels": (m.get("vessels") or {}).get("metrics"),
        "lesions": {"summary": (m.get("lesions") or {}).get("summary"), "spatial": (m.get("lesions") or {}).get("spatial")},
        "texture": (m.get("texture") or {}).get("features"), "evaluation": m.get("evaluation"),
        "screening": m.get("screening"), "module_status": m.get("module_status"), "features": m.get("features"),
        "downloads": {"pdf": f"/api/analysis/{analysis_id}/report.pdf", "csv": f"/api/analysis/{analysis_id}/features.csv",
                      "manifest": _file_url(ds, iid, "processing_manifest.json")},
    }


@app.get("/api/analysis/{analysis_id}/report.pdf")
def analysis_report_pdf(analysis_id: str) -> FileResponse:
    ds, iid = _split_analysis_id(analysis_id)
    p = analysis_dir(ds, iid) / "09_report" / "screening_report.pdf"
    if not p.exists():
        raise HTTPException(404, "Report not generated")
    return FileResponse(p, media_type="application/pdf", filename=f"{ds}_{iid}_report.pdf")


# ------------------------------------------------------------------ dataset-level evaluation
@app.get("/api/evaluation/summary")
def evaluation_summary() -> dict[str, Any]:
    p = OUTPUT_DIR / "_evaluation" / "stare" / "summary.json"
    if not p.exists():
        raise HTTPException(404, "Dataset evaluation not run yet - run: python -m scripts.evaluate_stare")
    data = json.loads(p.read_text())
    data["charts"] = [f"/files/_evaluation/stare/{c.name}" for c in sorted(p.parent.glob("*.png"))]
    return data


@app.get("/api/screening/rules")
def screening_rules() -> dict[str, Any]:
    from fundus.screening import load_rules
    return load_rules()
