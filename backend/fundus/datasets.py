"""Dataset registry: STARE (prepared locally), REFUGE and IDRiD (discovered
if the user has placed them under ``data/``), plus user uploads.

Only annotations that physically exist are reported. Nothing is inferred
or fabricated: an image without a vessel mask simply has no ``vessel_ah``
entry and vessel evaluation is skipped for it.
"""
from __future__ import annotations

import csv
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DATA_DIR, UPLOAD_DIR

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm"}

# STARE manifestation numbers used for image-level lesion presence labels.
# Character n of an ``imXXXX.fea.mg.txt`` code is manifestation n
# (verified against the counts on the STARE manifestations page).
STARE_MANIFESTATIONS = {
    9: ("Microaneurysm or dot hemorrhage", {0: "unknown", 1: "absent", 2: "few", 3: "many"}),
    11: ("Small/medium blot hemorrhage", {0: "unknown", 1: "absent", 2: "low density", 3: "high density"}),
    12: ("Retinal hemorrhage", {0: "unknown", 1: "absent", 2: "low density", 3: "high density", 4: "low density, regional"}),
    13: ("Retinal or subretinal exudate", {0: "unknown", 1: "absent", 2: "low severity", 3: "high severity"}),
    14: ("Circinate exudate pattern", {0: "unknown", 1: "absent", 2: "low severity", 3: "high severity"}),
    33: ("Cotton-wool spot", {0: "unknown", 1: "absent", 2: "few", 3: "many"}),
}

STARE_DIAGNOSES = {
    1: "Hollenhorst Emboli", 2: "Branch Retinal Artery Occlusion", 3: "Cilio-Retinal Artery Occlusion",
    4: "Branch Retinal Vein Occlusion", 5: "Central Retinal Vein Occlusion",
    6: "Hemi-Central Retinal Vein Occlusion", 7: "Background Diabetic Retinopathy",
    8: "Proliferative Diabetic Retinopathy", 9: "Arteriosclerotic Retinopathy",
    10: "Hypertensive Retinopathy", 11: "Coat's", 12: "Macroaneurism",
    13: "Choroidal Neovascularization", 0: "Normal", 14: "Other / see text",
}


@dataclass
class ImageRecord:
    dataset: str
    image_id: str
    path: Path
    annotations: dict[str, Any] = field(default_factory=dict)   # kind -> Path | value
    labels: dict[str, Any] = field(default_factory=dict)        # image-level reference labels
    warnings: list[str] = field(default_factory=list)

    def annotation_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.annotations.items():
            out[k] = v.name if isinstance(v, Path) else v
        return out


class Dataset:
    id: str = ""
    name: str = ""
    description: str = ""
    source_url: str = ""
    access: str = ""

    def available(self) -> bool:
        return bool(self.records())

    def records(self) -> dict[str, ImageRecord]:  # pragma: no cover - abstract
        raise NotImplementedError

    def status_message(self) -> str:
        return "ready" if self.available() else "not found under data/"

    def info(self) -> dict[str, Any]:
        recs = self.records()
        kinds: dict[str, int] = {}
        for r in recs.values():
            for k in r.annotations:
                kinds[k] = kinds.get(k, 0) + 1
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "source_url": self.source_url, "access": self.access,
            "available": bool(recs), "status": self.status_message(),
            "image_count": len(recs), "annotation_counts": kinds,
        }


# --------------------------------------------------------------------------- STARE
class StareDataset(Dataset):
    id = "stare"
    name = "STARE"
    description = ("STructured Analysis of the REtina (Hoover & Goldbaum). 397 images, 700x605. "
                   "Vessel masks for 20 images (two observers), optic-nerve centre coordinates, "
                   "manifestation grades and diagnosis codes.")
    source_url = "https://cecas.clemson.edu/~ahoover/stare/"
    access = "public download (prepared by scripts/prepare_stare.py)"

    def __init__(self, root: Path = DATA_DIR / "stare"):
        self.root = root

    @functools.lru_cache(maxsize=1)
    def records(self) -> dict[str, ImageRecord]:
        img_dir = self.root / "images"
        if not img_dir.exists():
            return {}
        recs = {p.stem: ImageRecord("stare", p.stem, p) for p in sorted(img_dir.glob("im*.ppm"))}

        for observer in ("ah", "vk"):
            for p in (self.root / "vessels" / observer).glob("*.ppm"):
                iid = p.name.split(".")[0]
                if iid in recs:
                    recs[iid].annotations[f"vessel_{observer}"] = p
        for p in (self.root / "vessels" / "hoover_results").glob("*.ppm"):
            iid = p.name.split("-")[0]
            if iid in recs:
                recs[iid].annotations["vessel_hoover_result"] = p

        gt = self.root / "GT_NERVES.txt"
        if gt.exists():
            for line in gt.read_text().splitlines():
                parts = line.split()
                if len(parts) != 3:
                    continue
                iid = parts[0].split(".")[0]
                if iid not in recs:
                    continue
                row, col = int(parts[1]), int(parts[2])   # file order is row, col
                if row < 0 or col < 0:
                    recs[iid].annotations["optic_nerve_center"] = "not visible"
                else:
                    recs[iid].annotations["optic_nerve_center"] = {"x": col, "y": row}
        exp = self.root / "nerve_experiment_ids.txt"
        if exp.exists():
            for iid in exp.read_text().split():
                if iid in recs:
                    recs[iid].labels["in_nerve_experiment_set"] = True

        for p in (self.root / "manifestations").glob("*.fea.mg.txt"):
            iid = p.name.split(".")[0]
            if iid not in recs:
                continue
            code = p.read_text().strip()
            recs[iid].annotations["manifestation_codes"] = code
            grades = {}
            for num, (label, states) in STARE_MANIFESTATIONS.items():
                if len(code) >= num and code[num - 1].isdigit():
                    s = int(code[num - 1])
                    grades[str(num)] = {"name": label, "state": s, "meaning": states.get(s, str(s))}
            recs[iid].labels["manifestations"] = grades

        dx = self.root / "all-mg-codes.txt"
        if dx.exists():
            for line in dx.read_text(encoding="latin1").splitlines():
                m = re.match(r"(im\d{4})\s+([\d ]+?)\s{2,}(.*)$", line.replace("\t", "  "))
                if not m or m.group(1) not in recs:
                    continue
                codes = [int(c) for c in m.group(2).split()]
                recs[m.group(1)].labels["diagnosis_codes"] = codes
                recs[m.group(1)].labels["diagnosis_text"] = m.group(3).strip().strip('"')
                recs[m.group(1)].labels["dr_reference"] = bool({7, 8} & set(codes))
        return recs


# --------------------------------------------------------------------------- REFUGE
class RefugeDataset(Dataset):
    """Discovers REFUGE images and disc/cup masks under ``data/refuge``.

    REFUGE masks encode cup = 0, disc rim = 128, background = 255. Glaucoma
    labels are taken from the folder name (``Glaucoma``/``Non-Glaucoma``) or
    the official training filename prefix (``g``/``n``) when present.
    """
    id = "refuge"
    name = "REFUGE"
    description = "Retinal Fundus Glaucoma Challenge: optic disc/cup masks and glaucoma labels."
    source_url = "https://refuge.grand-challenge.org/"
    access = "requires grand-challenge.org login + accepting the challenge terms"

    def __init__(self, root: Path = DATA_DIR / "refuge"):
        self.root = root

    def status_message(self) -> str:
        if self.available():
            return "ready"
        return "not downloaded - sign in at refuge.grand-challenge.org and extract into data/refuge/"

    @functools.lru_cache(maxsize=1)
    def records(self) -> dict[str, ImageRecord]:
        if not self.root.exists():
            return {}
        masks: dict[str, Path] = {}
        images: dict[str, Path] = {}
        for p in self.root.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
                continue
            parts = " ".join(x.lower() for x in p.parts)
            if "mask" in parts or "disc_cup" in parts or "annotation" in parts:
                masks[p.stem] = p
            else:
                images[p.stem] = p
        recs = {}
        for stem, p in sorted(images.items()):
            r = ImageRecord("refuge", stem, p)
            if stem in masks:
                r.annotations["disc_cup_mask"] = masks[stem]
            folder = " ".join(x.lower() for x in p.parts)
            if "non-glaucoma" in folder or re.match(r"^n\d+", stem):
                r.labels["glaucoma_reference"] = False
            elif "glaucoma" in folder or re.match(r"^g\d+", stem):
                r.labels["glaucoma_reference"] = True
            recs[stem] = r
        return recs


# --------------------------------------------------------------------------- IDRiD
class IdridDataset(Dataset):
    """Discovers IDRiD 'A. Segmentation' images and lesion masks under ``data/idrid``."""
    id = "idrid"
    name = "IDRiD"
    description = ("Indian Diabetic Retinopathy Image Dataset: pixel-level MA, HE, EX, SE and OD "
                   "masks (81 images) and DR grades.")
    source_url = "https://idrid.grand-challenge.org/"
    access = "requires IEEE DataPort login (open access, free account)"

    SUFFIX = {"MA": "mask_ma", "HE": "mask_he", "EX": "mask_ex", "SE": "mask_se", "OD": "mask_od"}

    def __init__(self, root: Path = DATA_DIR / "idrid"):
        self.root = root

    def status_message(self) -> str:
        if self.available():
            return "ready"
        return "not downloaded - sign in to IEEE DataPort and extract into data/idrid/"

    @functools.lru_cache(maxsize=1)
    def records(self) -> dict[str, ImageRecord]:
        if not self.root.exists():
            return {}
        recs: dict[str, ImageRecord] = {}
        masks: list[tuple[str, str, Path]] = []
        for p in self.root.rglob("IDRiD_*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
                continue
            m = re.match(r"^(IDRiD_\d+)(?:_(MA|HE|EX|SE|OD))?$", p.stem)
            if not m:
                continue
            if m.group(2):
                masks.append((m.group(1), m.group(2), p))
            elif p.suffix.lower() in {".jpg", ".jpeg"}:
                recs[m.group(1)] = ImageRecord("idrid", m.group(1), p)
        for iid, kind, p in masks:
            if iid in recs:
                recs[iid].annotations[self.SUFFIX[kind]] = p
        for csv_path in self.root.rglob("*Grading*Labels*.csv"):
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    iid = (row.get("Image name") or "").strip()
                    if iid in recs and row.get("Retinopathy grade", "").strip().isdigit():
                        g = int(row["Retinopathy grade"])
                        recs[iid].labels["dr_grade"] = g
                        recs[iid].labels["dr_reference"] = g > 0
        return dict(sorted(recs.items()))


# --------------------------------------------------------------------------- uploads
class UploadDataset(Dataset):
    id = "upload"
    name = "Uploaded images"
    description = "Images uploaded through the web interface (no reference annotations)."
    access = "local"

    def __init__(self, root: Path = UPLOAD_DIR):
        self.root = root

    def records(self) -> dict[str, ImageRecord]:
        recs = {}
        if not self.root.exists():
            return recs
        for d in sorted(self.root.iterdir()):
            meta = d / "upload.json"
            if not meta.exists():
                continue
            info = json.loads(meta.read_text())
            r = ImageRecord("upload", d.name, d / info["stored_name"])
            r.labels["original_filename"] = info.get("original_filename")
            recs[d.name] = r
        return recs

    def status_message(self) -> str:
        return "ready"


DATASETS: dict[str, Dataset] = {d.id: d for d in (StareDataset(), RefugeDataset(), IdridDataset(), UploadDataset())}


def get_record(dataset_id: str, image_id: str) -> ImageRecord:
    ds = DATASETS.get(dataset_id)
    if ds is None:
        raise KeyError(f"Unknown dataset '{dataset_id}'")
    rec = ds.records().get(image_id)
    if rec is None:
        raise KeyError(f"Image '{image_id}' not found in {dataset_id}")
    return rec


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None
