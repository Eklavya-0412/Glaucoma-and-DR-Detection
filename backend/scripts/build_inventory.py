"""Write ``data/inventory.csv`` and ``data/inventory.json``: one row per image
with dimensions, format, available annotations and matching ground-truth
filenames, read-check status and warnings.

Usage (from backend/):  python -m scripts.build_inventory
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fundus.config import DATA_DIR
from fundus.datasets import DATASETS, image_size

ANNOTATION_TYPES = {
    "vessel_ah": "pixel mask - vessels (observer AH)",
    "vessel_vk": "pixel mask - vessels (observer VK)",
    "vessel_hoover_result": "published algorithm output (Hoover 2000) - baseline, not ground truth",
    "optic_nerve_center": "point - optic nerve centre",
    "manifestation_codes": "image-level manifestation grades (44-digit code)",
    "disc_cup_mask": "pixel mask - optic disc + cup (REFUGE)",
    "mask_ma": "pixel mask - microaneurysms", "mask_he": "pixel mask - haemorrhages",
    "mask_ex": "pixel mask - hard exudates", "mask_se": "pixel mask - soft exudates",
    "mask_od": "pixel mask - optic disc",
}


def main() -> None:
    rows, status = [], {}
    for ds in DATASETS.values():
        if ds.id == "upload":
            continue
        recs = ds.records()
        status[ds.id] = {"name": ds.name, "source": ds.source_url, "access": ds.access,
                         "status": ds.status_message(), "images": len(recs)}
        for r in recs.values():
            size = image_size(r.path)
            warnings = list(r.warnings)
            if size is None:
                warnings.append("image unreadable")
            ann = r.annotations
            if ds.id == "stare":
                if "vessel_ah" not in ann:
                    warnings.append("no vessel mask (only 20 STARE images have one)")
                if ann.get("optic_nerve_center") == "not visible":
                    warnings.append("optic nerve not visible per GT_NERVES")
                if "manifestation_codes" not in ann:
                    warnings.append("no manifestation grades")
            rows.append({
                "dataset": ds.name, "image_id": r.image_id, "filename": r.path.name,
                "width": size[0] if size else None, "height": size[1] if size else None,
                "format": r.path.suffix.lstrip(".").upper(),
                "annotation_available": bool(ann),
                "annotation_types": "; ".join(ANNOTATION_TYPES.get(k, k) for k in ann),
                "ground_truth_files": "; ".join(v.name for v in ann.values() if isinstance(v, Path)),
                "optic_nerve_center": json.dumps(ann["optic_nerve_center"]) if "optic_nerve_center" in ann else "",
                "diagnosis": r.labels.get("diagnosis_text", ""),
                "processing_status": "readable" if size else "unreadable",
                "warnings": "; ".join(warnings),
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "inventory.csv", index=False)
    (DATA_DIR / "inventory.json").write_text(json.dumps({"datasets": status, "images": rows}, indent=2))
    print(json.dumps(status, indent=2))
    print(f"wrote {len(df)} rows to {DATA_DIR / 'inventory.csv'}")


if __name__ == "__main__":
    main()
