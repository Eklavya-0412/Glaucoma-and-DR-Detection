"""Artifact recording.

Every processing step that produces an image registers a *stage*: its global
step number, module, full technique name, explanation, parameters, image file
and basic statistics. The resulting list is written to
``processing_manifest.json`` and drives the frontend pipeline view, so the
UI can only ever show steps that actually ran.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .imageio import save_png, to_u8

MODULE_DIRS = {
    "original": "00_original",
    "quality": "01_quality",
    "enhancement": "02_enhancement",
    "roi": "03_roi",
    "vessels": "04_vessels",
    "glaucoma": "05_glaucoma",
    "lesions": "06_dr_lesions",
    "texture": "07_texture",
    "evaluation": "08_evaluation",
    "report": "09_report",
    "morphology": "10_morphology",
    "edges": "11_edges",
}


@dataclass
class Stage:
    step: int
    module: str
    key: str
    title: str
    technique: str
    explanation: str
    file: str                       # path relative to the analysis directory
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    kind: str = "image"             # image | mask | overlay | comparison | plot | reference
    group: str = ""                 # sub-pipeline (e.g. "microaneurysm")


def image_stats(img: np.ndarray) -> dict[str, Any]:
    s: dict[str, Any] = {"width": int(img.shape[1]), "height": int(img.shape[0])}
    if img.dtype == bool:
        s["foreground_pixels"] = int(img.sum())
        s["foreground_fraction"] = round(float(img.mean()), 5)
    else:
        u = to_u8(img)
        s["mean"] = round(float(u.mean()), 2)
        s["std"] = round(float(u.std()), 2)
    return s


class ArtifactRecorder:
    def __init__(self, root: Path | None):
        """``root=None`` disables image writing (used for fast batch evaluation)."""
        self.root = root
        self.stages: list[Stage] = []
        self.files: list[str] = []
        if root is not None:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def path(self, module: str, filename: str) -> Path:
        assert self.root is not None
        p = self.root / MODULE_DIRS[module] / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, p: Path) -> str:
        assert self.root is not None
        return p.relative_to(self.root).as_posix()

    def stage(self, module: str, key: str, title: str, img: np.ndarray, explanation: str,
              params: dict[str, Any] | None = None, technique: str | None = None,
              kind: str = "image", group: str = "", filename: str | None = None,
              extra_stats: dict[str, Any] | None = None) -> Stage | None:
        if not self.enabled:
            return None
        p = self.path(module, filename or f"{key}.png")
        save_png(p, img)
        st = Stage(len(self.stages) + 1, module, key, title, technique or title, explanation,
                   self.rel(p), params or {}, {**image_stats(img), **(extra_stats or {})}, kind, group)
        self.stages.append(st)
        self.files.append(st.file)
        return st

    def write_json(self, module: str, filename: str, data: Any) -> str | None:
        if not self.enabled:
            return None
        p = self.path(module, filename)
        p.write_text(json.dumps(data, indent=2, default=_json_default))
        self.files.append(self.rel(p))
        return self.rel(p)

    def write_bytes(self, module: str, filename: str, data: bytes) -> str | None:
        if not self.enabled:
            return None
        p = self.path(module, filename)
        p.write_bytes(data)
        self.files.append(self.rel(p))
        return self.rel(p)

    def stages_for(self, module: str) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.stages if s.module == module]

    def all_stages(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.stages]


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ----------------------------------------------------------------- comparison panels
def _as_rgb(img: np.ndarray) -> np.ndarray:
    u = to_u8(img)
    return cv2.cvtColor(u, cv2.COLOR_GRAY2RGB) if u.ndim == 2 else u


def labelled_panel(items: list[tuple[str, np.ndarray]], tile_width: int = 420, cols: int | None = None) -> np.ndarray:
    """Compose images side by side, each with its technique name above it."""
    cols = cols or len(items)
    tiles = []
    for title, img in items:
        rgb = _as_rgb(img)
        h = int(rgb.shape[0] * tile_width / rgb.shape[1])
        rgb = cv2.resize(rgb, (tile_width, h), interpolation=cv2.INTER_AREA)
        bar = np.full((34, tile_width, 3), 255, np.uint8)
        scale = 0.55
        while cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > tile_width - 12 and scale > 0.3:
            scale -= 0.03
        cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 30, 45), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, rgb]))
    th = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((th - t.shape[0], tile_width, 3), 255, np.uint8)]) for t in tiles]
    gap = np.full((th, 6, 3), 255, np.uint8)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 255))
        parts = []
        for i, t in enumerate(row):
            parts += [t] if i == 0 else [gap, t]
        rows.append(np.hstack(parts))
    out = []
    for i, r in enumerate(rows):
        out += [r] if i == 0 else [np.full((6, r.shape[1], 3), 255, np.uint8), r]
    return np.vstack(out)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55,
                 outline: bool = False) -> np.ndarray:
    out = rgb.copy()
    if outline:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color, 2)
        return out
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def prediction_vs_reference(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Colour-coded agreement map: TP white, FP red, FN green, TN black, outside FOV grey."""
    out = np.zeros(pred.shape + (3,), np.uint8)
    out[pred & ref] = (255, 255, 255)
    out[pred & ~ref] = (230, 60, 60)
    out[~pred & ref] = (60, 200, 90)
    if valid is not None:
        out[~valid] = (60, 60, 60)
    return out
