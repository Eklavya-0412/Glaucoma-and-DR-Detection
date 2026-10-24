"""Project-wide paths and tunable defaults.

Paths are resolved relative to the repository root unless overridden by the
environment variables ``FUNDUS_DATA_DIR`` / ``FUNDUS_OUTPUT_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FUNDUS_DATA_DIR", REPO_ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.environ.get("FUNDUS_OUTPUT_DIR", REPO_ROOT / "outputs")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"

# Large images (IDRiD is 4288x2848) are processed on a down-scaled working
# copy whose longest side is at most this many pixels. The original file is
# never modified.
MAX_WORKING_SIDE = int(os.environ.get("FUNDUS_MAX_SIDE", "1000"))

MAX_UPLOAD_BYTES = int(os.environ.get("FUNDUS_MAX_UPLOAD_MB", "25")) * 1024 * 1024
ALLOWED_UPLOAD_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
