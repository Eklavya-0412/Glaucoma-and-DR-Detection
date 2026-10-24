"""Prepare the STARE dataset: locate (or download) the official archives,
extract them into ``data/stare/`` and verify every file can be read.

Official source: https://cecas.clemson.edu/~ahoover/stare/

Archives already present in ``data/`` (or ``data/raw/stare``) are reused.
Anything missing is downloaded from the official STARE web site, which is
publicly accessible without login.

Usage (from ``backend/``)::

    python -m scripts.prepare_stare
"""
from __future__ import annotations

import gzip
import io
import re
import shutil
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import cv2

from fundus.config import DATA_DIR

BASE_URL = "https://cecas.clemson.edu/~ahoover/stare"

# name -> (relative URL on the STARE site, expected member pattern)
RESOURCES: dict[str, str] = {
    "all-images.zip": "images/all-images.zip",
    "annotations.zip": "manifestations/annotations.zip",
    "all-mg-codes.txt": "diagnoses/all-mg-codes.txt",
    "labels-ah.tar": "probing/labels-ah.tar",
    "labels-vk.tar": "probing/labels-vk.tar",
    "results-4.tar": "probing/results-4.tar",
    "GT_NERVES.txt": "nerve/GT_NERVES.txt",
    "nervelist.txt": "nerve/nervelist.txt",
    "manifestations.html": "manifestations.html",
}

SAFE_NAME = re.compile(r"^im\d{4}[\w.\-]*$")

STARE_DIR = DATA_DIR / "stare"
DOWNLOAD_DIR = DATA_DIR / "raw" / "stare"


def _find_local(name: str) -> Path | None:
    for root in (DATA_DIR, DOWNLOAD_DIR, DOWNLOAD_DIR / "nerve"):
        p = root / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _download(name: str, rel_url: str) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOAD_DIR / name
    url = f"{BASE_URL}/{rel_url}"
    print(f"  downloading {url}")
    # The Clemson server's certificate chain is not always present in the
    # Windows trust store; the files are public, so fall back gracefully.
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        data = urllib.request.urlopen(url, timeout=120, context=ctx).read()
    dest.write_bytes(data)
    return dest


def obtain(name: str) -> Path:
    local = _find_local(name)
    if local is not None:
        print(f"  found {name} -> {local}")
        return local
    return _download(name, RESOURCES[name])


def _write_member(name: str, payload: bytes, out_dir: Path) -> Path | None:
    base = Path(name).name
    if not SAFE_NAME.match(base):
        return None
    if base.endswith(".gz"):
        payload = gzip.decompress(payload)
        base = base[:-3]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / base
    out.write_bytes(payload)
    return out


def extract_tar(archive: Path, out_dir: Path) -> list[Path]:
    written = []
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            p = _write_member(m.name, f.read(), out_dir)
            if p:
                written.append(p)
    return written


def extract_zip(archive: Path, out_dir: Path) -> list[Path]:
    written = []
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"corrupt member {bad} in {archive}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = _write_member(info.filename, zf.read(info), out_dir)
            if p:
                written.append(p)
    return written


def verify_images(paths: list[Path]) -> list[str]:
    bad = []
    for p in paths:
        if p.suffix.lower() == ".ppm" and cv2.imread(str(p), cv2.IMREAD_UNCHANGED) is None:
            bad.append(p.name)
    return bad


def main() -> None:
    STARE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, int] = {}

    print("[1/5] fundus images")
    imgs = extract_zip(obtain("all-images.zip"), STARE_DIR / "images")
    report["images"] = len(imgs)

    print("[2/5] vessel reference masks (Hoover 'ah', Kouznetsova 'vk') + published results")
    report["vessels_ah"] = len(extract_tar(obtain("labels-ah.tar"), STARE_DIR / "vessels" / "ah"))
    report["vessels_vk"] = len(extract_tar(obtain("labels-vk.tar"), STARE_DIR / "vessels" / "vk"))
    report["hoover_results"] = len(extract_tar(obtain("results-4.tar"), STARE_DIR / "vessels" / "hoover_results"))

    print("[3/5] manifestation annotations")
    report["manifestations"] = len(extract_zip(obtain("annotations.zip"), STARE_DIR / "manifestations"))

    print("[4/5] diagnoses + optic nerve locations")
    for name in ("all-mg-codes.txt", "GT_NERVES.txt", "nervelist.txt", "manifestations.html"):
        shutil.copyfile(obtain(name), STARE_DIR / name)
    # The 81-image optic-nerve experiment set is the members of stare-images.tar
    ids: list[str] = []
    nerve_tar = _find_local("stare-images.tar")
    if nerve_tar is not None:
        with tarfile.open(nerve_tar) as tf:
            ids = sorted(Path(m.name).name.split(".")[0] for m in tf.getmembers() if m.isfile())
    if len(ids) != 81:  # fall back to the list published on the nerve page
        page = _download("nerve_index.html", "nerve/index.html").read_text("latin1")
        ids = sorted(set(re.findall(r"\b(im\d{4})\b", page)))
    (STARE_DIR / "nerve_experiment_ids.txt").write_text("\n".join(ids) + "\n")
    report["nerve_experiment_ids"] = len(ids)

    print("[5/5] verifying readability")
    all_ppm = sorted(STARE_DIR.rglob("*.ppm"))
    bad = verify_images(all_ppm)
    print(f"  checked {len(all_ppm)} images, unreadable: {len(bad)} {bad[:5]}")
    for k, v in report.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
