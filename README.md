# Fundus Image Analysis & Retinal Screening (IVP project)

A complete classical image-processing system for retinal fundus photographs: a Python/FastAPI backend with
OpenCV + scikit-image, and a React + Vite + TypeScript frontend. You can select a retinal image, see the
original, inspect the labelled output of **every** processing technique in execution order, read the extracted
features, and compare against ground truth wherever the dataset provides it.

> Nothing here is a clinical diagnosis. Lesion detections are *algorithmic candidates*; screening flags are
> *experimental rules*.

---

## 1. Quick start

Prerequisites: Python ≥ 3.10, Node ≥ 20.

```bash
# backend
cd backend
pip install -r requirements.txt
python -m scripts.prepare_stare        # extracts/downloads STARE into data/stare (≈0.5 GB)
python -m scripts.build_inventory      # writes data/inventory.csv
python -m scripts.evaluate_stare       # optional: dataset-level evaluation (~90 s)
python -m uvicorn app.main:app --port 8000   # "python -m" avoids picking up another Python's uvicorn
```

```bash
# frontend (second terminal)
cd frontend
npm install
npm run dev                            # http://localhost:5173  (proxies /api and /files to :8000)
```

Tests: `cd backend && python -m pytest -q`. There are 20 tests: 16 unit tests plus 4 API tests, one of which is a
full end-to-end run on STARE im0077.

---

## 2. Datasets

| Dataset | Status | What was obtained | Used for |
|---|---|---|---|
| **STARE** (Clemson, public) | ✅ ready | 397 images (`all-images.zip`), vessel masks for 20 images from observers AH and VK, Hoover 2000 vessel results, `GT_NERVES.txt` optic-nerve centres (397 rows, 63 "not visible"), the 81-image nerve-experiment list, manifestation grades (373 files), diagnosis codes | vessel segmentation (pixel level), disc localisation (point level), DR lesions (image level), DR flag |
| **REFUGE** | ⛔ needs login | – | disc/cup masks → Dice, VCDR error, glaucoma flag |
| **IDRiD** | ⛔ needs login | – | MA/HE/EX masks → pixel and component lesion metrics |

The user-supplied archives in `data/` were reused. `GT_NERVES.txt`, `nervelist.txt` and `manifestations.html` were
missing and were downloaded from the official STARE site. Every image was verified as readable (457/457 PPM files,
including masks).
REFUGE (grand-challenge.org) and IDRiD (IEEE DataPort) require an account and accepting terms, so they were **not**
downloaded. The loaders are already implemented. See [data/README.md](data/README.md) for the exact manual steps;
once the files are extracted, pixel-level disc/cup and lesion evaluation switches on automatically.

**Annotation coverage is uneven, and the code uses only what exists.** STARE has *no* disc/cup masks, *no* lesion
masks and *no* glaucoma labels. Disc evaluation on STARE is therefore localisation-only, lesion evaluation is
image-level against manifestation grades, and VCDR/glaucoma cannot be validated on STARE.

Details discovered while preparing STARE:
- In a manifestation code, character *n* is manifestation *n*. This was verified against the counts published on
  the site. Grades used: 9 (MA/dot hemorrhage), 11–12 (hemorrhages), 13–14 (exudates, circinate pattern).
- `GT_NERVES.txt` stores coordinates as **row, column**. This was verified visually.
- In `data/`, `stare-images.tar` is the 81-image nerve set and `stare-images (1).tar` is the 20-image vessel set.

---

## 3. What is implemented (modules A–N)

Every step that produces an image is recorded as a numbered **stage** with its technique name, explanation,
parameters and output statistics. The stage list is saved to `processing_manifest.json`, and the frontend's
pipeline view is built from it, so the UI can only show steps that actually ran.

| Module | Technique (as executed) |
|---|---|
| Original | Byte-identical copy of the source file + 8-bit RGB working copy (grayscale, RGBA and 16-bit inputs are normalised; images with a side > 1000 px are down-scaled for processing). |
| C · ROI | Red channel → Gaussian → scaled-Otsu threshold → opening → largest component → hole filling → erosion to a "valid field". Area, diameter, centroid, perimeter, circularity. |
| A · Quality | 15 metrics inside the ROI (mean, median, min, max, std, range, Michelson, RMS and local contrast, dark/bright %, entropy, Laplacian sharpness, Canny edge density, Immerkær noise σ) + histograms. |
| B · Enhancement | HE (mapping from ROI pixels) and CLAHE (clip 2.0, 8×8) on CIELAB L*; side-by-side panels, metrics table, and the effect of each on downstream vessel segmentation. |
| D · Vessels | Green → CLAHE → Gaussian → multi-scale black-hat (7/11/15 px) → Otsu inside the field → connected-component area filter → skeleton. Area, density, length, mean/max width, perimeter, branch/end points, segments, branching density, tortuosity. |
| E · Optic disc | Vessel-free brightness map (closing + Gaussian) → top-5 peaks → for each candidate, L* window → recursive Otsu → ellipse fit → accepted if disc-shaped (area 0.25–3× expected, elongation ≤ 1.7). Sobel/Canny boundary maps. Full shape descriptors. |
| F · Optic cup | Independent green-channel threshold inside the disc (Otsu or configurable percentile) → ellipse fit → clipped to disc. VCDR, HCDR, ACDR, rim area, rim/disc ratio, superior/inferior/left/right rim thickness. Nasal/temporal is marked unavailable because eye laterality is unknown. |
| G · DR lesions | Shared: green → CLAHE → median 3×3 → median background. **MA**: black-hat, vessel suppression, mean+6σ, size/circularity/elongation filter. **HE**: background − image, mean+3σ, opening, vessel-overlap/elongation/fovea rules (fovea located automatically). **EX**: image − background, CIELAB yellowness/brightness rule, optic disc excluded. Per-component shape, colour and distance-to-disc. |
| H · Morphology | Reusable erosion, dilation, opening, closing, gradient, top-hat, black-hat, CC labelling, area filter, centroids, bounding boxes, shape descriptors, plus a labelled demonstration of each operator. |
| I · Spatial | D_all, D_MA, D_HE, D_EX over the valid field; four quadrants about the field centroid; lesion-to-disc distances; spatial map and quadrant chart. |
| J · GLCM | 32 levels with background excluded; distances 1/2/4; 4 angles; symmetric; contrast, homogeneity, energy, correlation, entropy (mean and per-distance). |
| K · Edges | Sobel, Prewitt, Canny (automatic hysteresis from gradient percentiles) and Laplacian-of-Gaussian, with parameters recorded. |
| L · Features | The 32-feature core set, each with unit and status (`available` / `unavailable` + reason). Missing values are `null`, never 0. Exported as CSV and JSON. |
| M · Evaluation | Pixel metrics computed **inside the retinal field only** (TP/FP/TN/FN, sensitivity, specificity, precision, recall, F1, Dice, IoU, ROC-AUC); localisation error; component matching (overlap after dilation); VCDR absolute error (with REFUGE). |
| N · Screening | JSON-configurable rules (`backend/fundus/screening_rules.json`) producing experimental glaucoma-suspect and DR-lesion flags, kept separate from measurements and dataset labels. |
| Report | PDF (ReportLab), feature CSV, analysis metadata, evaluation JSON, processing manifest. |

**Output layout** per image: `outputs/{dataset}/{image_id}/00_original … 09_report`, plus `10_morphology`,
`11_edges` and `processing_manifest.json`. This follows the requested structure; files are only written for steps
that ran.

---

## 4. Results on STARE (all numbers come from `python -m scripts.evaluate_stare`)

**Vessel segmentation** (20 images, reference = observer AH, pixels inside the retinal field, mean over images):

| Method | Sens. | Spec. | Prec. | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Ours (CLAHE) — default | 0.672 | 0.979 | 0.793 | **0.721** | 0.955 |
| Ours (no enhancement) | 0.597 | 0.986 | 0.838 | 0.691 | 0.954 |
| Ours (HE) | 0.523 | 0.985 | 0.801 | 0.629 | 0.916 |
| Hoover et al. 2000 (published output) | 0.687 | 0.955 | 0.644 | 0.657 | – |
| Second human observer (VK) | 0.896 | 0.938 | 0.645 | 0.742 | – |

The method was chosen by comparing black-hat, line-opening top-hat and Frangi on these same 20 images. There is
no separate vessel test set, so these figures are not independent.

**Optic-disc localisation** (success = within 60 px of `GT_NERVES`, the criterion from Hoover & Goldbaum 2003):
- 81-image hard set: **65.4 %** (median error 31 px). Hoover & Goldbaum's published result on the same set is 88.9 %.
- All 334 images with a visible nerve: 57.8 %.

**DR lesion candidates vs STARE manifestation grades** (image level; count threshold chosen on im0001–0100):

| Category | AUC of count (held-out) | Sens. / Spec. at tuned threshold (held-out) |
|---|---|---|
| Microaneurysm (≥ 15) | 0.796 | 0.717 / 0.832 |
| Hemorrhage (≥ 5) | 0.709 | 0.753 / 0.554 |
| Exudate (≥ 10) | 0.692 | 0.456 / 0.824 |

**Experimental DR flag** (≥ 2 of the 3 count conditions; reference = diagnosis codes 7/8): held-out
sensitivity 0.535 and specificity 0.775 (n = 298). I compared this rule with a grid-searched "any category" rule
*after* seeing held-out numbers, so this estimate is optimistic.

**Glaucoma / VCDR:** cannot be evaluated on STARE (no cup annotations, no glaucoma labels). The median predicted
VCDR is 0.75, which shows that colour-based cup segmentation over-estimates the cup. Treat CDR values as
unvalidated until REFUGE is added.

---

## 5. API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | status + dataset availability |
| `GET /api/datasets` · `GET /api/datasets/{id}/images` | datasets, images, annotation availability |
| `GET /api/images/{dataset}/{image}/metadata` · `/preview` | metadata, preview JPEG |
| `POST /api/upload` | upload (type, size ≤ 25 MB, decodability and ≥ 128 px are validated) |
| `POST /api/analyze` `{dataset, image_id}` | queue an analysis (202) |
| `GET /api/analysis/{id}` | status / current stage |
| `GET /api/analysis/{id}/artifacts` · `/results` · `/features` · `/features.csv` · `/evaluation` · `/report` · `/report.pdf` | outputs |
| `GET /api/evaluation/summary` | STARE dataset-level evaluation |
| `GET /files/...` | generated images (served read-only from `outputs/` only) |

Analysis ids are `{dataset}--{image_id}`. Ids are validated against `^[A-Za-z0-9_-]+$`, so no arbitrary paths are
accepted.

---

## 6. Frontend

The workspace has a dataset and image selector (default filter: images with vessel ground truth), upload, Run
analysis with live stage progress, the original image, and quality metrics. The tabs are:
- **Processing pipeline**: per-module flow strip plus stage cards (step number, technique name, enlargeable image,
  explanation, parameters, output statistics, next step). Lesion and disc/cup pipelines are split into sub-pipelines.
- **Comparison**: preset side-by-side comparisons (original/HE/CLAHE, ROI, vessels, disc/cup, lesions,
  prediction vs ground truth) and a free A/B picker.
- **Features**: the 32-feature table with status, quality definitions, enhancement bars, geometry, lesions,
  quadrants and GLCM.
- **Evaluation**: per-image metrics, ROC curve, enhancement effect, localisation, image-level lesion grades.
- **Report**: provenance, screening flags with every condition shown, and PDF/CSV/JSON downloads.
- **Dataset evaluation**: the STARE summary above.

**About "Gravitus":** no design system of that name exists on npm (404) and none was in the project. The only
"Gravitus" found in this environment is an unrelated fitness-programme service. The intended restrained, academic
look was reproduced with Tailwind CSS v4 design tokens (`frontend/src/index.css`) and Lucide icons.

---

## 7. Known limitations

- Disc localisation fails on images whose brightest region is a lesion (65 % on the hard set). When localisation
  is right, the recursive Otsu sometimes keeps only the bright pallor and under-segments the disc (for example
  im0235), which inflates CDR.
- Cup segmentation from colour alone is unreliable; CDR is unvalidated without REFUGE.
- Vessel false positives appear at exudate and hemorrhage edges. Thin vessels are under-segmented (sensitivity 0.67).
- Lesion candidate counts are never zero on normal images, so the screening rules use count thresholds. The rules
  were tuned on 100 STARE images with only 21 DR-positive cases.
- Measurements are in pixels; there is no mm calibration.
- Image-quadrant labels are image-based (upper/lower, left/right), not anatomical, because laterality is unknown.
