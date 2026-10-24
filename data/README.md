# Data folder

Datasets are **not** committed (see `.gitignore`). Expected layout:

```
data/
├── *.zip / *.tar / *.txt            STARE archives as downloaded (reused if present)
├── raw/stare/                       anything prepare_stare.py downloaded itself
├── stare/                           created by `python -m scripts.prepare_stare`
│   ├── images/imXXXX.ppm            397 fundus images (700x605)
│   ├── vessels/ah|vk/               20 manual vessel masks per observer
│   ├── vessels/hoover_results/      Hoover et al. 2000 algorithm output (baseline, not GT)
│   ├── manifestations/*.fea.mg.txt  image-level manifestation grades
│   ├── GT_NERVES.txt                optic-nerve centre per image ("row col", -1 = not visible)
│   ├── nervelist.txt                Hoover & Goldbaum 2003 localisation result (baseline)
│   ├── nerve_experiment_ids.txt     the 81-image optic-nerve experiment set
│   └── all-mg-codes.txt             diagnosis codes
├── refuge/                          (manual) extract REFUGE here
├── idrid/                           (manual) extract IDRiD here
├── uploads/                         images uploaded through the web UI
└── inventory.csv / inventory.json   created by `python -m scripts.build_inventory`
```

## Manual steps (login required — cannot be automated)

### REFUGE (optic disc / cup masks, glaucoma labels)
1. Create a free account at <https://grand-challenge.org> and sign in.
2. Open <https://refuge.grand-challenge.org/REFUGE2Download/>, accept the challenge terms and download the
   training images and the `Annotation-Training400` disc/cup masks.
3. Extract everything into `data/refuge/` (any sub-folder structure). The loader pairs an image with a mask of the
   same file stem located under a folder whose path contains `mask`, `Disc_Cup` or `Annotation`. Glaucoma labels are
   read from `Glaucoma` / `Non-Glaucoma` folder names or `g####` / `n####` file names.

### IDRiD (MA / HE / EX / SE / OD masks, DR grades)
1. Create a free IEEE account and sign in at
   <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>.
2. Download `A. Segmentation.zip` (and optionally `B. Disease Grading.zip`).
3. Extract into `data/idrid/`. Files are matched by name: `IDRiD_NN.jpg` with `IDRiD_NN_MA.tif`, `_HE`, `_EX`, `_SE`, `_OD`.

Then restart the backend and run `python -m scripts.build_inventory`. The datasets appear in the UI automatically, and
pixel-level disc/cup and lesion evaluation switches on for those images.
