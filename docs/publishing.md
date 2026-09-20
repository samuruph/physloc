# Publishing to Hugging Face

For whoever uploads a finished run. Generation already writes the canonical schema-v4 tree, so
publishing is validate, package, inspect, upload, verify. Back to the [README](../README.md).

The dataset card that lands on the Hub is **generated**, not hand-written: `export` builds it from
`physloc/reference.py`, the same functions that write the README's tables, so the card cannot
drift from the dataset it describes.

Generation already writes the canonical schema-v4 tree. Publishing is a
validation, packaging, inspection, and upload sequence.

## 1. Validate the generated dataset

```bash
conda activate physloc

python -m physloc.cli validate out/review_conditions_f37
python -m physloc.cli stats out/review_conditions_f37
```

Validation must report `"ok": true`. The stats command writes the six review
figures and `stats/stats.json`. At the end of generation, PhysLoc also writes
`dataset.json`, `schema.json`, `index.parquet`, and `splits/` directly
into the generated root.

## 2. Build a clean publication directory

```bash
python -m physloc.cli export out/review_conditions_f37 \
  --outdir out/review_conditions_f37_hf \
  --license CC-BY-4.0
```

The source and destination must differ. The exported directory contains the
complete dataset card, licence, global metadata, JSON Schema, Parquet index,
split lists, standalone loader, and every `sample.json`, `rgb.mp4`, and
`data.h5`.

## 3. Inspect the packaged dataset locally

```bash
python test_dataset_loader.py out/review_conditions_f37_hf
python test_dataset_loader.py out/review_conditions_f37_hf --gui --port 8765
```

## 4. Authenticate once

```bash
hf auth login
```

Use a Hugging Face write token. Authentication is handled by the current
`hf` CLI; do not use the deprecated `huggingface-cli`.

## 5. Upload

The one-command PhysLoc route packages and uploads:

```bash
python -m physloc.cli export out/review_conditions_f37 \
  --outdir out/review_conditions_f37_hf \
  --license CC-BY-4.0 \
  --push-to YOUR_USERNAME/physloc-review-conditions
```

Add `--private` to create a private dataset repository:

```bash
python -m physloc.cli export out/review_conditions_f37 \
  --outdir out/review_conditions_f37_hf \
  --push-to YOUR_USERNAME/physloc-review-conditions \
  --private
```

Or upload an already-packaged directory directly:

```bash
hf upload YOUR_USERNAME/physloc-review-conditions \
  out/review_conditions_f37_hf . \
  --type dataset \
  --commit-message "Publish PhysLoc schema v4"
```

## 6. Download and verify

```bash
hf download YOUR_USERNAME/physloc-review-conditions \
  --repo-type dataset \
  --local-dir data/physloc-review-conditions

python test_dataset_loader.py data/physloc-review-conditions
python -m physloc.cli validate data/physloc-review-conditions
```

The Parquet index stores relative paths only; MP4 and HDF5 bytes are not
duplicated inside it. For a metadata-only inspection, download
`dataset.json`, `schema.json`, `index.parquet`, `splits/**`, and
`samples/**/sample.json`. Full localisation or energy experiments also need
the corresponding `data.h5`; RGB experiments need `rgb.mp4`.
