# DeepLIIF Web (OD path)

Flask viewer that reproduces the official deepliif.org result page on top
of the official DeepLIIF branch. The browser handles all OD / size
threshold interactivity; the backend only runs the model and serves two
PNGs per job.

## Install

```bash
cd models/DeepLIIF_web
# Use the same Python env that already runs DeepLIIF inference.
pip install -r Web/requirements.txt
unzip -n DeepLIIF_Latest_Model.zip
```

## Run

```bash
cd models/DeepLIIF_web
python -m Web.app
# open http://localhost:5000
```

## Environment variables

| Name | Default | Notes |
|---|---|---|
| `DEEPLIIF_WEB_MODEL_DIR` | `../DeepLIIF_Latest_Model` (sibling) | must contain `train_opt.txt` |
| `DEEPLIIF_WEB_JOBS_DIR` | `Web/jobs/` | runtime artefacts |
| `DEEPLIIF_WEB_TILE_SIZE` | `512` | inference tile size |
| `DEEPLIIF_WEB_MAX_UPLOAD_MB` | `30` | per-request size cap |
| `DEEPLIIF_WEB_MAX_EDGE_PX` | `4096` | longest-edge cap |
| `DEEPLIIF_WEB_MAX_JOBS` | `20` | older jobs are pruned |

## Tests

```bash
python -m pytest Web/tests/ -v
```

The `test_e2e.py` integration test is auto-skipped if the model
directory or sample image isn't present.

## Manual acceptance checklist

1. Upload `Sample_Large_Tissues/ROI_1.png`.
2. Result page loads within ~90 s on a single GPU.
3. Drag **OD lower** slider from 0 → max: cells flip red ↔ blue, counts
   update.
4. Drag **OD upper** slider from max → 0: confirm band-pass behavior
   (very high OD cells also drop to negative).
5. Drag **size min/max**: KDE distribution chart and cell coloring
   update in sync.
6. "Show Size" mode + click a cell: cell size appears as a yellow tick
   on the distribution chart.
7. "Select Region" / "Exclude Region" lasso: drawing a lasso changes the
   counts.
8. **View** toggle switches between IHC and Segmentation overlays.
9. **Download** button packages PNGs into a zip.

## Architecture

See `docs/superpowers/specs/2026-05-14-deepliif-web-design.md`.
