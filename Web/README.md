# DeepLIIF Web

## Install

```bash
pip install -r Web/requirements.txt
unzip -n DeepLIIF_Latest_Model.zip
```

## Run

```bash
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
| `DEEPLIIF_WEB_SEG_WEIGHTS` | `0.25,0.15,0.25,0.1,0.25` | Weights for the 5 seg generators (IHC, Hema, DAPI, Lap2, Marker). Default matches the official train-time weighting (`deepliif/scripts/train.py:338`). Override with comma list, e.g. `0.5,0,0,0,0.5` to match deepliif.org's website behavior, or `0.2,0.2,0.2,0.2,0.2` for uniform. |
| `DEEPLIIF_WEB_SEG_ONLY` | `1` | Pass `seg_only=True` to `infer_modalities` so unused Hema/DAPI/Lap2 modality generators are skipped (Marker stays because seg consumes it). Saves ~30-40% of inference time in the OD path. Set to `0` to restore full multi-modality inference. |

## Acknowledgement

DeepLIIF Web vendors client-side assets from https://deepliif.org

Source files retained verbatim under static/script/ and static/css/