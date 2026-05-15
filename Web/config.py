"""Runtime configuration for DeepLIIF Web.

All values can be overridden via env vars prefixed `DEEPLIIF_WEB_`.
"""
from __future__ import annotations

import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent  # .../models/DeepLIIF

def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val is not None else default


def _env_float_list(name: str, default: list[float]) -> list[float]:
    val = os.environ.get(name)
    if val is None:
        return default
    return [float(x) for x in val.split(',')]


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


MODEL_DIR: Path = _env_path(
    'DEEPLIIF_WEB_MODEL_DIR',
    _WORKTREE_ROOT / 'DeepLIIF_Latest_Model',
)
JOBS_DIR: Path = _env_path(
    'DEEPLIIF_WEB_JOBS_DIR',
    _THIS_DIR / 'jobs',
)
TILE_SIZE: int = _env_int('DEEPLIIF_WEB_TILE_SIZE', 512)
MAX_UPLOAD_MB: int = _env_int('DEEPLIIF_WEB_MAX_UPLOAD_MB', 30)
MAX_EDGE_PX: int = _env_int('DEEPLIIF_WEB_MAX_EDGE_PX', 4096)
MAX_JOBS: int = _env_int('DEEPLIIF_WEB_MAX_JOBS', 20)

# Seg generator weights [IHC, Hema, DAPI, Lap2, Marker].
SEG_WEIGHTS: list[float] = _env_float_list(
    'DEEPLIIF_WEB_SEG_WEIGHTS',
    [0.25, 0.15, 0.25, 0.10, 0.25],
)

# Pass `seg_only=True` to deepliif.infer_modalities so the unused Hema/DAPI/Lap2
SEG_ONLY: bool = _env_bool('DEEPLIIF_WEB_SEG_ONLY', True)
