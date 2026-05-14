"""Runtime configuration for DeepLIIF Web.

All values can be overridden via env vars prefixed `DEEPLIIF_WEB_`.
"""
import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent  # .../models/DeepLIIF_web


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val is not None else default


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
