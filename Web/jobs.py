"""Job-directory lifecycle for DeepLIIF Web."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def create_job_dir(root: Path) -> Path:
    """Create a fresh directory with a uuid4-hex name under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    job_dir = root / job_id
    job_dir.mkdir()
    return job_dir


def list_jobs(root: Path) -> list[Path]:
    """List subdirectories of `root`, newest first by mtime."""
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def prune_jobs(root: Path, keep: int) -> int:
    """Delete all but the newest `keep` job directories. Returns count deleted."""
    if keep < 0:
        raise ValueError(f'keep must be >= 0, got {keep}')
    victims = list_jobs(root)[keep:]
    for p in victims:
        shutil.rmtree(p, ignore_errors=True)
    return len(victims)
