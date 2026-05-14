import time
from pathlib import Path

import pytest


def test_create_returns_unique_dir(tmp_path):
    from Web.jobs import create_job_dir
    a = create_job_dir(tmp_path)
    b = create_job_dir(tmp_path)
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert a.parent == tmp_path
    # job id is a uuid4 hex string
    assert len(a.name) == 32 and all(c in '0123456789abcdef' for c in a.name)


def test_list_jobs_by_mtime_descending(tmp_path):
    from Web.jobs import create_job_dir, list_jobs
    a = create_job_dir(tmp_path)
    time.sleep(0.02)
    b = create_job_dir(tmp_path)
    time.sleep(0.02)
    c = create_job_dir(tmp_path)
    ordered = list_jobs(tmp_path)
    assert ordered == [c, b, a]


def test_prune_keeps_newest_n(tmp_path):
    from Web.jobs import create_job_dir, prune_jobs
    paths = []
    for _ in range(5):
        paths.append(create_job_dir(tmp_path))
        time.sleep(0.02)
    prune_jobs(tmp_path, keep=3)
    survivors = sorted(p.name for p in tmp_path.iterdir())
    expected = sorted(p.name for p in paths[-3:])
    assert survivors == expected


def test_prune_keep_zero_clears_all(tmp_path):
    from Web.jobs import create_job_dir, prune_jobs
    create_job_dir(tmp_path)
    create_job_dir(tmp_path)
    prune_jobs(tmp_path, keep=0)
    assert list(tmp_path.iterdir()) == []


def test_list_jobs_ignores_files(tmp_path):
    from Web.jobs import create_job_dir, list_jobs
    create_job_dir(tmp_path)
    (tmp_path / 'stray.txt').write_text('nope')
    result = list_jobs(tmp_path)
    assert len(result) == 1
    assert result[0].is_dir()
