from pathlib import Path
import importlib
import os


def test_defaults():
    """All defaults are sensible and types are correct."""
    import Web.config as cfg
    importlib.reload(cfg)
    assert cfg.TILE_SIZE == 512
    assert cfg.MAX_UPLOAD_MB == 30
    assert cfg.MAX_EDGE_PX == 4096
    assert cfg.MAX_JOBS == 20
    assert isinstance(cfg.MODEL_DIR, Path)
    assert isinstance(cfg.JOBS_DIR, Path)
    assert cfg.MODEL_DIR.name == 'DeepLIIF_Latest_Model'
    assert cfg.JOBS_DIR.name == 'jobs'


def test_env_overrides(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv('DEEPLIIF_WEB_MODEL_DIR', '/tmp/fake_model')
    monkeypatch.setenv('DEEPLIIF_WEB_TILE_SIZE', '256')
    monkeypatch.setenv('DEEPLIIF_WEB_MAX_JOBS', '5')
    import Web.config as cfg
    importlib.reload(cfg)
    assert cfg.MODEL_DIR == Path('/tmp/fake_model')
    assert cfg.TILE_SIZE == 256
    assert cfg.MAX_JOBS == 5
