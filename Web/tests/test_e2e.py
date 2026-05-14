"""End-to-end test that exercises the real DeepLIIF model.

Skipped automatically when the model dir or CUDA is unavailable.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

WORKTREE = Path(__file__).resolve().parents[2]
MODEL_DIR = WORKTREE / 'DeepLIIF_Latest_Model'
SAMPLE = WORKTREE / 'Web' / 'static' / 'samples' / 'ROI_1.png'  # 使用已知存在的样本路径

requires_model = pytest.mark.skipif(
    not (MODEL_DIR / 'train_opt.txt').exists() or not SAMPLE.exists(),
    reason='DeepLIIF model or sample image not available',
)


@requires_model
def test_full_pipeline_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPLIIF_WEB_JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setenv('DEEPLIIF_WEB_MODEL_DIR', str(MODEL_DIR))
    monkeypatch.setenv('DEEPLIIF_WEB_TILE_SIZE', '512')
    monkeypatch.setenv('DEEPLIIF_WEB_MAX_EDGE_PX', '4096')

    import importlib
    import Web.config as cfg; importlib.reload(cfg)
    import Web.app as appmod; importlib.reload(appmod)
    appmod.app.config['TESTING'] = True

    with appmod.app.test_client() as c:
        with open(SAMPLE, 'rb') as f:
            data = {'file': (f, 'ROI_1.png')}
            r = c.post('/upload', data=data,
                       content_type='multipart/form-data',
                       follow_redirects=False)
        assert r.status_code == 302
        job_id = r.location.rstrip('/').split('/')[-1]

        r_orig = c.get(f'/image/{job_id}/orig.png')
        r_seg = c.get(f'/image/{job_id}/seg.png')
        assert r_orig.status_code == 200
        assert r_seg.status_code == 200

        orig = np.array(Image.open(io.BytesIO(r_orig.data)).convert('RGB'))
        seg = np.array(Image.open(io.BytesIO(r_seg.data)))
        # the seg map should match orig spatial size
        assert seg.shape[:2] == orig.shape[:2]
        # seg shouldn't be entirely zero (model produced something)
        assert seg.max() > 10
