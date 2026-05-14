import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from PIL import Image
import pytest


def make_test_image(path: Path, size=(256, 256), color='white'):
    Image.new('RGB', size, color).save(path)


def test_run_writes_orig_and_seg(tmp_path):
    """run() saves orig.png + seg.png from the inputs."""
    input_path = tmp_path / 'input.png'
    make_test_image(input_path, (300, 200))
    job_dir = tmp_path / 'job1'
    job_dir.mkdir()

    fake_seg = Image.new('L', (300, 200), 200)
    with patch('Web.inference.infer_modalities',
               return_value=({'Seg': fake_seg}, None)) as mock_infer, \
         patch('Web.inference.get_opt',
               return_value=MagicMock(use_dp=False)):
        from Web.inference import run
        meta = run(input_path, job_dir)

    assert (job_dir / 'orig.png').exists()
    assert (job_dir / 'seg.png').exists()
    assert (job_dir / 'meta.json').exists()
    # orig roundtrip preserves size
    saved_orig = Image.open(job_dir / 'orig.png')
    assert saved_orig.size == (300, 200)
    # meta json
    meta_disk = json.loads((job_dir / 'meta.json').read_text())
    assert meta_disk['width'] == 300
    assert meta_disk['height'] == 200
    assert meta_disk['tile_size'] == meta['tile_size']
    mock_infer.assert_called_once()


def test_run_converts_non_rgb(tmp_path):
    """RGBA / L inputs are coerced to RGB before inference."""
    input_path = tmp_path / 'input.png'
    Image.new('RGBA', (128, 128), (255, 0, 0, 128)).save(input_path)
    job_dir = tmp_path / 'job'; job_dir.mkdir()

    captured = {}
    def capture(img, *a, **kw):
        captured['mode'] = img.mode
        return ({'Seg': Image.new('L', img.size)}, None)

    with patch('Web.inference.infer_modalities', side_effect=capture), \
         patch('Web.inference.get_opt', return_value=MagicMock(use_dp=False)):
        from Web.inference import run
        run(input_path, job_dir)

    assert captured['mode'] == 'RGB'


def test_run_oom_retries_with_smaller_tile(tmp_path):
    """On CUDA OOM, retry once at tile_size // 2."""
    import torch
    input_path = tmp_path / 'input.png'
    make_test_image(input_path)
    job_dir = tmp_path / 'job'; job_dir.mkdir()

    call_log = []
    def side_effect(img, tile_size, *a, **kw):
        call_log.append(tile_size)
        if len(call_log) == 1:
            raise torch.cuda.OutOfMemoryError('test')
        return ({'Seg': Image.new('L', img.size)}, None)

    with patch('Web.inference.infer_modalities', side_effect=side_effect), \
         patch('Web.inference.get_opt', return_value=MagicMock(use_dp=False)):
        from Web.inference import run
        run(input_path, job_dir, tile_size=512)

    assert call_log == [512, 256]
