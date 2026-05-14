import io
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with jobs dir redirected to tmp_path."""
    monkeypatch.setenv('DEEPLIIF_WEB_JOBS_DIR', str(tmp_path / 'jobs'))
    # Avoid model dir checks in tests.
    monkeypatch.setenv('DEEPLIIF_WEB_MODEL_DIR', str(tmp_path / 'model'))
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model' / 'train_opt.txt').write_text('# stub\n')

    import importlib
    import Web.config as cfg; importlib.reload(cfg)
    import Web.app as appmod; importlib.reload(appmod)
    appmod.app.config['TESTING'] = True
    with appmod.app.test_client() as c:
        yield c


def _png_bytes(size=(64, 64), color='red') -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


def test_index_renders_upload_form(client):
    r = client.get('/')
    assert r.status_code == 200
    assert b'DeepLIIF' in r.data
    assert b'type="file"' in r.data or b'type=file' in r.data


def test_upload_redirects_to_result(client, tmp_path):
    fake_meta = {'width': 64, 'height': 64, 'tile_size': 512}
    def fake_run(input_path, job_dir, tile_size=None):
        Image.new('RGB', (64, 64), 'red').save(job_dir / 'orig.png')
        Image.new('L', (64, 64), 128).save(job_dir / 'seg.png')
        (job_dir / 'meta.json').write_text('{}')
        return fake_meta

    with patch('Web.app.inference.run', side_effect=fake_run):
        data = {'file': (io.BytesIO(_png_bytes()), 'sample.png')}
        r = client.post('/upload', data=data, content_type='multipart/form-data',
                        follow_redirects=False)
    assert r.status_code == 302
    assert '/result/' in r.location


def test_result_page_injects_imageid(client, tmp_path):
    def fake_run(input_path, job_dir, tile_size=None):
        Image.new('RGB', (32, 32)).save(job_dir / 'orig.png')
        Image.new('L', (32, 32)).save(job_dir / 'seg.png')
        return {'width': 32, 'height': 32, 'tile_size': 512}

    with patch('Web.app.inference.run', side_effect=fake_run):
        data = {'file': (io.BytesIO(_png_bytes((32, 32))), 's.png')}
        r = client.post('/upload', data=data, content_type='multipart/form-data')
    job_url = r.location
    r2 = client.get(job_url)
    assert r2.status_code == 200
    assert b'window.imageID' in r2.data
    assert b'window.imageNames' in r2.data
    # Required DOM ids exist (the JS audits these on load).
    assert b'id="result-canvas"' in r2.data
    assert b'id="slider-marker-lower"' in r2.data
    assert b'id="slider-size-min"' in r2.data


def test_image_endpoint_serves_orig_and_seg(client):
    def fake_run(input_path, job_dir, tile_size=None):
        Image.new('RGB', (40, 40), 'blue').save(job_dir / 'orig.png')
        Image.new('L', (40, 40), 99).save(job_dir / 'seg.png')
        return {'width': 40, 'height': 40, 'tile_size': 512}

    with patch('Web.app.inference.run', side_effect=fake_run):
        data = {'file': (io.BytesIO(_png_bytes((40, 40))), 's.png')}
        r = client.post('/upload', data=data, content_type='multipart/form-data')
    job_id = r.location.rstrip('/').split('/')[-1]

    r_orig = client.get(f'/image/{job_id}/orig.png')
    assert r_orig.status_code == 200
    assert r_orig.mimetype == 'image/png'
    r_seg = client.get(f'/image/{job_id}/seg.png')
    assert r_seg.status_code == 200

    r_bad = client.get(f'/image/{job_id}/foo.png')
    assert r_bad.status_code == 404


def test_upload_rejects_oversized_image(client, monkeypatch):
    monkeypatch.setenv('DEEPLIIF_WEB_MAX_EDGE_PX', '100')
    import importlib, Web.config as cfg; importlib.reload(cfg)
    import Web.app as appmod; importlib.reload(appmod)
    with appmod.app.test_client() as c:
        big = _png_bytes((200, 50))
        r = c.post('/upload',
                   data={'file': (io.BytesIO(big), 'big.png')},
                   content_type='multipart/form-data')
        assert r.status_code == 413


def test_upload_rejects_non_image(client):
    r = client.post('/upload',
                    data={'file': (io.BytesIO(b'not an image'), 'x.png')},
                    content_type='multipart/form-data')
    assert r.status_code == 400
