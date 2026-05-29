"""DeepLIIF Web Flask application.

Routes:
  GET  /                       upload form
  POST /upload                 run inference, redirect to /result/<id>
  GET  /result/<job_id>        render viewer
  GET  /image/<job_id>/<name>  serve orig.png or seg.png
"""
import io
import json
import threading

from flask import (Flask, abort, redirect, render_template, request,
                   send_from_directory, url_for)
from PIL import Image, UnidentifiedImageError

from . import config, inference, jobs

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_MB * 1024 * 1024

_INFERENCE_LOCK = threading.Lock()
_ALLOWED_IMAGE_NAMES = {'orig.png', 'seg.png'}

# 分辨率到 DeepLIIF tile_size 的映射(deepliif/models/__init__.py 的 resolution 推断)
_RESOLUTION_TILE_SIZE = {'10': 128, '20': 256, '40': 512}


@app.errorhandler(413)
def too_large(_e):
    return ('Image exceeds the configured upload size limit.', 413)


@app.route('/')
def index():
    return render_template('index.html',
                           max_mb=config.MAX_UPLOAD_MB,
                           max_edge=config.MAX_EDGE_PX)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        abort(400, 'no file field')
    f = request.files['file']
    if not f.filename:
        abort(400, 'empty filename')

    # Validate it is actually an image and within size limits.
    raw = f.read()
    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        abort(400, 'not a recognized image')

    # Re-open after verify (verify consumes the stream).
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    if max(img.size) > config.MAX_EDGE_PX:
        abort(413)

    # 解析分辨率选项,缺省 40X(tile_size=512)
    resolution = request.form.get('resolution', '40')
    tile_size = _RESOLUTION_TILE_SIZE.get(resolution, config.TILE_SIZE)

    # Create the job dir and write the input.
    job_dir = jobs.create_job_dir(config.JOBS_DIR)
    input_path = job_dir / 'input.png'
    img.save(input_path)

    # Serialize inference: one GPU, one job at a time.
    with _INFERENCE_LOCK:
        inference.run(input_path, job_dir, tile_size=tile_size,
                      original_name=f.filename)

    # Keep storage bounded.
    jobs.prune_jobs(config.JOBS_DIR, keep=config.MAX_JOBS)

    return redirect(url_for('result', job_id=job_dir.name))


@app.route('/result/<job_id>')
def result(job_id):
    job_dir = config.JOBS_DIR / job_id
    if not job_dir.is_dir():
        abort(404)
    # 推断推理时使用的 resolution (同 deepliif/models/__init__.py 的映射)
    meta_path = job_dir / 'meta.json'
    tile_size = config.TILE_SIZE
    image_name = ''
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        tile_size = meta.get('tile_size', tile_size)
        image_name = meta.get('original_name') or ''
    infer_resolution = 40 if tile_size > 384 else (20 if tile_size > 192 else 10)
    return render_template('result.html',
                           job_id=job_id,
                           infer_resolution=infer_resolution,
                           image_name=image_name)


@app.route('/image/<job_id>/<name>')
def image(job_id, name):
    if name not in _ALLOWED_IMAGE_NAMES:
        abort(404)
    job_dir = config.JOBS_DIR / job_id
    if not job_dir.is_dir():
        abort(404)
    return send_from_directory(job_dir, name, mimetype='image/png')


if __name__ == '__main__':
    if not (config.MODEL_DIR / 'train_opt.txt').exists():
        raise SystemExit(
            f'Model directory {config.MODEL_DIR} is missing train_opt.txt. '
            f'Unzip DeepLIIF_Latest_Model.zip into the worktree root.'
        )
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
