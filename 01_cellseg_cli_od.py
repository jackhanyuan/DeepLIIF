"""DeepLIIF batch CLI using the optical-density (OD) postprocessing path.

Mirrors the algorithm used by the Web viewer (`Web/`) so offline batch results
match what users see in the browser.

Pipeline (connected components are computed ONCE, then reused):

  inference(seg_only=True)        -> seg image only
  compute_cell_results(version=5) -> per-cell data {size, positive, od, ...}
  cells_to_final_results(data)    -> overlay + refined seg + scoring

The legacy marker-thresh path lives in `models/DeepLIIF/cellseg_cli.py` on the
ihc-cellseg branch and is left untouched.

OD threshold scale note:
    Browser JS  : od_uint8  = round(sum(log10(255/c)) * 35.3)   range [0, 255]
    Backend Py  : od_uint16 = round(sum(log10(255/c)) * 100)    range [0, ~722]
    Conversion  : od_py = round(od_js * 100 / 35.3) ~= round(od_js * 2.833)
    The Web viewer's runtime default OD lower bound is 0: the result.html
    attribute value="24" is immediately overwritten by result.js resetOption()
    (result.js:507), which sets the lower slider to 0. DEFAULT_OD_THRESH_LOWER
    is therefore 0 to match. Use the conversion above only when replicating a
    specific non-zero slider value a user dialed in.
"""
from __future__ import annotations

import gc
import gzip
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import click
import torch
from PIL import Image

# ===========================================================================
# DEFAULT CONFIG -- edit values here to change CLI defaults in one place.
# Every click option below simply references one of these constants.
# ===========================================================================

# --- Input / output paths (relative paths resolve against the project root) --
DEFAULT_INPUT_DIR = 'datasets/04_Registered/C2_AYCH_Postop_TMA/AY_LADX-31_20.0x/AY_LADX-31_20.0x_X1Y1'
DEFAULT_OUTPUT_DIR = 'datasets/05_Results/C2_AYCH_Postop_TMA/AY_LADX-31_20.0x/AY_LADX-31_20.0x_X1Y1/seg_od'
DEFAULT_MODEL_DIR = 'models/DeepLIIF/DeepLIIF_Latest_Model/'
DEFAULT_FILENAME_PATTERNS = ('*_reg.*',)
DEFAULT_GPU_IDS = ()  # 空 = 用全部 GPU;(-1,) = CPU

# --- Inference ---------------------------------------------------------------
DEFAULT_TILE_SIZE = 512                           # 512=40X, 256=20X, 128=10X
DEFAULT_SEG_WEIGHTS = '0.25,0.15,0.25,0.10,0.25'  # 与 Web/config.py SEG_WEIGHTS 一致

# --- Postprocessing thresholds ----------------------------------------------
DEFAULT_SEG_THRESH = 120          # 像素阴/阳分类阈值 (deepliif DEFAULT_SEG_THRESH)
DEFAULT_SIZE_THRESH = 210         # 细胞面积下限;整数 或 'default' (KDE 谷底)
DEFAULT_SIZE_THRESH_UPPER = 2000  # 细胞面积上限;整数 或 'none'
DEFAULT_OD_THRESH_LOWER = 90      # Web 运行时默认下限 = 0(# OD 量纲换算OD_PY = OD_JS * 100.0 / 35.3, Python 后端 ×100, 浏览器 JS ×35.3 )
DEFAULT_OD_THRESH_UPPER = 'none'  # 整数 或 'none' (不设上限,与 Web 默认一致)
DEFAULT_NOISE_THRESH = 4          # 微小噪声下限 (deepliif DEFAULT_NOISE_THRESH)
DEFAULT_LARGE_NOISE_THRESH = 'default' # 'none' (与 Web 一致) / 'default' (40X=16000px) / 整数

# --- Output options ----------------------------------------------------------
# save_seg/overlay/refined 控制全尺寸 PNG;save_thumb 独立控制缩略图。
# 三者解耦:可关掉全尺寸图(--no-save-overlay 等),只保留缩略图。
DEFAULT_SAVE_SEG = True
DEFAULT_SAVE_OVERLAY = True
DEFAULT_SAVE_REFINED = False
DEFAULT_SAVE_CELLS_GZ = True
DEFAULT_SAVE_THUMB = True
DEFAULT_SYMLINK_ORIG = True
DEFAULT_CELL_DATA_VERSION = 6       # 5 = OD 路径明文 dict;6 = OD 路径 base92 编码
DEFAULT_LOG_MODE = 'w'

# ===========================================================================

# Resolve project root so relative --input-dir / --output-dir mirror the layout
# used by the existing cellseg_cli.py on the ihc-cellseg branch.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# 让共享的 utils/ 包可被 import(与 cellseg_cli.py 一致的做法)
sys.path.insert(0, str(PROJECT_ROOT))

from deepliif.models import inference
from deepliif.options import Options, print_options
from deepliif.postprocessing import compute_cell_results, cells_to_final_results
from deepliif.util import allowed_file
from utils.handle_file import copy_related_files, link_or_copy
from utils.handle_log import setup_logger


# ---------------------------------------------------------------------------
# CLI 参数解析帮手


def resolve_project_path(path_str: str) -> Path:
    """Resolve a path relative to the project root directory."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_seg_weights(seg_weights: str, opt: Options) -> List[float]:
    """Parse comma-separated segmentation weights and validate them."""
    values = [float(v.strip()) for v in seg_weights.split(',') if v.strip()]
    expected = opt.modalities_no + 1 if opt.model in ['DeepLIIF', 'DeepLIIFKD'] else opt.modalities_no
    if len(values) != expected:
        raise click.BadParameter(
            f'Expected {expected} weights for model {opt.model}, got {len(values)}: {values}'
        )
    if abs(sum(values) - 1.0) > 1e-6:
        raise click.BadParameter(f'Segmentation weights must sum to 1.0, got {sum(values):.6f}')
    return values


def parse_optional_int(value, param_name: str) -> Optional[int]:
    """Parse a CLI option as int; empty/none/null -> None."""
    text = str(value).strip().lower()
    if text in ('', 'none', 'null'):
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise click.BadParameter(f"{param_name} must be an integer or 'none', got '{value}'") from exc
    if parsed < 0:
        raise click.BadParameter(f'{param_name} must be non-negative, got {parsed}')
    return parsed


def parse_size_thresh_option(_ctx, param, value):
    text = str(value).strip().lower()
    if text == 'default':
        return 'default'
    return parse_optional_int(value, param.name)


def parse_size_thresh_upper_option(_ctx, param, value):
    return parse_optional_int(value, param.name)


def parse_large_noise_thresh_option(_ctx, param, value):
    text = str(value).strip().lower()
    if text == 'default':
        return 'default'
    return parse_optional_int(value, param.name)


def parse_od_thresh_option(_ctx, param, value):
    return parse_optional_int(value, param.name)


# ---------------------------------------------------------------------------
# 文件工具


def collect_image_files(input_path: Path, filename_patterns) -> List[str]:
    """Collect image files matching one or more filename patterns."""
    patterns = [p for p in filename_patterns if p] or ['*_reg.*']
    files: List[str] = []
    seen = set()
    for pattern in patterns:
        if pattern == '*':
            matches = [f.name for f in sorted(input_path.iterdir()) if f.is_file() and allowed_file(f.name)]
        else:
            matches = [f.name for f in sorted(input_path.glob(pattern)) if f.is_file() and allowed_file(f.name)]
        for m in matches:
            if m not in seen:
                seen.add(m)
                files.append(m)
    return files


def build_thumbnail(image: Image.Image, max_side: int = 2048) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return thumb


def infer_resolution(tile_size: int) -> str:
    """Same mapping the deepliif backend uses for the DeepLIIF model family."""
    if tile_size > 384:
        return '40x'
    if tile_size > 192:
        return '20x'
    return '10x'


# ---------------------------------------------------------------------------
# CLI


@click.group()
def cli():
    """DeepLIIF batch operations using the OD postprocessing path."""
    pass


@cli.command()
@click.option('--input-dir', default=DEFAULT_INPUT_DIR, show_default=True,
              help='Directory of input images.')
@click.option('--output-dir', default=DEFAULT_OUTPUT_DIR, show_default=True,
              help='Directory for inference outputs.')
@click.option('--tile-size', default=DEFAULT_TILE_SIZE, type=click.IntRange(min=1),
              show_default=True, help='Tile size; 512=40X, 256=20X, 128=10X.')
@click.option('--model-dir', default=DEFAULT_MODEL_DIR, show_default=True,
              help='Path to DeepLIIF_Latest_Model directory.')
@click.option('--filename-pattern', 'filename_patterns', multiple=True,
              default=DEFAULT_FILENAME_PATTERNS, show_default=True,
              help='Glob pattern for input filenames; may be provided multiple times.')
@click.option('--gpu-ids', type=int, multiple=True, default=DEFAULT_GPU_IDS,
              help='GPU id(s); pass --gpu-ids -1 for CPU. Empty = all GPUs.')
@click.option('--seg-weights', default=DEFAULT_SEG_WEIGHTS, show_default=True,
              help='Comma-separated weights for the 5 seg generators (matches Web default).')
@click.option('--seg-thresh', default=DEFAULT_SEG_THRESH, type=click.IntRange(min=0, max=254),
              show_default=True, help='Pixel threshold for positive/negative classification.')
@click.option('--noise-thresh', default=DEFAULT_NOISE_THRESH, type=click.IntRange(min=0),
              show_default=True, help='Tiny-noise floor; cells smaller than this are dropped.')
@click.option('--size-thresh', default=DEFAULT_SIZE_THRESH, callback=parse_size_thresh_option,
              show_default=True, help="Min cell area (pixels^2); int or 'default' (KDE).")
@click.option('--size-thresh-upper', default=DEFAULT_SIZE_THRESH_UPPER,
              callback=parse_size_thresh_upper_option, show_default=True,
              help="Max cell area (pixels^2); int or 'none'.")
@click.option('--od-thresh-lower', default=DEFAULT_OD_THRESH_LOWER, callback=parse_od_thresh_option,
              show_default=True,
              help='Lower OD threshold (Python x100 scale). Default 0 = lower gate off (matches Web).')
@click.option('--od-thresh-upper', default=DEFAULT_OD_THRESH_UPPER, callback=parse_od_thresh_option,
              show_default=True, help="Upper OD threshold; int or 'none' for no cap.")
@click.option('--large-noise-thresh', default=DEFAULT_LARGE_NOISE_THRESH,
              callback=parse_large_noise_thresh_option, show_default=True,
              help="Large-noise gate; 'none' (matches Web), 'default' (40X=16000 px), or int.")
@click.option('--cell-data-version', default=str(DEFAULT_CELL_DATA_VERSION),
              type=click.Choice(['5', '6']), show_default=True,
              help='Per-cell data format: 5=plain dict, 6=base92-encoded.')
@click.option('--save-seg/--no-save-seg', default=DEFAULT_SAVE_SEG, show_default=True,
              help='Save the full-size raw segmentation PNG (<stem>_Seg.png).')
@click.option('--save-overlay/--no-save-overlay', default=DEFAULT_SAVE_OVERLAY, show_default=True,
              help='Save the full-size overlay PNG (<stem>_..._SegOverlaid.png).')
@click.option('--save-refined/--no-save-refined', default=DEFAULT_SAVE_REFINED, show_default=True,
              help='Save the full-size refined segmentation PNG (<stem>_SegRefined.png).')
@click.option('--save-cells-gz/--no-save-cells-gz', default=DEFAULT_SAVE_CELLS_GZ, show_default=True,
              help='Save per-cell OD data as <stem>_cells.json.gz.')
@click.option('--save-thumb/--no-save-thumb', default=DEFAULT_SAVE_THUMB, show_default=True,
              help='Save 2048-px thumbnails of the original image plus '
                   'Seg/SegOverlaid/SegRefined (independent of the '
                   '--save-seg/overlay/refined full-size switches).')
@click.option('--symlink-orig/--no-symlink-orig', default=DEFAULT_SYMLINK_ORIG, show_default=True,
              help='Symlink original input image into output dir for convenience.')
@click.option('--log-mode', default=DEFAULT_LOG_MODE, show_default=True,
              help="Log file open mode ('w' or 'a').")
def test(input_dir, output_dir, tile_size, model_dir, filename_patterns, gpu_ids,
         seg_weights, seg_thresh, noise_thresh, size_thresh, size_thresh_upper,
         od_thresh_lower, od_thresh_upper, large_noise_thresh, cell_data_version,
         save_seg, save_overlay, save_refined, save_cells_gz, save_thumb,
         symlink_orig, log_mode):
    """Run OD-path inference + postprocessing on a batch of images."""
    input_path = resolve_project_path(input_dir)
    output_path = resolve_project_path(output_dir)
    model_path = resolve_project_path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cell_data_version = int(cell_data_version)

    if not input_path.exists():
        raise click.ClickException(f"Input directory '{input_path}' not found.")
    if not model_path.exists():
        raise click.ClickException(f"Model directory '{model_path}' not found.")
    if not (model_path / 'train_opt.txt').exists():
        raise click.ClickException(f"train_opt.txt missing from model dir '{model_path}'.")

    sample_id = output_path.parent.name or output_path.name
    logger = setup_logger(output_path, name_prefix=f'{sample_id}_cellseg', log_mode=log_mode)

    logger.info('=' * 70)
    logger.info(f'CELL SEGMENTATION (OD path) - {sample_id}')
    logger.info('=' * 70)
    logger.info(f'Input  : {input_path}')
    logger.info(f'Output : {output_path}')
    logger.info(f'Model  : {model_path}')
    logger.info(f'tile_size={tile_size}, resolution={infer_resolution(tile_size)}')
    logger.info(f'seg_weights={seg_weights}')
    logger.info(f'seg_thresh={seg_thresh}, noise_thresh={noise_thresh}')
    logger.info(f'size_thresh={size_thresh}, size_thresh_upper={size_thresh_upper}')
    logger.info(f'od_thresh_lower={od_thresh_lower}, od_thresh_upper={od_thresh_upper}')
    logger.info(f'large_noise_thresh={large_noise_thresh}, cell_data_version={cell_data_version}')
    logger.info(f'save_seg={save_seg}, save_overlay={save_overlay}, save_refined={save_refined}, '
                f'save_cells_gz={save_cells_gz}, save_thumb={save_thumb}')
    logger.info('-' * 70)

    # Build opt object from the model's train_opt.txt.
    opt = Options(path_file=str(model_path / 'train_opt.txt'), mode='test')
    opt.use_dp = False
    opt.epoch = 'latest'

    # GPU 解析:同 cellseg_cli.py 习惯
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(gpu_ids)
    if n_gpus < len(gpu_ids) and -1 not in gpu_ids:
        logger.warning(f'Requested {len(gpu_ids)} GPUs but only {n_gpus} available. Falling back to CPU.')
        gpu_ids = [-1]
    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(n_gpus))
    opt.gpu_ids = gpu_ids

    # 兼容旧 train_opt.txt:targets_no -> modalities_no
    if not hasattr(opt, 'modalities_no') and hasattr(opt, 'targets_no'):
        opt.modalities_no = opt.targets_no - 1
        del opt.targets_no

    seg_weights_value = parse_seg_weights(seg_weights, opt)
    print_options(opt)

    # 把同目录的 HE 文件及其伴随文件软链进输出目录(与 cellseg_cli.py 一致)
    he_files = sorted(list(input_path.glob('*HE*.tif')) + list(input_path.glob('*HE*.tiff')), key=lambda x: x.name)
    if he_files:
        copied, message = copy_related_files(he_files[0], output_path, use_symlink=True)
        if copied:
            logger.info(f'{message} for {he_files[0].name}')

    image_files = collect_image_files(input_path, filename_patterns)
    logger.info(f'Found {len(image_files)} images:')
    for f in image_files:
        logger.info(f'  - {f}')
    logger.info('-' * 70)

    overlap_size = tile_size // 16
    resolution = infer_resolution(tile_size)

    with click.progressbar(image_files, label=f'Processing {len(image_files)}',
                           item_show_func=lambda x: x) as bar:
        for filename in bar:
            img = None
            images = None
            try:
                logger.info(f'\n=== {filename} ===')
                t0 = time.time()
                img_in = input_path / filename
                stem = Path(filename).stem

                if symlink_orig:
                    link_or_copy(img_in, output_path / filename, use_symlink=True)

                img = Image.open(img_in).convert('RGB')

                # 1) 推理:只跑 Seg(seg_only=True),跳过 Hema/DAPI/Lap2
                images = inference(
                    img, tile_size, overlap_size, str(model_path),
                    eager_mode=False, color_dapi=False, color_marker=False,
                    opt=opt, return_seg_intermediate=False,
                    seg_only=True, mod_only=False,
                    seg_weights=seg_weights_value,
                )
                if 'Seg' not in images:
                    logger.error('Inference did not produce a Seg image; skipping.')
                    continue
                seg = images['Seg']

                # 2) 后处理只算一次连通域:compute_cell_results -> data
                cell_data = compute_cell_results(
                    seg, img, resolution,
                    version=cell_data_version,
                    seg_thresh=seg_thresh,
                    noise_thresh=noise_thresh,
                    large_noise_thresh=large_noise_thresh,
                )
                # cell_data 只记录"决定细胞集合"的参数:seg/noise/large_noise
                # (compute_cell_results 自带) + seg_weights/tile_size/resolution。
                # size/od 阈值属于分类阶段,不写入 cell_data。
                cell_data['settings']['tile_size'] = tile_size
                cell_data['settings']['resolution'] = resolution
                cell_data['settings']['seg_weights'] = seg_weights_value
                cell_data['imageSize'] = {'width': seg.size[0], 'height': seg.size[1]}
                cell_data['num_cells'] = len(cell_data['cells'])

                # 3) 从 data 派生 overlay/refined/scoring(不重算连通域)
                overlay, refined, scoring = cells_to_final_results(
                    cell_data, img,
                    size_thresh=size_thresh,
                    marker_thresh=None,
                    size_thresh_upper=size_thresh_upper,
                    od_thresh_lower=od_thresh_lower,
                    od_thresh_upper=od_thresh_upper,
                )
                pos, total = scoring['num_pos'], scoring['num_total']

                # scoring 记录这次计数结果 + 决定它的全部阈值。marker_thresh 是
                # cells_to_final_results 硬塞的,OD 路径恒为 None,删除以免误读。
                scoring.pop('marker_thresh', None)
                scoring['num_cells'] = cell_data['num_cells']
                scoring['noise_thresh'] = noise_thresh
                scoring['large_noise_thresh'] = large_noise_thresh
                scoring['od_thresh_lower'] = od_thresh_lower
                scoring['od_thresh_upper'] = od_thresh_upper
                scoring['tile_size'] = tile_size
                scoring['resolution'] = resolution
                scoring['seg_weights'] = seg_weights_value

                # 4) 保存结果文件;命名与 cellseg_cli.py 对齐。
                # 三张结果图都在内存生成;全尺寸 PNG 是否落盘由各自 save_* 开关决定,
                # 缩略图由 save_thumb 独立决定 —— thumb_targets 收集原图 + 三张结果图。
                thumb_targets = []

                # 原图缩略图:与 Seg/Overlay/Refined 一同生成,便于阈值核对时对照
                thumb_targets.append((filename, img))

                seg_name = f'{stem}_Seg.png'
                if save_seg:
                    seg.save(output_path / seg_name)
                    logger.info(f'Saved: {seg_name}')
                thumb_targets.append((seg_name, seg))

                overlay_img = Image.fromarray(overlay)
                overlay_name = f'{stem}_pos-{pos}-all-{total}_SegOverlaid.png'
                if save_overlay:
                    overlay_img.save(output_path / overlay_name)
                    logger.info(f'Saved: {overlay_name}')
                thumb_targets.append((overlay_name, overlay_img))

                refined_img = Image.fromarray(refined)
                refined_name = f'{stem}_SegRefined.png'
                if save_refined:
                    refined_img.save(output_path / refined_name)
                    logger.info(f'Saved: {refined_name}')
                thumb_targets.append((refined_name, refined_img))

                (output_path / f'{stem}.json').write_text(json.dumps(scoring, indent=2))
                logger.info(f'Saved: {stem}.json')
                logger.info(f"  total={total}  pos={pos}  neg={scoring['num_neg']}  "
                            f"percent_pos={scoring['percent_pos']:.1f}%")

                # 5) per-cell OD 数据(可选);复用步骤 2 的结果,不重算
                if save_cells_gz:
                    cell_path = output_path / f'{stem}_cells.json.gz'
                    with gzip.open(cell_path, 'wt', encoding='utf-8') as f:
                        json.dump(cell_data, f)
                    logger.info(f'Saved: {cell_path.name}')

                # 6) 缩略图:对三张图各生成一张,与全尺寸图开关无关
                if save_thumb:
                    for name, im in thumb_targets:
                        thumb = build_thumbnail(im)
                        thumb_name = f'{Path(name).stem}_thumb.png'
                        thumb.save(output_path / thumb_name)
                        thumb.close()
                        logger.info(f'Saved: {thumb_name}')

                logger.info(f'Done in {time.time() - t0:.2f}s')
            except Exception:
                logger.exception(f'Failed on {filename}')
            finally:
                if images is not None:
                    for v in images.values():
                        if isinstance(v, Image.Image):
                            v.close()
                if img is not None:
                    img.close()
                del images, img
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info('=' * 70)
    logger.info(f'All images processed for {sample_id}')
    logger.info('=' * 70)


if __name__ == '__main__':
    cli()
