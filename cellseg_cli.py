import os
import sys
import json
import gzip
import time
import gc
import click
import torch
from PIL import Image
from pathlib import Path
from typing import List

# 添加项目根目录到 Python 路径
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from deepliif.models import infer_modalities, build_cellseg_runtime_config, DEFAULT_EMPTY_TILE_VAR_THRESH, find_marker_key
from deepliif.util import allowed_file
from deepliif.options import Options, print_options
from deepliif.postprocessing import (
    DEFAULT_SEG_THRESH,
    DEFAULT_MARKER_SCORE_MODE,
    compute_cell_results,
    infer_resolution_from_tile_size,
)
from utils.handle_log import setup_logger
from utils.handle_img import get_color_dict
from utils.handle_file import link_or_copy, copy_related_files


def resolve_project_path(path_str: str) -> Path:
    """Resolve a path relative to the project root directory."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def detect_marker_tag(filename: str, available_markers: List[str]) -> str:
    """Infer marker tag from file name."""
    marker_pool = {marker.upper() for marker in available_markers if marker.upper() != 'DEFAULT'}
    tokens = Path(filename).stem.replace('-', '_').split('_')
    for token in tokens:
        token_upper = token.upper()
        if token_upper in marker_pool:
            return token_upper
    return 'DEFAULT'


def parse_seg_weights(seg_weights: str, opt: Options) -> List[float]:
    """Parse segmentation weights from CLI string."""
    values = [float(value.strip()) for value in seg_weights.split(',') if value.strip()]
    expected = opt.modalities_no + 1 if opt.model in ['DeepLIIF', 'DeepLIIFKD'] else opt.modalities_no
    if len(values) != expected:
        raise click.BadParameter(
            f'Expected {expected} weights for model {opt.model}, got {len(values)}: {values}'
        )
    if abs(sum(values) - 1.0) > 1e-6:
        raise click.BadParameter(f'Segmentation weights must sum to 1.0, got {sum(values):.6f}')
    return values


def parse_non_negative_int(value, param_name: str):
    """Parse a non-negative integer CLI option."""
    if isinstance(value, int):
        parsed = value
    else:
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise click.BadParameter(f"{param_name} must be an integer, got '{value}'") from exc

    if parsed < 0:
        raise click.BadParameter(f'{param_name} must be non-negative, got {parsed}')
    return parsed


def parse_size_thresh_option(ctx, param, value):
    text = str(value).strip().lower()
    if text == 'default':
        return 'default'
    return parse_non_negative_int(value, param.name)


def parse_size_thresh_upper_option(ctx, param, value):
    text = str(value).strip().lower()
    if text in ['', 'none', 'null']:
        return None
    return parse_non_negative_int(value, param.name)


def parse_marker_thresh_option(ctx, param, value):
    text = str(value).strip().lower()
    if text in ['', 'none', 'null']:
        return None
    if text == 'default':
        return 'default'
    return parse_non_negative_int(value, param.name)


def build_thumbnail(image: Image.Image, max_side: int = 2048) -> Image.Image:
    """Create a thumbnail image while preserving aspect ratio."""
    thumb = image.copy()
    thumb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return thumb


def save_cell_data_gzip(images, tile_size, model_name, seg_thresh, output_dir: Path, filename: str, logger):
    """Save compressed cell-level results derived from in-memory Seg/Marker images."""
    if 'Seg' not in images:
        logger.warning('Skipping cell-data export because Seg image is unavailable')
        return

    marker_key = find_marker_key(images)
    marker_image = images.get(marker_key) if marker_key is not None else None
    resolution = infer_resolution_from_tile_size(tile_size, model_name)
    cell_data = compute_cell_results(
        images['Seg'],
        marker_image,
        resolution,
        version=4,
        seg_thresh=seg_thresh,
        large_noise_thresh='default',
    )
    cell_data['settings']['tile_size'] = tile_size
    cell_data['settings']['resolution'] = resolution
    cell_data['settings']['marker_score_mode'] = DEFAULT_MARKER_SCORE_MODE
    cell_data['imageSize'] = {
        'width': images['Seg'].size[0],
        'height': images['Seg'].size[1],
    }
    cell_data['num_cells'] = len(cell_data['cells'])
    cell_path = output_dir / f"{Path(filename).stem}_cells.json.gz"
    with gzip.open(cell_path, 'wt', encoding='utf-8') as f:
        json.dump(cell_data, f)
    logger.info(f"Saved: {cell_path.name}")


def collect_image_files(input_path: Path, filename_patterns):
    """Collect image files for one or more filename patterns."""
    patterns = [pattern for pattern in filename_patterns if pattern]
    if not patterns:
        patterns = ['*_reg.*']

    image_files = []
    seen = set()
    for pattern in patterns:
        if pattern == '*':
            matches = [fn.name for fn in sorted(input_path.iterdir()) if fn.is_file() and allowed_file(fn.name)]
        else:
            matches = [fn.name for fn in sorted(input_path.glob(pattern)) if fn.is_file() and allowed_file(fn.name)]

        for match in matches:
            if match not in seen:
                seen.add(match)
                image_files.append(match)

    return image_files


@click.group()
def cli():
    """Commonly used DeepLIIF batch operations for cell segmentation"""
    pass


@cli.command()
@click.option('--input-dir', default='datasets/04_Registered/C2_AYCH_Postop_TMA/AY_LADX-31_20.0x/AY_LADX-31_20.0x_X1Y1', help='reads images from here')
@click.option('--output-dir', default='datasets/05_Results/C2_AYCH_Postop_TMA/AY_LADX-31_20.0x/AY_LADX-31_20.0x_X1Y1/seg_results', help='saves results here.')
@click.option('--tile-size', default=512, type=click.IntRange(min=1, max=None), help='Tile size used for inference and postprocessing resolution inference')
@click.option('--model-dir', default='models/DeepLIIF/checkpoints/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--filename-pattern', 'filename_patterns', multiple=True, default=('*_reg.*',), help='run inference on files of which the name matches the pattern; may be provided multiple times.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--seg-intermediate', is_flag=True, help='also save intermediate segmentation images (currently only applies to DeepLIIF model)')
@click.option('--seg-only', is_flag=True, default=True, help='save only the final segmentation image (currently only applies to DeepLIIF model); overwrites --seg-intermediate')
@click.option('--mod-only', is_flag=True, help='save only translated modality images; overwrites --seg-only and --seg-intermediate')
@click.option('--eager-mode', is_flag=True, help='use eager mode (loading original models instead of serialized ones)')
@click.option('--epoch', default='latest', help='for eager mode, which epoch to load')
@click.option('--color-dapi', is_flag=True, help='color dapi image to produce the same coloring as in the paper')
@click.option('--color-marker', is_flag=True, help='color marker image to produce the same coloring as in the paper')
@click.option('--btoa', is_flag=True, help='for unaligned models, load generatorB instead of generatorA')
@click.option('--seg-color', default=True, help='enable marker-specific positive segmentation recoloring')
@click.option('--seg-weights', default='', help='comma-separated weights used to aggregate segmentation branches, for example 0.5,0,0,0,0.5')
@click.option('--seg-thresh', default=DEFAULT_SEG_THRESH, type=click.IntRange(min=0, max=254), show_default=True, help='Pixel threshold for segmentation; accepts an integer in [0, 254]')
@click.option('--size-thresh', default='default', callback=parse_size_thresh_option, show_default=True, help="Minimum cell area threshold; use an integer or 'default'")
@click.option('--size-thresh-upper', default='none', callback=parse_size_thresh_upper_option, show_default=True, help="Maximum cell area threshold; use an integer or 'none'")
@click.option('--marker-thresh-lower', default=0, type=click.IntRange(min=0, max=255), show_default=True, help='Marker lower gate; cells below this value are forced negative')
@click.option('--marker-thresh-upper', default=255, type=click.IntRange(min=0, max=255), show_default=True, help='Marker upper gate; cells above this value are forced positive')
@click.option('--marker-thresh', 'marker_thresh_legacy', default='none', callback=parse_marker_thresh_option, hidden=True)
@click.option('--empty-tile-var-thresh', default=DEFAULT_EMPTY_TILE_VAR_THRESH, type=click.IntRange(min=0, max=None), show_default=True, help='Gray-variance threshold for skipping empty tiles; accepts a non-negative integer')
@click.option('--save-thumb', is_flag=True, help='save thumbnail previews for generated result images')
@click.option('--log-mode', default='w', help='Logging mode for cellseg_cli.py')
def test(input_dir, output_dir, tile_size, model_dir, filename_patterns, gpu_ids, seg_only,
         seg_intermediate, mod_only, eager_mode, epoch, color_dapi, color_marker, btoa,
         seg_color, seg_weights, seg_thresh, size_thresh, size_thresh_upper,
         marker_thresh_lower, marker_thresh_upper, marker_thresh_legacy,
         empty_tile_var_thresh, save_thumb, log_mode):
    """Test trained models"""
    input_path = resolve_project_path(input_dir)
    output_path = resolve_project_path(output_dir)
    model_path = resolve_project_path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory '{input_path}' not found.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory '{model_path}' not found.")
    sample_id = output_path.parent.name

    if mod_only:
        seg_only = False
        seg_intermediate = False
    elif seg_intermediate and seg_only:
        seg_intermediate = False

    if marker_thresh_legacy is not None:
        if marker_thresh_lower != 0 or marker_thresh_upper != 255:
            raise click.BadParameter(
                'Use either --marker-thresh or the --marker-thresh-lower/--marker-thresh-upper pair, not both'
            )
        marker_thresh_lower = 0
        marker_thresh_upper = marker_thresh_legacy

    if marker_thresh_upper == 'default':
        if marker_thresh_lower != 0:
            raise click.BadParameter('--marker-thresh-upper default requires --marker-thresh-lower 0')
    elif marker_thresh_lower > marker_thresh_upper:
        raise click.BadParameter(
            f'marker-thresh-lower must be less than or equal to marker-thresh-upper, got {marker_thresh_lower} > {marker_thresh_upper}'
        )

    logger = setup_logger(
        save_dir=str(output_path),
        name_prefix=f"{sample_id}_cellseg",
        log_mode=log_mode
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"{'CELL SEGMENTATION - ' + sample_id:^70}")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Tile size: {tile_size}")
    logger.info(f"Model directory: {model_path}")
    logger.info(f"Eager mode: {eager_mode}")
    logger.info(f"Epoch: {epoch}")
    logger.info(f"Seg only: {seg_only}")
    logger.info(f"Seg intermediate: {seg_intermediate}")
    logger.info(f"Mod only: {mod_only}")
    logger.info(f"Color dapi: {color_dapi}")
    logger.info(f"Color marker: {color_marker}")
    logger.info(f"Marker recolor: {seg_color}")
    logger.info(f"Seg threshold: {seg_thresh}")
    logger.info(f"Size threshold: {size_thresh}")
    logger.info(f"Size threshold upper: {size_thresh_upper}")
    logger.info(f"Marker threshold lower: {marker_thresh_lower}")
    logger.info(f"Marker threshold upper: {marker_thresh_upper}")
    logger.info(f"Empty tile variance threshold: {empty_tile_var_thresh}")
    logger.info(f"Save thumbnail previews: {save_thumb}")
    logger.info(f"Logging mode: {log_mode}")
    logger.info("-" * 70)

    markers = ['DEFAULT', 'CD4', 'CD8', 'CD20', 'CD56', 'CD68', 'CD138', 'CD163', 'FOXP3']
    color_dict = get_color_dict(markers, type='rgb')

    he_files = sorted(list(input_path.glob('*HE*.tif')) + list(input_path.glob('*HE*.tiff')), key=lambda x: x.name)
    if he_files:
        he_file = he_files[0]
        copied, message = copy_related_files(he_file, output_path, use_symlink=True)
        if copied:
            logger.info(f"{message} for {he_file.name}")

    image_files = collect_image_files(input_path, filename_patterns)
    logger.info(f"Found {len(image_files)} images to process:")
    for img in image_files:
        logger.info(f"  - {img}")
    logger.info("-" * 70)

    files = os.listdir(model_path)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_path}'
    opt = Options(path_file=os.path.join(model_path, 'train_opt.txt'), mode='test')
    opt.use_dp = False
    opt.BtoA = btoa
    opt.epoch = epoch
    if seg_weights:
        seg_weights_value = parse_seg_weights(seg_weights, opt)
        seg_weights_source = 'cli'
    elif hasattr(opt, 'seg_weights'):
        seg_weights_value = list(opt.seg_weights)
        seg_weights_source = 'model_options'
    else:
        seg_weights_value = None
        seg_weights_source = 'official_default'
    logger.info(f"Segmentation weights ({seg_weights_source}): {seg_weights_value if seg_weights_value is not None else 'default'}")

    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        gpu_ids = [-1]
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))

    opt.gpu_ids = gpu_ids

    if not hasattr(opt, 'modalities_no') and hasattr(opt, 'targets_no'):
        opt.modalities_no = opt.targets_no - 1
        del opt.targets_no
    opt.cellseg_runtime = build_cellseg_runtime_config(
        seg_thresh=seg_thresh,
        size_thresh=size_thresh,
        size_thresh_upper=size_thresh_upper,
        marker_thresh_lower=marker_thresh_lower,
        marker_thresh_upper=marker_thresh_upper,
        empty_tile_var_thresh=empty_tile_var_thresh,
    )
    print_options(opt)

    with click.progressbar(
        image_files,
        label=f'Processing {len(image_files)} images',
        item_show_func=lambda fn: fn
    ) as bar:
        for filename in bar:
            img = None
            images = None
            scoring = None
            logger.info("\n" + "=" * 50)
            logger.info(f"Processing image: {filename}")
            start_time = time.time()

            try:
                if seg_color:
                    marker_tag = detect_marker_tag(filename, markers)
                    seg_color_value = color_dict.get(marker_tag, color_dict['DEFAULT'])
                else:
                    seg_color_value = color_dict['DEFAULT']
                logger.info(f"Parameters - Tile size: {tile_size}, Marker color: {seg_color_value}")

                img_input_path = input_path / filename
                img_output_path = output_path / filename
                link_or_copy(img_input_path, img_output_path, use_symlink=True)

                img = Image.open(img_input_path).convert('RGB')
                images, scoring = infer_modalities(
                    img,
                    tile_size,
                    str(model_path),
                    eager_mode=eager_mode,
                    color_dapi=color_dapi,
                    color_marker=color_marker,
                    opt=opt,
                    return_seg_intermediate=seg_intermediate,
                    seg_only=seg_only,
                    mod_only=mod_only,
                    seg_weights=seg_weights_value,
                    seg_color=seg_color_value,
                )
                if scoring is not None:
                    scoring['tile_size'] = tile_size
                    scoring['empty_tile_var_thresh'] = empty_tile_var_thresh

                if scoring is not None:
                    save_cell_data_gzip(
                        images,
                        tile_size,
                        opt.model,
                        seg_thresh,
                        output_path,
                        filename,
                        logger,
                    )

                for name, image in images.items():
                    if name == 'Seg' and scoring is not None:
                        image_name = filename.replace(
                            '.' + filename.split('.')[-1],
                            f'_pos-{scoring["num_pos"]}-all-{scoring["num_total"]}_{name}.png'
                        )
                    else:
                        image_name = filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                    image.save(output_path / image_name)
                    logger.info(f"Saved: {image_name}")
                    if save_thumb:
                        thumb_name = Path(image_name).with_suffix('').as_posix() + '_thumb.png'
                        thumb = build_thumbnail(image)
                        thumb.save(output_path / thumb_name)
                        thumb.close()
                        logger.info(f"Saved: {thumb_name}")

                if scoring is not None:
                    json_path = output_path / filename.replace('.' + filename.split('.')[-1], '.json')
                    with open(json_path, 'w') as f:
                        json.dump(scoring, f, indent=2)
                    logger.info("\nCell counting results:")
                    logger.info(f"  Total cells: {scoring['num_total']}")
                    logger.info(f"  Positive cells: {scoring['num_pos']}")
                    logger.info(f"  Negative cells: {scoring['num_neg']}")
                    logger.info(f"  Positive rate: {scoring['percent_pos']:.1f}%")

                logger.info(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
                logger.info("-" * 50)
            finally:
                if images is not None:
                    for image in images.values():
                        if isinstance(image, Image.Image):
                            image.close()
                if img is not None:
                    img.close()
                del images
                del img
                del scoring
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info(f"\nAll images processed for {sample_id}")
    logger.info("=" * 70)


if __name__ == '__main__':
    cli()
