"""Standalone script to regenerate PosSegRecolor images from saved cell data.

PosSegRecolor = a white-background image where every positive-cell pixel is
painted with a marker-specific RGB color. The official ihc-cellseg branch
produces it inline via `compute_final_results_with_seg_color`; the OD-path
viewer (`cellseg_cli_od.py`) does not, because the official DeepLIIF branch has
no `seg_color` support.

This script rebuilds it offline from `<stem>_cells.json.gz` (produced by
`cellseg_cli_od.py --save-cells-gz`) plus the matching `<stem>.json` scoring
file (for the exact od/size thresholds). No re-inference, no re-segmentation:
the label mask is rebuilt from the stored per-cell boundaries, reusing the
official postprocessing helpers so the positive-cell set matches the OD-path
overlay exactly.

`--input-dir` may point at a leaf directory or any parent: with `--recursive`
(default) every `*_cells.json.gz` under it is processed. Each PosSegRecolor PNG
is written next to its source cell file unless `--output-dir` is given.

Only OD-path cell data (dataVersion 5 or 6) is supported; marker-path v4 data
has no `od` field.
"""
from __future__ import annotations

import gc
import gzip
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
from PIL import Image

# ===========================================================================
# DEFAULT CONFIG -- edit values here to change CLI defaults in one place.
# ===========================================================================

DEFAULT_INPUT_DIR = 'datasets/05_Results/C2_AYCH_Postop_TMA'
DEFAULT_OUTPUT_DIR = ''            # 空 = 各图写到对应 cell 文件的同级目录
DEFAULT_CELLS_PATTERN = '*_cells.json.gz'
DEFAULT_RECURSIVE = True           # 递归扫描子目录下的 cell 文件
DEFAULT_RECOLOR_RGB = ''           # 空 = 按文件名自动检测 marker 取色;否则 'R,G,B'
DEFAULT_LOG_MODE = 'w'

# 回退阈值:仅当某个 cell 文件找不到同名 scoring JSON 时才使用。正常情况下
# 阈值全部从 <stem>.json 读取,以与 _SegOverlaid.png 的 positive 集合一致。
FALLBACK_OD_THRESH_LOWER = 90      # Web 运行时默认下限 = 0(result.js resetOption)
FALLBACK_OD_THRESH_UPPER = 'none'
FALLBACK_SIZE_THRESH = 210
FALLBACK_SIZE_THRESH_UPPER = 2000

# 与 cellseg_cli.py 一致的 marker 池(自动取色用)
MARKERS = ['DEFAULT', 'CD4', 'CD8', 'CD20', 'CD56', 'CD68', 'CD138', 'CD163', 'FOXP3']

# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# 让共享的 utils/ 包可被 import(与 cellseg_cli_od.py 一致)
sys.path.insert(0, str(PROJECT_ROOT))

from deepliif.postprocessing import (
    LABEL_BORDER_NEG,
    LABEL_BORDER_POS,
    LABEL_POSITIVE,
    LABEL_UNKNOWN,
    create_outer_boundary,
    decode_cell_data_v4,
    enlarge_cell_boundaries,
    fill_cells,
    make_full_contour,
    mark_background,
)
from utils.handle_img import get_color_dict
from utils.handle_log import setup_logger


# ---------------------------------------------------------------------------
# 帮手


def resolve_project_path(path_str: str) -> Path:
    """Resolve a path relative to the project root directory."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_rgb(text: str) -> Optional[Tuple[int, int, int]]:
    """Parse 'R,G,B' into an int tuple; empty string -> None (auto mode)."""
    text = text.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) != 3:
        raise click.BadParameter(f"--recolor-rgb expects 'R,G,B', got '{text}'")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise click.BadParameter(f"--recolor-rgb values must be integers, got '{text}'") from exc
    if any(c < 0 or c > 255 for c in rgb):
        raise click.BadParameter(f'--recolor-rgb values must be in [0, 255], got {rgb}')
    return rgb  # type: ignore[return-value]


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


def parse_optional_int_option(_ctx, param, value):
    return parse_optional_int(value, param.name)


def parse_size_thresh_option(_ctx, param, value):
    text = str(value).strip().lower()
    if text == 'default':
        return 'default'
    return parse_optional_int(value, param.name)


def detect_marker_tag(filename: str, available_markers: List[str]) -> str:
    """Infer marker tag from file name (same logic as cellseg_cli.py)."""
    pool = {m.upper() for m in available_markers if m.upper() != 'DEFAULT'}
    for token in Path(filename).stem.replace('-', '_').split('_'):
        if token.upper() in pool:
            return token.upper()
    return 'DEFAULT'


def build_label_mask(cell_data: dict, height: int, width: int, *,
                     size_thresh, size_thresh_upper: Optional[int],
                     od_thresh_lower: Optional[int],
                     od_thresh_upper: Optional[int]) -> np.ndarray:
    """Rebuild the official cells_to_final_results label mask from cell data.

    Replays the exact mask pipeline so the positive-cell set matches the
    OD-path overlay produced by cellseg_cli_od.py.
    """
    mask = np.full((height, width), LABEL_UNKNOWN, dtype=np.uint8)
    version = cell_data['dataVersion']
    if size_thresh is None:
        size_thresh = 0
    elif size_thresh == 'default':
        size_thresh = cell_data['settings']['default_size_thresh']

    for cell in cell_data['cells']:
        # v6 为 base92 编码,需解码;v5 已是明文 dict
        c = decode_cell_data_v4(cell, v6=True) if version == 6 else cell

        if c['size'] > size_thresh and (size_thresh_upper is None or c['size'] < size_thresh_upper):
            is_pos = c['positive']
            # OD 门控:同 cells_to_final_results 的判定顺序
            if od_thresh_lower is not None and c['od'] < od_thresh_lower:
                is_pos = False
            elif od_thresh_upper is not None and c['od'] > od_thresh_upper:
                is_pos = False
            label = LABEL_BORDER_POS if is_pos else LABEL_BORDER_NEG
            for b in make_full_contour(c['boundary']):
                mask[b[1], b[0]] = label

    mark_background(mask)
    fill_cells(mask)
    create_outer_boundary(mask)
    enlarge_cell_boundaries(mask)
    enlarge_cell_boundaries(mask)
    return mask


def recolor_from_mask(mask: np.ndarray, seg_color: Tuple[int, int, int]) -> np.ndarray:
    """White background; paint every positive-cell pixel with seg_color."""
    pos = np.logical_or(mask == LABEL_BORDER_POS, mask == LABEL_POSITIVE)
    out = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    out[pos] = seg_color
    return out


def cell_stem(cells_file: Path) -> str:
    """Recover the original image stem from a <stem>_cells.json.gz filename."""
    name = cells_file.name
    if name.endswith('_cells.json.gz'):
        return name[:-len('_cells.json.gz')]
    return name.split('.')[0]


# ---------------------------------------------------------------------------
# CLI


@click.group()
def cli():
    """Regenerate PosSegRecolor images from saved OD-path cell data."""
    pass


@cli.command()
@click.option('--input-dir', default=DEFAULT_INPUT_DIR, show_default=True,
              help='Directory with cell-data files, or any parent of such dirs.')
@click.option('--output-dir', default=DEFAULT_OUTPUT_DIR, show_default=True,
              help='Output directory; empty = write next to each cell file.')
@click.option('--cells-pattern', default=DEFAULT_CELLS_PATTERN, show_default=True,
              help='Glob pattern for the gzipped cell-data files.')
@click.option('--recursive/--no-recursive', default=DEFAULT_RECURSIVE, show_default=True,
              help='Recursively scan sub-directories for cell-data files.')
@click.option('--recolor-rgb', default=DEFAULT_RECOLOR_RGB, callback=lambda c, p, v: parse_rgb(v),
              show_default=True,
              help="Fixed 'R,G,B' color; empty = auto-detect marker from filename.")
@click.option('--fallback-od-lower', default=FALLBACK_OD_THRESH_LOWER, type=int, show_default=True,
              help='OD lower threshold used only when a scoring JSON is missing.')
@click.option('--fallback-od-upper', default=FALLBACK_OD_THRESH_UPPER,
              callback=parse_optional_int_option, show_default=True,
              help="OD upper threshold fallback; int or 'none'.")
@click.option('--fallback-size-thresh', default=FALLBACK_SIZE_THRESH,
              callback=parse_size_thresh_option, show_default=True,
              help="Size lower threshold fallback; int or 'default'.")
@click.option('--fallback-size-upper', default=FALLBACK_SIZE_THRESH_UPPER,
              callback=parse_optional_int_option, show_default=True,
              help="Size upper threshold fallback; int or 'none'.")
@click.option('--log-mode', default=DEFAULT_LOG_MODE, show_default=True,
              help="Log file open mode ('w' or 'a').")
def recolor(input_dir, output_dir, cells_pattern, recursive, recolor_rgb,
            fallback_od_lower, fallback_od_upper, fallback_size_thresh,
            fallback_size_upper, log_mode):
    """Generate <stem>_PosSegRecolor.png for every cell-data file found."""
    input_path = resolve_project_path(input_dir)
    fixed_output = resolve_project_path(output_dir) if output_dir else None

    if not input_path.exists():
        raise click.ClickException(f"Input directory '{input_path}' not found.")

    sample_id = input_path.name
    log_dir = fixed_output if fixed_output else input_path
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir, name_prefix=f'{sample_id}_recolor', log_mode=log_mode)

    # 递归 / 单层扫描 cell 文件
    if recursive:
        cells_files = sorted(input_path.rglob(cells_pattern))
    else:
        cells_files = sorted(input_path.glob(cells_pattern))

    color_dict = get_color_dict(MARKERS, type='rgb')

    logger.info('=' * 70)
    logger.info(f'POS-SEG RECOLOR - {sample_id}')
    logger.info('=' * 70)
    logger.info(f'Input  : {input_path}  (recursive={recursive})')
    logger.info(f"Output : {fixed_output if fixed_output else 'next to each cell file'}")
    logger.info(f'Found {len(cells_files)} cell-data files')
    logger.info(f"Color  : {'fixed ' + str(recolor_rgb) if recolor_rgb else 'auto (marker from filename)'}")
    logger.info('-' * 70)

    with click.progressbar(cells_files, label=f'Recoloring {len(cells_files)}',
                           item_show_func=lambda x: x.name if x else '') as bar:
        for cells_file in bar:
            try:
                t0 = time.time()
                logger.info(f'\n=== {cells_file} ===')
                stem = cell_stem(cells_file)
                # 输出落到 cell 文件同级目录,除非显式指定 --output-dir
                out_dir = fixed_output if fixed_output else cells_file.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                with gzip.open(cells_file, 'rt', encoding='utf-8') as f:
                    cell_data = json.load(f)

                version = cell_data.get('dataVersion')
                if version not in (5, 6):
                    logger.error(f'Unsupported dataVersion={version} (need 5 or 6); skipping.')
                    continue

                size = cell_data.get('imageSize')
                if not size:
                    logger.error('cell_data has no imageSize; skipping.')
                    continue
                width, height = size['width'], size['height']

                # 阈值优先从同级 scoring JSON 读取,保证与 overlay 一致;
                # 缺失时退回 CLI 的 fallback-* 参数。
                scoring_file = cells_file.parent / f'{stem}.json'
                if scoring_file.exists():
                    scoring = json.loads(scoring_file.read_text())
                    size_thresh = scoring.get('size_thresh', fallback_size_thresh)
                    size_thresh_upper = scoring.get('size_thresh_upper', fallback_size_upper)
                    od_lower = scoring.get('od_thresh_lower', fallback_od_lower)
                    od_upper = scoring.get('od_thresh_upper', fallback_od_upper)
                    thresh_src = scoring_file.name
                else:
                    logger.warning(f'{scoring_file.name} missing; using fallback thresholds.')
                    size_thresh = fallback_size_thresh
                    size_thresh_upper = fallback_size_upper
                    od_lower = fallback_od_lower
                    od_upper = fallback_od_upper
                    thresh_src = 'fallback'

                # 决定染色:固定色优先,否则按文件名检测 marker
                if recolor_rgb is not None:
                    seg_color = recolor_rgb
                    marker_tag = 'fixed'
                else:
                    marker_tag = detect_marker_tag(stem, MARKERS)
                    seg_color = color_dict.get(marker_tag, color_dict['DEFAULT'])

                mask = build_label_mask(
                    cell_data, height, width,
                    size_thresh=size_thresh,
                    size_thresh_upper=size_thresh_upper,
                    od_thresh_lower=od_lower,
                    od_thresh_upper=od_upper,
                )
                recolor_img = recolor_from_mask(mask, seg_color)

                out_path = out_dir / f'{stem}_PosSegRecolor.png'
                Image.fromarray(recolor_img).save(out_path)
                logger.info(f'Saved: {out_path}  (marker={marker_tag}, color={seg_color}, '
                            f'thresholds from {thresh_src})')
                logger.info(f'Done in {time.time() - t0:.2f}s')
            except Exception:
                logger.exception(f'Failed on {cells_file}')
            finally:
                gc.collect()

    logger.info('=' * 70)
    logger.info(f'All files processed for {sample_id}')
    logger.info('=' * 70)


if __name__ == '__main__':
    cli()
