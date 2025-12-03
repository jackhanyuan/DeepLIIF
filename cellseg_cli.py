import os
import sys
import json
import time
import click
import torch
from PIL import Image
from pathlib import Path
from typing import List

# add project root directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from deepliif.models import infer_modalities
from deepliif.util import allowed_file
from deepliif.options import Options, print_options
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

@click.group()
def cli():
    """Commonly used DeepLIIF batch operations for cell segmentation"""
    pass

@cli.command()
@click.option('--input-dir', default='datasets/04_Registered/C2_AYCH_Postop_TMA/AY_LADX-31_20.0x/AY_LADX-31_20.0x_X1Y1', help='reads images from here')
@click.option('--output-dir', default='datasets/05_Results/C2_AYCH_Postop_TMA/AY_LADX-31_20.0x/AY_LADX-31_20.0x_X1Y1/seg_results', help='saves results here.')
@click.option('--tile-size', default=512, type=click.IntRange(min=1, max=None), help='tile size')
@click.option('--model-dir', default='models/DeepLIIF/checkpoints/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--filename-pattern', default='*_reg.*', help='run inference on files of which the name matches the pattern.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--seg-only', is_flag=True, default=True, help='save only the final segmentation image (currently only applies to DeepLIIF model); overwrites --seg-intermediate')
@click.option('--log-mode', default='w', help='Logging mode for cellseg_cli.py')
def test(input_dir, output_dir, tile_size, model_dir, filename_pattern, gpu_ids, eager_mode=False, epoch='latest',
         seg_intermediate=False, seg_only=True, color_dapi=False, color_marker=False, btoa=False, seg_color=True, log_mode='w'):
    
    """Test trained models
    """
    input_path = resolve_project_path(input_dir)
    output_path = resolve_project_path(output_dir)
    model_path = resolve_project_path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory '{input_path}' not found.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory '{model_path}' not found.")
    sample_id = output_path.parent.name  # Get the core directory name

    # Setup logger
    logger = setup_logger(
        save_dir=str(output_path),
        name_prefix=f"{sample_id}_cellseg",
        log_mode=log_mode
    )

    # Log initialization
    logger.info("\n" + "="*70)
    logger.info(f"{'CELL SEGMENTATION - ' + sample_id:^70}")
    logger.info("="*70)
    logger.info("Configuration:")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Tile size: {tile_size}")
    logger.info(f"Model directory: {model_path}")
    logger.info(f"Logging mode: {log_mode}")
    logger.info("-"*70)
    
    # Color configuration for different markers
    markers = ['DEFAULT', 'CD4', 'CD8', 'CD20', 'CD56', 'CD68', 'CD138', 'CD163', 'FOXP3']
    color_dict = get_color_dict(markers, type='rgb')

    # Copy or link HE and related files to seg_results
    he_files = sorted(list(input_path.glob('*HE*.tif')) + list(input_path.glob('*HE*.tiff')), key=lambda x: x.name)
    if he_files:
        he_file = he_files[0]
        copied, message = copy_related_files(he_file, output_path, use_symlink=True)
        if copied:
            logger.info(f"{message} for {he_file.name}")
    
    if seg_intermediate and seg_only:
        seg_intermediate = False

    if filename_pattern == '*':
        print('use all alowed files')
        image_files = [fn.name for fn in sorted(input_path.iterdir()) if fn.is_file() and allowed_file(fn.name)]
    else:
        print('match files using filename pattern',filename_pattern)
        image_files = [fn.name for fn in sorted(input_path.glob(filename_pattern)) if fn.is_file() and allowed_file(fn.name)]
    logger.info(f"Found {len(image_files)} images to process:")
    for img in image_files:
        logger.info(f"  - {img}")
    logger.info("-"*70)
    
    files = os.listdir(model_path)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_path}'
    opt = Options(path_file=os.path.join(model_path,'train_opt.txt'), mode='test')
    opt.use_dp = False
    opt.BtoA = btoa
    opt.epoch = epoch
    
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        gpu_ids = [-1]
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))
    
    opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
    
    # fix opt from old settings
    if not hasattr(opt,'modalities_no') and hasattr(opt,'targets_no'):
        opt.modalities_no = opt.targets_no - 1
        del opt.targets_no
    print_options(opt)

    with click.progressbar(
            image_files,
            label=f'Processing {len(image_files)} images',
            item_show_func=lambda fn: fn
    ) as bar:
        for filename in bar:
            logger.info("\n" + "="*50)
            logger.info(f"Processing image: {filename}")
            start_time = time.time()
            
            # 获取染色标记以安全映射颜色
            if seg_color:
                marker_tag = detect_marker_tag(filename, markers)
                seg_color_value = color_dict.get(marker_tag, color_dict['DEFAULT'])
            else:
                seg_color_value = color_dict['DEFAULT']
            logger.info(f"Parameters - Tile size: {tile_size}, Marker color: {seg_color_value}")
            
            # symlink original image
            img_input_path = input_path / filename
            img_output_path = output_path / filename
            link_or_copy(img_input_path, img_output_path, use_symlink=True)

            img = Image.open(img_input_path).convert('RGB')
            images, scoring = infer_modalities(img, tile_size, str(model_path), seg_color_value, eager_mode, color_dapi, color_marker, opt, return_seg_intermediate=seg_intermediate, seg_only=seg_only)
            scoring['tile_size'] = tile_size

            for name, i in images.items():
                if name == "Seg":
                    image_name = filename.replace('.' + filename.split('.')[-1],
                                                    f'_pos-{scoring["num_pos"]}-all-{scoring["num_total"]}_{name}.png')
                else:
                    image_name = filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                i.save(output_path / image_name)
                logger.info(f"Saved: {image_name}")

            # Save and log cell counts
            if scoring is not None:
                json_path = output_path / filename.replace('.' + filename.split('.')[-1], f'.json')
                with open(json_path, 'w') as f:
                    json.dump(scoring, f, indent=2)
                logger.info("\nCell counting results:")
                logger.info(f"  Total cells: {scoring['num_total']}")
                logger.info(f"  Positive cells: {scoring['num_pos']}")
                logger.info(f"  Negative cells: {scoring['num_neg']}")
                logger.info(f"  Positive rate: {scoring['percent_pos']:.1f}%")
            
            logger.info(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
            logger.info("-"*50)
        
    logger.info(f"\nAll images processed for {sample_id}")
    logger.info("="*70)


if __name__ == '__main__':
    cli()
