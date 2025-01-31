import os
import sys
import json
import time
import click
import torch
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
import subprocess

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from deepliif.data import transform
from deepliif.models import infer_modalities, infer_results_for_wsi
from deepliif.util import allowed_file
from deepliif.options import Options, print_options
from utils.handle_log import setup_logger

def ensure_exists(d):
    # Clean directory if exists
    # if os.path.exists(d):
    #     shutil.rmtree(d)
    # Create new directory
    os.makedirs(d, exist_ok=True)

@click.group()
def cli():
    """Commonly used DeepLIIF batch operations for cell segmentation"""
    pass

@cli.command()
@click.option('--input-dir', default='datasets/TMA_registered/2.5x/EC001-01/EC001-01-A7/reg_results', help='reads images from here')
@click.option('--output-dir', default='datasets/TMA_registered/2.5x/EC001-01/EC001-01-A7/seg_results', help='saves results here.')
@click.option('--tile-size', default=512, type=click.IntRange(min=1, max=None), help='tile size')
@click.option('--model-dir', default='./checkpoints/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--region-size', default=20000, help='Due to limits in the resources, the whole slide image cannot be processed in whole.\nSo the WSI image is read region by region. \nThis parameter specifies the size each region to be read into GPU for inferrence.')
@click.option('--log-mode', default='w', help='Logging mode for cellseg_cli.py')
def test(input_dir, output_dir, tile_size, model_dir, gpu_ids, region_size, eager_mode=False,
         color_dapi=False, color_marker=False, seg_color=True, log_mode='w'):
    """Test trained models"""
    ensure_exists(output_dir)
    
    # Get sample ID from the path
    output_path = Path(output_dir)
    sample_id = output_path.parent.name  # Get the core directory name (EC00X-0X-AX)
    
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
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tile size: {tile_size}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Logging mode: {log_mode}")
    logger.info("-"*70)
    
    # Color configuration for different markers
    distinct_colors = [
        "#000000", "#5A0007", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    ]
    # HEX to RGB
    distinct_colors = [tuple(int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for h in distinct_colors]
    color_dict = {
        'DEFAULT': distinct_colors[0],
        'CD4': distinct_colors[1],
        'CD8': distinct_colors[2],
        'CD20': distinct_colors[3],
        'CD56': distinct_colors[4],
        'CD68': distinct_colors[5],
        'CD138': distinct_colors[6],
        'CD163': distinct_colors[7],
        'FOXP3': distinct_colors[8],
        'PDL1': distinct_colors[9]
    }

    # Get image files
    image_files = [fn for fn in os.listdir(input_dir) if allowed_file(fn)]
    image_files = sorted([f for f in image_files if 'reg' in f.lower()])
    logger.info(f"Found {len(image_files)} images to process:")
    for img in image_files:
        logger.info(f"  - {img}")
    logger.info("-"*70)
    
    # Load model configuration
    files = os.listdir(model_dir)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.use_dp = False
    
    # GPU configuration
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        number_of_gpus = 0
        gpu_ids = [-1]
        logger.warning(f'Requested GPU(s) {opt.gpu_ids} not available. Found {number_of_gpus_all} GPU(s). Switching to CPU.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))
    
    opt.gpu_ids = gpu_ids
    
    if not hasattr(opt,'modalities_no') and hasattr(opt,'targets_no'):
        opt.modalities_no = opt.targets_no - 1
        del opt.targets_no
    print_options(opt)

    # Process each image
    with click.progressbar(
            image_files,
            label=f'Processing {len(image_files)} images',
            item_show_func=lambda fn: fn
    ) as bar:
        for filename in bar:
            logger.info("\n" + "="*50)
            logger.info(f"Processing image: {filename}")
            start_time = time.time()
            
            if '.svs' in filename:
                infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size)
                logger.info(f"WSI processing completed in {time.time() - start_time:.2f} seconds")
            else:
                # Get marker color
                if seg_color:
                    seg_tag = filename.split('-')[2].upper()
                    seg_color = color_dict[seg_tag] if seg_tag in color_dict else color_dict['DEFAULT']
                else:
                    seg_color = color_dict['DEFAULT']
                logger.info(f"Parameters - Tile size: {tile_size}, Marker color: {seg_color}")
                
                # Process image
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                images, scoring = infer_modalities(img, tile_size, model_dir, seg_color, eager_mode, color_dapi, color_marker, opt)
                
                # Save original image
                img_path = os.path.join(output_dir, filename)
                if not os.path.exists(img_path):
                    img.save(img_path)

                # Save result images
                for name, i in images.items():
                    if name == "Seg":
                        image_name = filename.replace('.' + filename.split('.')[-1],
                                                      f'_pos-{scoring["num_pos"]}-all-{scoring["num_total"]}_{name}.png')
                    else:
                        image_name = filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                    i.save(os.path.join(output_dir, image_name))
                    logger.info(f"Saved: {image_name}")

                # Save and log cell counts
                if scoring is not None:
                    json_path = os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], f'.json'))
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
