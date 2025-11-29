import os
import sys
import json
import time
import click
import torch
from PIL import Image
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from deepliif.models import infer_modalities
from deepliif.util import allowed_file
from deepliif.options import Options, print_options
from utils.handle_log import setup_logger
from utils.handle_img import get_color_dict

def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

@click.group()
def cli():
    """Commonly used DeepLIIF batch operations for cell segmentation"""
    pass

@cli.command()
@click.option('--input-dir', default='datasets/TMA_registered/20x/EC002-01/EC002-01-A2/reg_results', help='reads images from here')
@click.option('--output-dir', default='datasets/TMA_registered/20x/EC002-01/EC002-01-A2/seg_results', help='saves results here.')
@click.option('--tile-size', default=512, type=click.IntRange(min=1, max=None), help='tile size')
@click.option('--model-dir', default='./checkpoints/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--filename-pattern', default='*', help='run inference on files of which the name matches the pattern.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--seg-only', is_flag=True, default=True, help='save only the final segmentation image (currently only applies to DeepLIIF model); overwrites --seg-intermediate')
@click.option('--log-mode', default='w', help='Logging mode for cellseg_cli.py')
def test(input_dir, output_dir, tile_size, model_dir, filename_pattern, gpu_ids, eager_mode=False, epoch='latest',
         seg_intermediate=False, seg_only=True, color_dapi=False, color_marker=False, btoa=False, seg_color=True, log_mode='w'):
    
    """Test trained models
    """
    os.makedirs(output_dir, exist_ok=True)
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
    markers = ['DEFAULT', 'CD4', 'CD8', 'CD20', 'CD56', 'CD68', 'CD138', 'CD163', 'FOXP3']
    color_dict = get_color_dict(markers, type='rgb')
    
    if seg_intermediate and seg_only:
        seg_intermediate = False

    if filename_pattern == '*':
        print('use all alowed files')
        image_files = [fn for fn in os.listdir(input_dir) if allowed_file(fn)]
    else:
        import glob
        print('match files using filename pattern',filename_pattern)
        image_files = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, filename_pattern))]
    logger.info(f"Found {len(image_files)} images to process:")
    for img in image_files:
        logger.info(f"  - {img}")
    logger.info("-"*70)
    
    files = os.listdir(model_dir)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.use_dp = False
    opt.BtoA = btoa
    opt.epoch = epoch
    
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        number_of_gpus = 0
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
            
            # Get marker color
            if seg_color:
                seg_tag = filename.split('-')[2].upper()
                seg_color = color_dict[seg_tag] if seg_tag in color_dict else color_dict['DEFAULT']
            else:
                seg_color = color_dict['DEFAULT']
            logger.info(f"Parameters - Tile size: {tile_size}, Marker color: {seg_color}")
            
            # symlink original image
            img_input_path = Path(input_dir) / filename
            img_output_path = output_path / filename
            if img_output_path.is_symlink() or os.path.exists(img_output_path):
                os.unlink(img_output_path)
            # relative symlink
            relative_path = os.path.relpath(img_input_path, img_output_path.parent)
            os.symlink(relative_path, img_output_path)

            img = Image.open(img_input_path).convert('RGB')
            images, scoring = infer_modalities(img, tile_size, model_dir, seg_color, eager_mode, color_dapi, color_marker, opt, return_seg_intermediate=seg_intermediate, seg_only=seg_only)

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
