import os
import json
import time
import click
import torch
import numpy as np
from PIL import Image

from deepliif.data import transform
from deepliif.models import infer_modalities, infer_results_for_wsi
from deepliif.util import allowed_file
from deepliif.options import Options, print_options

def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

@click.group()
def cli():
    """Commonly used DeepLIIF batch operations for cell segmentation"""
    pass

@cli.command()
# @click.option('--input-dir', default='./Sample_Large_Tissues/', help='reads images from here')
# @click.option('--output-dir', help='saves results here.')
# @click.option('--tile-size', type=click.IntRange(min=1, max=None), required=True, help='tile size')
# @click.option('--model-dir', default='./model-server/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--input-dir', default='datasets/TMA_registered/20x/EC002-01/EC002-01-A2/reg_results', help='reads images from here')
@click.option('--output-dir', default='datasets/TMA_registered/20x/EC002-01/EC002-01-A2/seg_results', help='saves results here.')
@click.option('--tile-size', default=512, type=click.IntRange(min=1, max=None), help='tile size')
@click.option('--model-dir', default='./checkpoints/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--region-size', default=20000, help='Due to limits in the resources, the whole slide image cannot be processed in whole.'
                                                   'So the WSI image is read region by region. '
                                                   'This parameter specifies the size each region to be read into GPU for inferrence.')
# @click.option('--eager-mode', is_flag=True, help='use eager mode (loading original models, otherwise serialized ones)')
# @click.option('--color-dapi', is_flag=True, help='color dapi image to produce the same coloring as in the paper')
# @click.option('--color-marker', is_flag=True, help='color marker image to produce the same coloring as in the paper')
def test(input_dir, output_dir, tile_size, model_dir, gpu_ids, region_size, eager_mode=False,
         color_dapi=False, color_marker=False, seg_color=True):
    """Test trained models
    """
    distinct_colors = [
        "#000000", "#5A0007", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    ]
    # HEX to RGB
    distinct_colors = [tuple(int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for h in
                       distinct_colors]
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
    print(color_dict)

    output_dir = output_dir or input_dir
    ensure_exists(output_dir)

    image_files = [fn for fn in os.listdir(input_dir) if allowed_file(fn)]
    image_files = sorted([f for f in image_files if 'reg' in f.lower()])
    print(image_files)
    files = os.listdir(model_dir)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.use_dp = False
    
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
            if '.svs' in filename:
                start_time = time.time()
                infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size)
                print(time.time() - start_time)
            else:
                if seg_color:
                    seg_tag = filename.split('-')[2].upper()
                    seg_color = color_dict[seg_tag] if seg_tag in color_dict else color_dict['DEFAULT']
                else:
                    seg_color = color_dict['DEFAULT']
                print(f"{filename}, tile_size: {tile_size}, seg_color: {seg_color}")
                
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                images, scoring = infer_modalities(img, tile_size, model_dir, seg_color, eager_mode, color_dapi, color_marker, opt)
                
                img_path = os.path.join(output_dir, filename)
                if not os.path.exists(img_path):
                    img.save(img_path)

                for name, i in images.items():
                    if name == "Mask":
                        image_name = filename.replace('.' + filename.split('.')[-1],
                                                      f'_pos-{scoring["num_pos"]}-all-{scoring["num_total"]}_{name}.png')
                    else:
                        image_name = filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                    i.save(os.path.join(output_dir, image_name))

                if scoring is not None:
                    with open(os.path.join(
                            output_dir,
                            filename.replace('.' + filename.split('.')[-1], f'.json')
                    ), 'w') as f:
                        json.dump(scoring, f, indent=2)

if __name__ == '__main__':
    cli()
