"""Thin wrapper around deepliif.infer_modalities that persists job artefacts."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from deepliif.models import get_opt, infer_modalities

from . import config


def run(input_path: Path, job_dir: Path, tile_size: int | None = None) -> dict:
    """Run DeepLIIF inference and persist orig.png, seg.png, meta.json.

    On CUDA OOM at the chosen tile_size, retries once at tile_size // 2.

    Returns the meta dict written to disk.
    """
    if tile_size is None:
        tile_size = config.TILE_SIZE

    img = Image.open(input_path).convert('RGB')  # 确保输入为 RGB 格式
    opt = get_opt(str(config.MODEL_DIR))
    opt.use_dp = False  # 禁用数据并行，单 GPU 推理

    images = _infer_with_oom_retry(img, tile_size, opt)

    # 将浏览器需要加载的两个输出文件持久化到磁盘
    img.save(job_dir / 'orig.png')
    images['Seg'].save(job_dir / 'seg.png')

    meta = {
        'width': img.size[0],
        'height': img.size[1],
        'tile_size': tile_size,
    }
    (job_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return meta


def _infer_with_oom_retry(img: Image.Image, tile_size: int, opt) -> dict:
    """Run infer_modalities with OOM fallback to half tile size.

    Passes config.SEG_WEIGHTS so the aggregated Seg matches deepliif.org's
    weighting (IHC + Marker dominant) rather than the official Python
    branch's newer uniform default.
    """
    try:
        images, _ = infer_modalities(img, tile_size, str(config.MODEL_DIR),
                                     opt=opt, seg_weights=config.SEG_WEIGHTS)
        return images
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()  # 释放显存后以更小的 tile 重试
        smaller = max(64, tile_size // 2)
        images, _ = infer_modalities(img, smaller, str(config.MODEL_DIR),
                                     opt=opt, seg_weights=config.SEG_WEIGHTS)
        return images
