"""Image transforms and defaults for classifier datasets."""
from __future__ import annotations

from typing import Tuple

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: Tuple[int, int], normalize: bool) -> transforms.Compose:
    resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
    ops = [transforms.ToTensor(), resize]
    if normalize:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def empty_image(image_size: Tuple[int, int]) -> torch.Tensor:
    return torch.zeros(3, image_size[0], image_size[1], dtype=torch.float32)
