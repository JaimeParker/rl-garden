"""Image augmentations used during training.

Port of the pieces we need from
``3rd_party/hil-serl/.../vision/data_augmentations.py``. Kept small and
torch-native so augmentations can run in-place on GPU tensors pulled from a
replay buffer sample — no PIL / torchvision roundtrips.

Inputs are NCHW float tensors in [0, 1]. The augmentation modules are
``nn.Module`` so they can live inside an encoder stack and respond to
``.train()`` / ``.eval()``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomCrop(nn.Module):
    """Reflect-pad by ``padding`` then crop back to the original (H, W).

    Mimics ``random_crop`` in hil-serl augmentations: each image in the batch
    gets an independent offset. No-op when ``self.training`` is False.
    """

    def __init__(self, padding: int = 4) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.padding <= 0:
            return x
        b, _c, h, w = x.shape
        padded = F.pad(x, [self.padding] * 4, mode="reflect")
        offs_h = torch.randint(
            0, 2 * self.padding + 1, (b,), device=x.device
        )
        offs_w = torch.randint(
            0, 2 * self.padding + 1, (b,), device=x.device
        )
        out = torch.empty_like(x)
        for i in range(b):
            out[i] = padded[i, :, offs_h[i] : offs_h[i] + h, offs_w[i] : offs_w[i] + w]
        return out
