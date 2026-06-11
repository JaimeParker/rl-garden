"""PlainConv image encoder used by ManiSkill's SAC-RGBD baseline.

Lifted from ``3rd_party/ManiSkill/examples/baselines/sac/sac_rgbd.py`` and
repackaged as a ``BaseFeaturesExtractor`` so it's pluggable into any policy.

Input: NCHW float tensor (already divided by 255 and channels-first).
Expected image sizes: 64x64 or 128x128.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import create_mlp


class PlainConv(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        image_size: tuple[int, int] = (128, 128),
        pool_feature_map: bool = False,
        pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten",
        last_act: bool = True,
        weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform",
    ) -> None:
        super().__init__(observation_space, features_dim)
        if weight_init not in ("kaiming_uniform", "orthogonal"):
            raise ValueError(
                "weight_init must be either 'kaiming_uniform' or 'orthogonal', "
                f"got {weight_init!r}"
            )
        if pooling not in ("flatten", "gap", "adaptive_max"):
            raise ValueError(
                "pooling must be 'flatten', 'gap', or 'adaptive_max', "
                f"got {pooling!r}"
            )
        if pool_feature_map:
            pooling = "adaptive_max"
        self.weight_init = weight_init
        self.pooling = pooling

        # observation_space is the channels-first image space (C, H, W).
        in_channels = int(observation_space.shape[0])
        first_pool = 4 if (image_size[0] == 128 and image_size[1] == 128) else 2

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(first_pool, first_pool),  # 128->32 or 64->32
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pooling == "gap":
            self.pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
            fc_input_dim = 64
        elif pooling == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            fc_input_dim = 64
        else:
            self.pool = nn.Identity()
            fc_input_dim = 64 * 4 * 4

        fc = create_mlp(
            input_dim=fc_input_dim,
            output_dim=features_dim,
            net_arch=[],
            squash_output=False,
        )
        self.fc = nn.Sequential(fc, nn.ReLU()) if last_act else fc

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if self.weight_init == "orthogonal":
                    gain = (
                        nn.init.calculate_gain("relu")
                        if isinstance(m, (nn.Conv1d, nn.Conv2d))
                        else 1.0
                    )
                    nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
