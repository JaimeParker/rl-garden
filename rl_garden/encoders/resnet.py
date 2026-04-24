"""ResNetV1 image encoder — PyTorch port of hil-serl's Flax ResNet.

Source: ``3rd_party/hil-serl/serl_launcher/serl_launcher/vision/resnet_v1.py``.

Parity notes vs the Flax version:
  * GroupNorm(num_groups=4) replaces ``MyGroupNorm`` (Flax L119).
  * Kaiming-normal init on convs (mode=fan_out, relu), zeros on biases — Flax
    uses ``nn.initializers.kaiming_normal`` for convs (L235).
  * ImageNet mean/std normalization runs inside ``forward`` (Flax L224-L226),
    so the encoder accepts NCHW uint8-equivalent float tensors in [0, 1]
    (CombinedExtractor already /255's them).
  * The 7x7 stem + maxpool stack reproduces Flax L252-L262.
  * No pre-trained weights are loaded; ``resnet10``/``resnet18`` map to the
    same ``stage_sizes`` as ``resnetv1_configs`` in the Flax file.
  * Pooling head defaults to ``SpatialLearnedEmbeddings`` + a bottleneck
    projection (Linear -> LayerNorm -> tanh) matching L319-L322.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.pooling import AvgPool, SpatialLearnedEmbeddings, SpatialSoftmax

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PoolingMethod = Literal["spatial_learned_embeddings", "spatial_softmax", "avg"]


def _make_norm(channels: int) -> nn.Module:
    # num_groups=4 matches MyGroupNorm in hil-serl.
    return nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-5)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = _make_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _make_norm(out_channels)

        if in_channels != out_channels or stride != 1:
            self.proj = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=False
            )
            self.proj_norm = _make_norm(out_channels)
        else:
            self.proj = None
            self.proj_norm = None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.proj is not None:
            residual = self.proj_norm(self.proj(residual))
        return self.act(residual + y)


class ResNetEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        stage_sizes: Sequence[int] = (1, 1, 1, 1),
        num_filters: int = 64,
        pooling_method: PoolingMethod = "spatial_softmax",
        num_spatial_blocks: int = 8,
        bottleneck_dim: int = 256,
        pretrained_weights: Optional[str] = None,
    ) -> None:
        # observation_space is channels-first (C, H, W) — CombinedExtractor
        # produces this before calling us.
        assert len(observation_space.shape) == 3, observation_space.shape
        c, h, w = observation_space.shape

        # Compute spatial dim after stem+stages to size the pooling head.
        feat_h, feat_w = h, w
        # stem: stride-2 conv -> stride-2 maxpool
        feat_h = (feat_h + 1) // 2
        feat_w = (feat_w + 1) // 2
        feat_h = (feat_h + 1) // 2
        feat_w = (feat_w + 1) // 2
        out_channels = num_filters
        # stages: every stage after the first halves spatial dims.
        for i, _blocks in enumerate(stage_sizes):
            if i > 0:
                feat_h = (feat_h + 1) // 2
                feat_w = (feat_w + 1) // 2
                out_channels = num_filters * (2**i)
            else:
                out_channels = num_filters

        # Features dim is determined by the pooling head.
        if pooling_method == "spatial_learned_embeddings":
            pooled_dim = out_channels * num_spatial_blocks
        elif pooling_method == "spatial_softmax":
            pooled_dim = 2 * out_channels
        elif pooling_method == "avg":
            pooled_dim = out_channels
        else:
            raise ValueError(f"unknown pooling method {pooling_method!r}")

        features_dim = bottleneck_dim if bottleneck_dim > 0 else pooled_dim
        super().__init__(observation_space, features_dim)

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        self._in_channels = c

        # --- stem ---
        self.stem_conv = nn.Conv2d(c, num_filters, 7, stride=2, padding=3, bias=False)
        self.stem_norm = _make_norm(num_filters)
        self.stem_act = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # --- stages ---
        blocks: list[nn.Module] = []
        in_c = num_filters
        for i, block_size in enumerate(stage_sizes):
            stage_filters = num_filters * (2**i)
            for j in range(block_size):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(ResNetBlock(in_c, stage_filters, stride=stride))
                in_c = stage_filters
        self.blocks = nn.Sequential(*blocks)

        # --- pooling head ---
        if pooling_method == "spatial_learned_embeddings":
            self.pool: nn.Module = SpatialLearnedEmbeddings(
                channels=out_channels,
                height=max(feat_h, 1),
                width=max(feat_w, 1),
                num_features=num_spatial_blocks,
            )
            self.post_pool_drop = nn.Dropout(0.1)
        elif pooling_method == "spatial_softmax":
            self.pool = SpatialSoftmax(
                channels=out_channels, height=max(feat_h, 1), width=max(feat_w, 1)
            )
            self.post_pool_drop = nn.Identity()
        else:
            self.pool = AvgPool()
            self.post_pool_drop = nn.Identity()

        if bottleneck_dim > 0:
            self.bottleneck = nn.Sequential(
                nn.Linear(pooled_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.Tanh(),
            )
        else:
            self.bottleneck = nn.Identity()

        self._init_weights()

        if pretrained_weights is not None:
            self.load_pretrained(pretrained_weights)

    # --- pretrained weight loading ---

    @staticmethod
    def pretrained_dir() -> Path:
        """Directory searched for pretrained ResNet checkpoints.

        Resolution order:
          1. ``$RL_GARDEN_PRETRAINED_DIR`` (if set)
          2. ``pretrained_models/`` at the repo root.
        """
        env = os.environ.get("RL_GARDEN_PRETRAINED_DIR")
        if env:
            return Path(env).expanduser().resolve()
        # rl_garden/encoders/resnet.py -> repo root is 3 parents up.
        return Path(__file__).resolve().parents[2] / "pretrained_models"

    def load_pretrained(self, name: str, strict: bool = False) -> list[str]:
        """Look up ``<name>.pt`` under :meth:`pretrained_dir` and load it.

        Returns the list of missing/unexpected keys (concatenated) so the
        caller can sanity-check that only the expected heads (e.g.
        bottleneck / pooling) were left un-initialized.
        """
        path = self.pretrained_dir() / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained weights {name!r} not found at {path}. "
                f"Set $RL_GARDEN_PRETRAINED_DIR or drop the .pt file in "
                f"{self.pretrained_dir()}."
            )
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.load_state_dict(state, strict=strict)
        return list(missing) + list(unexpected)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: NCHW in [0, 1]. Handle 1-channel (depth) by broadcasting the norm.
        if x.shape[1] == 3:
            x = (x - self.mean) / self.std
        else:
            # Non-RGB channels (e.g. depth-only or rgb+depth concat).
            # Only rescale the first 3 if present; leave extras alone.
            if x.shape[1] > 3:
                rgb = (x[:, :3] - self.mean) / self.std
                x = torch.cat([rgb, x[:, 3:]], dim=1)
        x = self.stem_pool(self.stem_act(self.stem_norm(self.stem_conv(x))))
        x = self.blocks(x)
        x = self.pool(x)
        if x.ndim == 4:
            x = x.flatten(1)
        x = self.post_pool_drop(x)
        return self.bottleneck(x)


# --- config factories, aligned with hil-serl ``resnetv1_configs`` ---

_RESNET_STAGES = {
    "resnet10": (1, 1, 1, 1),
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}


def resnet_encoder_factory(
    name: str = "resnet10",
    features_dim: int = 256,
    num_spatial_blocks: int = 8,
    pooling_method: PoolingMethod = "spatial_softmax",
    num_filters: int = 64,
    pretrained_weights: Optional[str] = None,
):
    """Return an image-encoder factory suitable for ``CombinedExtractor``.

    ``pretrained_weights`` (optional): checkpoint name looked up via
    :py:meth:`ResNetEncoder.pretrained_dir`. E.g. ``"resnet10-imagenet"``
    loads ``pretrained_models/resnet10-imagenet.pt``.
    """
    if name not in _RESNET_STAGES:
        raise ValueError(f"unknown resnet config {name!r}; one of {list(_RESNET_STAGES)}")
    stages = _RESNET_STAGES[name]

    def _factory(img_space: spaces.Box) -> ResNetEncoder:
        return ResNetEncoder(
            img_space,
            stage_sizes=stages,
            num_filters=num_filters,
            pooling_method=pooling_method,
            num_spatial_blocks=num_spatial_blocks,
            bottleneck_dim=features_dim,
            pretrained_weights=pretrained_weights,
        )

    return _factory
