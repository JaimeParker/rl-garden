"""Combined extractor for Dict observations ({rgb[/depth], state, ...}).

Design
------
For each observation key we plug in one of:
  - image encoder (``PlainConv`` or a ResNet) when the key is an image,
  - proprio branch (Dense -> LayerNorm -> tanh) when the key is ``state``,
  - plain flatten for any other vector key.

Image keys are combined by channel-concatenation BEFORE the encoder (matches
``EncoderObsWrapper`` in ManiSkill's sac_rgbd.py), so a single encoder sees
all image modalities stacked along channel dim. The proprio branch follows
hil-serl's ``EncodingWrapper`` design (common/encoding.py L65-L69).

Inputs from ManiSkill's ``FlattenRGBDObservationWrapper`` are HWC uint8
tensors for images; we permute to NCHW and normalize to [0,1] here.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.plain_conv import PlainConv

# A factory takes the stacked image ``spaces.Box`` (channels-first) and returns
# a ``BaseFeaturesExtractor``. This lets the caller swap PlainConv for a ResNet
# without changing the CombinedExtractor.
ImageEncoderFactory = Callable[[spaces.Box], BaseFeaturesExtractor]


def default_image_encoder_factory(features_dim: int = 256) -> ImageEncoderFactory:
    def _factory(img_space: spaces.Box) -> BaseFeaturesExtractor:
        _, h, w = img_space.shape
        return PlainConv(img_space, features_dim=features_dim, image_size=(h, w))

    return _factory


class ProprioEncoder(BaseFeaturesExtractor):
    """Proprio branch: Linear -> LayerNorm -> tanh. Port of hil-serl's block."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        image_keys: Iterable[str] = ("rgb", "depth"),
        state_key: str = "state",
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        proprio_latent_dim: int = 64,
        use_proprio: bool = True,
    ) -> None:
        assert isinstance(observation_space, spaces.Dict)

        # Determine which image keys are actually present.
        present_image_keys = [k for k in image_keys if k in observation_space.spaces]
        has_state = use_proprio and state_key in observation_space.spaces

        # Build the stacked image space by channel-concat.
        # ManiSkill's FlattenRGBDObservationWrapper emits HWC uint8; we treat
        # them as channels-last and stack along the last dim, then transpose.
        total_channels = 0
        image_hw: Optional[tuple[int, int]] = None
        for k in present_image_keys:
            sp = observation_space.spaces[k]
            assert isinstance(sp, spaces.Box) and len(sp.shape) == 3, (
                f"image key {k!r} must be a 3D Box (H, W, C); got {sp.shape}"
            )
            h, w, c = sp.shape
            total_channels += c
            image_hw = (h, w) if image_hw is None else image_hw
            assert image_hw == (h, w), "all image keys must share the same H, W"

        features_dim = 0
        self._has_images = total_channels > 0
        image_encoder: Optional[BaseFeaturesExtractor] = None
        if self._has_images:
            assert image_hw is not None
            stacked_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(total_channels, image_hw[0], image_hw[1]),
                dtype=np.float32,
            )
            factory = image_encoder_factory or default_image_encoder_factory()
            image_encoder = factory(stacked_space)
            features_dim += image_encoder.features_dim

        proprio: Optional[ProprioEncoder] = None
        if has_state:
            proprio = ProprioEncoder(
                observation_space.spaces[state_key], features_dim=proprio_latent_dim
            )
            features_dim += proprio.features_dim

        assert features_dim > 0, "CombinedExtractor produced 0-dim output."
        super().__init__(observation_space, features_dim)

        self.image_keys: tuple[str, ...] = tuple(present_image_keys)
        self.state_key = state_key
        self.has_state = has_state
        self.image_encoder = image_encoder
        self.proprio = proprio

    def _stack_images(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = []
        for k in self.image_keys:
            x = obs[k]
            if k == "rgb":
                # uint8 HWC -> float32, normalized to [0, 1]
                x = x.float() / 255.0
            else:
                x = x.float()
            tensors.append(x)
        x = torch.cat(tensors, dim=-1)  # (B, H, W, Ctotal)
        x = x.permute(0, 3, 1, 2).contiguous()  # NCHW
        return x

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = []
        if self._has_images:
            assert self.image_encoder is not None
            img = self._stack_images(obs)
            out.append(self.image_encoder(img))
        if self.has_state:
            assert self.proprio is not None
            out.append(self.proprio(obs[self.state_key]))
        return torch.cat(out, dim=-1)
