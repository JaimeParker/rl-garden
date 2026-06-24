"""DrQ-v2 style CNN encoder.

Ported from ``3rd_party/drqv2/drqv2.py:48-67``.  Four convolutional layers
with no pooling, no fully-connected head — just flatten at the end.  The
motivation is to preserve spatial information up to the critic trunk, which
performs its own learned feature compression via ``Linear → LayerNorm → Tanh``.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor, image_needs_normalization

# Deferred import to avoid circular dependency at module level.
ImageEncoderFactory = Callable[..., BaseFeaturesExtractor]


def _conv_output_size(h: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (h - kernel + 2 * padding) // stride + 1


class DrQv2Encoder(BaseFeaturesExtractor):
    """DrQ-v2 convolutional encoder for image observations.

    Input: ``(B, C, H, W)`` float tensor **in [0, 255]** (uint8 range).
    The encoder internally normalizes to zero-center ``x / 255 - 0.5``, matching
    the original DrQ-v2 pipeline.

    Architecture::

        Conv2d(C → 32, 3, stride=2)  →  ReLU
        Conv2d(32 → 32, 3, stride=1) →  ReLU
        Conv2d(32 → 32, 3, stride=1) →  ReLU
        Conv2d(32 → 32, 3, stride=1) →  ReLU
        → flatten → (B, 32·H_out·W_out)

    ``features_dim`` is computed from the input spatial dimensions so it
    adapts to different resolutions (84×84 → 39200, 64×64 → 20000, …).

    Weights are initialised with orthogonal gain matching the original
    ``utils.weight_init``.
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        in_channels = int(observation_space.shape[0])
        h, w = int(observation_space.shape[1]), int(observation_space.shape[2])

        # Compute output spatial dims layer-by-layer so features_dim is exact.
        h1 = _conv_output_size(h, 3, 2)  # Conv1
        h2 = _conv_output_size(h1, 3, 1)  # Conv2
        h3 = _conv_output_size(h2, 3, 1)  # Conv3
        h4 = _conv_output_size(h3, 3, 1)  # Conv4
        w1 = _conv_output_size(w, 3, 2)  # Conv1
        w2 = _conv_output_size(w1, 3, 1)  # Conv2
        w3 = _conv_output_size(w2, 3, 1)  # Conv3
        w4 = _conv_output_size(w3, 3, 1)  # Conv4
        repr_dim = 32 * h4 * w4

        super().__init__(observation_space, repr_dim)

        self._needs_norm = image_needs_normalization(observation_space)

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(m.weight.data, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs comes in as uint8 [0,255] from replay buffer; normalize to [-0.5, 0.5]
        if self._needs_norm:
            obs = obs.float() / 255.0 - 0.5
        else:
            obs = obs.float() - 0.5
        h = self.convnet(obs)
        return h.reshape(h.shape[0], -1)


def drq_v2_encoder_factory() -> ImageEncoderFactory:
    """Return a factory that creates ``DrQv2Encoder`` instances.

    Usable as ``image_encoder_factory`` in ``CombinedExtractor`` /
    ``DDPG`` so the entire image pathway uses the DrQ-v2 architecture
    with a single-liner:

        DDPG(..., image_encoder_factory=drq_v2_encoder_factory(), ...)
    """

    def _factory(img_space: spaces.Box) -> DrQv2Encoder:
        return DrQv2Encoder(img_space)

    return _factory
