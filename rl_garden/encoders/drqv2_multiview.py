"""Controlled multi-view variants of the DrQ-v2 convolutional encoder.

All encoders accept the same 9-channel ``head + left wrist + right wrist``
tensor as B9B and return the same flattened ``32 * H_out * W_out`` feature
shape.  This keeps the replay layout and SAC actor/critic input dimensions
unchanged while moving only the point at which the camera views interact.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor, image_needs_normalization
from rl_garden.encoders.drqv2_conv import _conv_output_size

ImageEncoderFactory = Callable[..., BaseFeaturesExtractor]

_NUM_VIEWS = 3
_CHANNELS_PER_VIEW = 3
_TOTAL_CHANNELS = _NUM_VIEWS * _CHANNELS_PER_VIEW
_INDEPENDENT_VIEW_CHANNELS = 18


def _representation_dim(height: int, width: int) -> int:
    for stride in (2, 1, 1, 1):
        height = _conv_output_size(height, 3, stride)
        width = _conv_output_size(width, 3, stride)
    return 32 * height * width


def _validate_space(observation_space: spaces.Box) -> tuple[int, int]:
    if len(observation_space.shape) != 3:
        raise ValueError(
            "multi-view DrQ-v2 encoders require a CHW image space, "
            f"got {observation_space.shape!r}"
        )
    channels, height, width = map(int, observation_space.shape)
    if channels != _TOTAL_CHANNELS:
        raise ValueError(
            "multi-view DrQ-v2 encoders require exactly three RGB views "
            f"stacked as 9 channels, got {channels} channels"
        )
    return height, width


def _reset_conv_parameters(module: nn.Module) -> None:
    gain = nn.init.calculate_gain("relu")
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.orthogonal_(layer.weight.data, gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class _ThreeViewDrQv2Encoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box) -> None:
        height, width = _validate_space(observation_space)
        super().__init__(
            observation_space,
            features_dim=_representation_dim(height, width),
        )
        self._needs_norm = image_needs_normalization(observation_space)

    def _normalize_and_split(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._needs_norm:
            obs = obs.float() / 255.0 - 0.5
        else:
            obs = obs.float() - 0.5
        return torch.split(obs, _CHANNELS_PER_VIEW, dim=1)


class MultiViewStemDrQv2Encoder(_ThreeViewDrQv2Encoder):
    """Camera-specific small stems followed by one shared spatial trunk.

    Each RGB view gets its own ``3 -> 12`` stride-2 convolution.  The three
    resulting 12-channel maps are concatenated and processed by the same three
    convolutional layers used as B9B's later trunk.  At 64x64 the output is
    still ``32x25x25 = 20,000`` values.

    The 12-channel choice keeps trainable convolution parameters within about
    2% of the original 9-channel B9B encoder, avoiding a hidden capacity jump.
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__(observation_space)
        self.view_stems = nn.ModuleList(
            [
                nn.Conv2d(_CHANNELS_PER_VIEW, 12, 3, stride=2)
                for _ in range(_NUM_VIEWS)
            ]
        )
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(36, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        _reset_conv_parameters(self)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        views = self._normalize_and_split(obs)
        stemmed = [torch.relu(stem(view)) for stem, view in zip(self.view_stems, views)]
        spatial = self.shared_trunk(torch.cat(stemmed, dim=1))
        return spatial.reshape(spatial.shape[0], -1)


class SpatialLateFusionDrQv2Encoder(_ThreeViewDrQv2Encoder):
    """Keep per-view spatial maps until a trainable late fusion layer.

    One shared DrQ-v2 convolutional trunk processes each RGB view separately.
    Its three ``32x25x25`` maps are concatenated as ``96x25x25`` and fused by
    a trainable 1x1 convolution before flattening.  Thus fusion can mix view and
    feature channels independently at every spatial cell while downstream SAC
    still receives the same 20,000-dimensional image feature as B9B.
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__(observation_space)
        self.shared_convnet = nn.Sequential(
            nn.Conv2d(_CHANNELS_PER_VIEW, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.spatial_fusion = nn.Conv2d(_NUM_VIEWS * 32, 32, 1)
        _reset_conv_parameters(self.shared_convnet)
        self._reset_fusion_as_view_mean()

    def _reset_fusion_as_view_mean(self) -> None:
        # Initially average corresponding feature channels across cameras.
        # Training remains free to learn arbitrary cross-view/channel mixing.
        with torch.no_grad():
            self.spatial_fusion.weight.zero_()
            for view_index in range(_NUM_VIEWS):
                offset = view_index * 32
                for channel in range(32):
                    self.spatial_fusion.weight[channel, offset + channel, 0, 0] = (
                        1.0 / _NUM_VIEWS
                    )
            if self.spatial_fusion.bias is not None:
                self.spatial_fusion.bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        per_view_maps = [
            self.shared_convnet(view) for view in self._normalize_and_split(obs)
        ]
        spatial = torch.relu(self.spatial_fusion(torch.cat(per_view_maps, dim=1)))
        return spatial.reshape(spatial.shape[0], -1)


class IndependentLateFusionDrQv2Encoder(_ThreeViewDrQv2Encoder):
    """Camera-specific compact trunks followed by spatial late fusion.

    Unlike :class:`MultiViewStemDrQv2Encoder`, sharing does not begin after the
    first convolution. Unlike :class:`SpatialLateFusionDrQv2Encoder`, the
    three cameras do not have to express head- and wrist-camera observations
    through one shared set of convolution weights. Each RGB view instead gets
    an independent four-layer, 18-channel DrQ trunk. The three
    ``18x25x25`` maps are concatenated and fused to ``32x25x25`` by a 1x1
    convolution before flattening.

    Eighteen channels keep the total convolution parameter count within about
    3% of B9B's 9-to-32 DrQ encoder. The comparison therefore changes where
    camera parameters are shared and where views interact without hiding a
    large capacity increase.
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__(observation_space)
        self.view_convnets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        _CHANNELS_PER_VIEW,
                        _INDEPENDENT_VIEW_CHANNELS,
                        3,
                        stride=2,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        _INDEPENDENT_VIEW_CHANNELS,
                        _INDEPENDENT_VIEW_CHANNELS,
                        3,
                        stride=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        _INDEPENDENT_VIEW_CHANNELS,
                        _INDEPENDENT_VIEW_CHANNELS,
                        3,
                        stride=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        _INDEPENDENT_VIEW_CHANNELS,
                        _INDEPENDENT_VIEW_CHANNELS,
                        3,
                        stride=1,
                    ),
                    nn.ReLU(),
                )
                for _ in range(_NUM_VIEWS)
            ]
        )
        self.spatial_fusion = nn.Conv2d(
            _NUM_VIEWS * _INDEPENDENT_VIEW_CHANNELS,
            32,
            1,
        )
        _reset_conv_parameters(self)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        per_view_maps = [
            convnet(view)
            for convnet, view in zip(
                self.view_convnets,
                self._normalize_and_split(obs),
            )
        ]
        spatial = torch.relu(self.spatial_fusion(torch.cat(per_view_maps, dim=1)))
        return spatial.reshape(spatial.shape[0], -1)


def drq_v2_multiview_stem_encoder_factory() -> ImageEncoderFactory:
    def _factory(img_space: spaces.Box) -> MultiViewStemDrQv2Encoder:
        return MultiViewStemDrQv2Encoder(img_space)

    return _factory


def drq_v2_spatial_late_fusion_encoder_factory() -> ImageEncoderFactory:
    def _factory(img_space: spaces.Box) -> SpatialLateFusionDrQv2Encoder:
        return SpatialLateFusionDrQv2Encoder(img_space)

    return _factory


def drq_v2_independent_late_fusion_encoder_factory() -> ImageEncoderFactory:
    def _factory(img_space: spaces.Box) -> IndependentLateFusionDrQv2Encoder:
        return IndependentLateFusionDrQv2Encoder(img_space)

    return _factory
