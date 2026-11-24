from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.encoders import CombinedExtractor, PlainConv
from rl_garden.encoders.combined import default_image_encoder_factory
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObservationSchema

POOLING_MODES = ("flatten", "gap", "adaptive_max")
FEATURE_MAP_SIZES = {
    (64, 64): (4, 4),
    (79, 79): (4, 4),
    (84, 84): (5, 5),
    (128, 128): (4, 4),
    (224, 224): (14, 14),
    (120, 160): (7, 10),
    (32, 128): (2, 8),
}


def _image_space(height: int, width: int) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32)


def _projection(encoder: PlainConv) -> nn.Linear:
    return next(module for module in encoder.fc.modules() if isinstance(module, nn.Linear))


@pytest.mark.parametrize("pooling", POOLING_MODES)
@pytest.mark.parametrize("image_hw", FEATURE_MAP_SIZES, ids=lambda hw: f"{hw[0]}x{hw[1]}")
def test_forward_backward_at_image_size(image_hw: tuple[int, int], pooling: str) -> None:
    height, width = image_hw
    map_height, map_width = FEATURE_MAP_SIZES[image_hw]
    encoder = PlainConv(_image_space(height, width), features_dim=16, pooling=pooling)
    image = torch.rand(2, 3, height, width)

    features = encoder(image)
    features.sum().backward()

    assert features.shape == (2, 16)
    assert torch.isfinite(features).all()
    assert encoder.cnn(image).shape[-2:] == (map_height, map_width)
    assert _projection(encoder).in_features == (
        64 * map_height * map_width if pooling == "flatten" else 64
    )
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("side", [64, 128])
def test_flatten_checkpoint_layout_is_unchanged(side: int) -> None:
    encoder = PlainConv(_image_space(side, side), pooling="flatten")

    assert _projection(encoder).in_features == 1024
    assert list(encoder.state_dict()) == [
        "cnn.0.weight",
        "cnn.0.bias",
        "cnn.3.weight",
        "cnn.3.bias",
        "cnn.6.weight",
        "cnn.6.bias",
        "cnn.9.weight",
        "cnn.9.bias",
        "cnn.12.weight",
        "cnn.12.bias",
        "fc.0.0.weight",
        "fc.0.0.bias",
    ]


@pytest.mark.parametrize(
    ("height", "width", "first_pool"),
    [(128, 128, 4), (64, 64, 2), (84, 84, 2), (128, 64, 2), (224, 224, 2)],
)
def test_only_128x128_uses_the_wider_first_pool(height: int, width: int, first_pool: int) -> None:
    encoder = PlainConv(_image_space(height, width))

    assert encoder.cnn[2].kernel_size == first_pool
    assert encoder.cnn[2].stride == first_pool


@pytest.mark.parametrize("pooling", POOLING_MODES)
@pytest.mark.parametrize(("height", "width"), [(8, 8), (15, 64), (64, 15)])
def test_image_below_minimum_side_raises(height: int, width: int, pooling: str) -> None:
    with pytest.raises(ValueError, match=rf"at least 16 px per side, got {height}x{width}"):
        PlainConv(_image_space(height, width), pooling=pooling)


@pytest.mark.parametrize("pooling", POOLING_MODES)
def test_minimum_side_is_accepted(pooling: str) -> None:
    encoder = PlainConv(_image_space(16, 16), features_dim=16, pooling=pooling)

    assert encoder(torch.rand(2, 3, 16, 16)).shape == (2, 16)


@pytest.mark.parametrize("side", [84, 224])
def test_default_factory_forward(side: int) -> None:
    encoder = default_image_encoder_factory(features_dim=16)(_image_space(side, side))

    assert encoder(torch.rand(2, 3, side, side)).shape == (2, 16)


@pytest.mark.parametrize("fusion_mode", ["stack_channels", "per_key"])
@pytest.mark.parametrize("side", [84, 224])
def test_combined_extractor_forward(side: int, fusion_mode: str) -> None:
    obs_space = spaces.Dict(
        {
            "rgb_front": spaces.Box(0, 255, (side, side, 3), dtype=np.uint8),
            "depth_front": spaces.Box(0.0, 1.0, (side, side, 1), dtype=np.float32),
            "state": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        }
    )
    extractor = CombinedExtractor(
        obs_space,
        ObservationSchema.from_space(obs_space),
        EncoderConfig(features_dim=16, image_fusion_mode=fusion_mode, proprio_latent_dim=8),
    )
    obs = {
        "rgb_front": torch.randint(0, 256, (2, side, side, 3), dtype=torch.uint8),
        "depth_front": torch.rand(2, side, side, 1),
        "state": torch.rand(2, 4),
    }

    num_image_encoders = 1 if fusion_mode == "stack_channels" else 2
    assert extractor(obs).shape == (2, 16 * num_image_encoders + 8)
