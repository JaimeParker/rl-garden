"""Image extractors apply ``/255`` only to RGB keys; depth reaches the backbone
unscaled, whatever its declared range (each env backend owns depth units)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders import CombinedExtractor, ViTTokenAndPropExtractor
from rl_garden.encoders.base import image_needs_normalization
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObservationSchema


def _obs_space(hw: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb_a": spaces.Box(0, 255, (hw, hw, 3), np.uint8),
            "rgb_f": spaces.Box(0.0, 255.0, (hw, hw, 3), np.float32),
            "depth_a": spaces.Box(-np.inf, np.inf, (hw, hw, 1), np.float32),
            "depth_b": spaces.Box(0.0, 1.0, (hw, hw, 1), np.float32),
            "depth_c": spaces.Box(0.0, 255.0, (hw, hw, 1), np.float32),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )


def _combined(hw: int) -> CombinedExtractor:
    obs_space = _obs_space(hw)
    return CombinedExtractor(
        obs_space,
        ObservationSchema.from_space(obs_space),
        EncoderConfig(image_fusion_mode="per_key", features_dim=16),
    )


def _vit(hw: int) -> ViTTokenAndPropExtractor:
    obs_space = _obs_space(hw)
    return ViTTokenAndPropExtractor(
        obs_space,
        ObservationSchema.from_space(obs_space),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )


# (builder, image side, converts HWC -> NCHW inside _prepare_image)
_EXTRACTORS = {
    "combined": (_combined, 64, False),
    "vit": (_vit, 32, True),
}


@pytest.mark.parametrize("name", sorted(_EXTRACTORS))
def test_only_rgb_keys_are_marked_for_normalization(name):
    build, hw, _ = _EXTRACTORS[name]
    extractor = build(hw)
    assert extractor._needs_norm == frozenset({"rgb_a", "rgb_f"})


@pytest.mark.parametrize("name", sorted(_EXTRACTORS))
@pytest.mark.parametrize("key", ["depth_a", "depth_b", "depth_c"])
def test_depth_reaches_backbone_unscaled(name, key):
    build, hw, to_nchw = _EXTRACTORS[name]
    extractor = build(hw)
    depth = torch.full((2, hw, hw, 1), 2.5)
    depth[:, 0, 0, 0] = 0.25

    prepared = extractor._prepare_image(key, depth)

    expected = depth.permute(0, 3, 1, 2) if to_nchw else depth
    assert prepared.dtype == torch.float32
    assert torch.equal(prepared, expected)


@pytest.mark.parametrize("name", sorted(_EXTRACTORS))
@pytest.mark.parametrize(
    "key,dtype", [("rgb_a", torch.uint8), ("rgb_f", torch.float32)]
)
def test_rgb_is_divided_by_255(name, key, dtype):
    build, hw, to_nchw = _EXTRACTORS[name]
    extractor = build(hw)
    rgb = torch.full((2, hw, hw, 3), 255, dtype=dtype)
    rgb[:, 0, 0, 0] = 51

    prepared = extractor._prepare_image(key, rgb)

    expected = rgb.float() / 255.0
    if to_nchw:
        expected = expected.permute(0, 3, 1, 2)
    assert torch.allclose(prepared, expected)
    assert float(prepared.max()) == pytest.approx(1.0)


def test_image_needs_normalization_rgb_rule_unchanged():
    assert image_needs_normalization(spaces.Box(0, 255, (4, 4, 3), np.uint8))
    assert image_needs_normalization(spaces.Box(0.0, 255.0, (4, 4, 3), np.float32))
    assert not image_needs_normalization(spaces.Box(0.0, 1.0, (4, 4, 3), np.float32))
