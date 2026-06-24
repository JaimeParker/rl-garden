from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.cli_args import VisionArgs, image_encoder_factory_from_args
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders import CombinedExtractor, PlainConv, RandomShiftsAug


def _image_space(channels: int = 3) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(channels, 64, 64), dtype=np.float32)


class RecordingImageEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__(observation_space, features_dim=1)
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return x.flatten(1).mean(dim=1, keepdim=True)


def test_plain_conv_default_init_does_not_reinitialize_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _forbidden_orthogonal(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("default PlainConv init must not call orthogonal_")

    monkeypatch.setattr(nn.init, "orthogonal_", _forbidden_orthogonal)
    enc = PlainConv(_image_space(), weight_init="kaiming_uniform")

    assert calls == 0
    for module in enc.modules():
        if (
            isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))
            and module.bias is not None
        ):
            assert torch.count_nonzero(module.bias) == 0


def test_plain_conv_orthogonal_init_zeros_biases() -> None:
    enc = PlainConv(_image_space(), features_dim=16, weight_init="orthogonal")

    first_conv = next(
        module for module in enc.modules() if isinstance(module, nn.Conv2d)
    )
    rows = first_conv.weight.detach().flatten(1)
    expected = torch.eye(rows.shape[0]) * nn.init.calculate_gain("relu") ** 2
    assert torch.allclose(rows @ rows.T, expected, atol=1e-5, rtol=1e-5)

    first_linear = next(
        module for module in enc.modules() if isinstance(module, nn.Linear)
    )
    linear_rows = first_linear.weight.detach()
    assert torch.allclose(
        linear_rows @ linear_rows.T,
        torch.eye(linear_rows.shape[0]),
        atol=1e-5,
        rtol=1e-5,
    )

    for module in enc.modules():
        if (
            isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))
            and module.bias is not None
        ):
            assert torch.count_nonzero(module.bias) == 0


def test_plain_conv_last_act_false_removes_fc_relu() -> None:
    enc = PlainConv(_image_space(), last_act=False)

    assert not any(isinstance(module, nn.ReLU) for module in enc.fc.modules())


@pytest.mark.parametrize("pooling", ["gap", "adaptive_max"])
def test_plain_conv_pooled_modes_forward(pooling: str) -> None:
    enc = PlainConv(_image_space(), features_dim=16, pooling=pooling)
    out = enc(torch.zeros(2, 3, 64, 64))

    assert out.shape == (2, 16)
    first_linear = next(
        module for module in enc.modules() if isinstance(module, nn.Linear)
    )
    assert first_linear.in_features == 64


def test_plain_conv_gap_is_spatial_mean_before_projection() -> None:
    enc = PlainConv(_image_space(), features_dim=16, pooling="gap")
    image = torch.randn(2, 3, 64, 64)

    feature_map = enc.cnn(image)
    pooled = enc.pool(feature_map).flatten(1)

    assert torch.allclose(pooled, feature_map.mean(dim=(-2, -1)))


def test_plain_conv_gap_reduces_bottleneck_parameters() -> None:
    flatten = PlainConv(_image_space(), pooling="flatten")
    gap = PlainConv(_image_space(), pooling="gap")
    flatten_fc = next(
        module for module in flatten.fc.modules() if isinstance(module, nn.Linear)
    )
    gap_fc = next(
        module for module in gap.fc.modules() if isinstance(module, nn.Linear)
    )

    assert flatten_fc.in_features == 64 * 4 * 4
    assert gap_fc.in_features == 64
    assert sum(p.numel() for p in flatten.fc.parameters()) == 262_400
    assert sum(p.numel() for p in gap.fc.parameters()) == 16_640


def test_plain_conv_gap_applies_to_each_per_key_camera() -> None:
    obs_space = spaces.Dict(
        {
            "rgb_base_camera": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "rgb_hand_camera": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        }
    )
    factory = image_encoder_factory_from_args(
        VisionArgs(
            encoder="plain_conv",
            encoder_features_dim=16,
            image_fusion_mode="per_key",
            plain_conv_pooling="gap",
        )
    )
    extractor = CombinedExtractor(
        obs_space,
        image_keys=("rgb_base_camera", "rgb_hand_camera"),
        image_encoder_factory=factory,
        fusion_mode="per_key",
    )

    assert set(extractor.image_encoders) == {"rgb_base_camera", "rgb_hand_camera"}
    assert all(
        isinstance(encoder, PlainConv) and encoder.pooling == "gap"
        for encoder in extractor.image_encoders.values()
    )


def test_plain_conv_pool_feature_map_alias_uses_adaptive_max() -> None:
    enc = PlainConv(_image_space(), features_dim=16, pool_feature_map=True)
    out = enc(torch.zeros(2, 3, 64, 64))

    assert enc.pooling == "adaptive_max"
    assert out.shape == (2, 16)


def test_plain_conv_factory_passes_cli_options() -> None:
    factory = image_encoder_factory_from_args(
        VisionArgs(
            encoder="plain_conv",
            plain_conv_weight_init="orthogonal",
            plain_conv_last_act=False,
            plain_conv_pooling="gap",
        )
    )
    enc = factory(_image_space())

    assert isinstance(enc, PlainConv)
    assert enc.weight_init == "orthogonal"
    assert enc.pooling == "gap"
    assert not any(isinstance(module, nn.ReLU) for module in enc.fc.modules())


def test_random_shifts_aug_with_generator_preserves_cpu_rng_state() -> None:
    torch.manual_seed(123)
    x = torch.rand(4, 3, 32, 32)
    state_before = torch.get_rng_state()

    generator = torch.Generator(device=x.device)
    generator.manual_seed(999)
    _ = RandomShiftsAug(padding=4)(x, generator=generator)

    assert torch.equal(torch.get_rng_state(), state_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_shifts_aug_with_generator_preserves_cuda_rng_state() -> None:
    torch.cuda.manual_seed_all(123)
    x = torch.rand(4, 3, 32, 32, device="cuda")
    state_before = torch.cuda.get_rng_state()

    generator = torch.Generator(device=x.device)
    generator.manual_seed(999)
    _ = RandomShiftsAug(padding=4).to(x.device)(x, generator=generator)

    assert torch.equal(torch.cuda.get_rng_state(), state_before)


def test_combined_extractor_default_prepare_batch_is_noop() -> None:
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        }
    )
    extractor = CombinedExtractor(obs_space, image_keys=("rgb",))
    obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    keys_before = set(obs)

    extractor.prepare_batch(obs)

    assert set(obs) == keys_before


def test_combined_extractor_random_shift_stack_channels_caches_obs_and_next() -> None:
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "depth": spaces.Box(0.0, 1.0, (64, 64, 1), dtype=np.float32),
        }
    )
    image_encoder = RecordingImageEncoder(
        spaces.Box(0.0, 1.0, (4, 64, 64), dtype=np.float32)
    )
    extractor = CombinedExtractor(
        obs_space,
        image_keys=("rgb", "depth"),
        fusion_mode="stack_channels",
        image_augmentation="random_shift",
        augmentation_seed=7,
        image_encoder_factory=lambda _space: image_encoder,
    )
    obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "depth": torch.rand(2, 64, 64, 1),
    }
    next_obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "depth": torch.rand(2, 64, 64, 1),
    }
    original_rgb = obs["rgb"].clone()
    original_next_rgb = next_obs["rgb"].clone()

    extractor.prepare_batch(obs, next_obs)
    _ = extractor.extract(obs)
    _ = extractor.extract(next_obs)

    assert torch.equal(obs["rgb"], original_rgb)
    assert torch.equal(next_obs["rgb"], original_next_rgb)
    assert len(image_encoder.inputs) == 2
    assert image_encoder.inputs[0].shape == (2, 4, 64, 64)
    assert image_encoder.inputs[1].shape == (2, 4, 64, 64)


def test_combined_extractor_random_shift_per_key_caches_independent_images() -> None:
    obs_space = spaces.Dict(
        {
            "rgb_base": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "rgb_hand": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        }
    )
    encoders: list[RecordingImageEncoder] = []

    def _factory(space: spaces.Box) -> RecordingImageEncoder:
        encoder = RecordingImageEncoder(space)
        encoders.append(encoder)
        return encoder

    extractor = CombinedExtractor(
        obs_space,
        image_keys=("rgb_base", "rgb_hand"),
        fusion_mode="per_key",
        image_augmentation="random_shift",
        augmentation_seed=11,
        image_encoder_factory=_factory,
    )
    base = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    hand = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    obs = {"rgb_base": base.clone(), "rgb_hand": hand.clone()}

    extractor.prepare_batch(obs)
    _ = extractor.extract(obs)

    assert torch.equal(obs["rgb_base"], base)
    assert torch.equal(obs["rgb_hand"], hand)
    assert len(encoders) == 2
    assert [len(encoder.inputs) for encoder in encoders] == [1, 1]
    assert encoders[0].inputs[0].shape == (2, 3, 64, 64)
    assert encoders[1].inputs[0].shape == (2, 3, 64, 64)
