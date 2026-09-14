"""Unit tests for ``rl_garden.encoders.dreamer_conv.DreamerConvEncoder`` and
``rl_garden.world_models.decoder.ObservationDecoder`` (plan model-based-base
Part 2, section C).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.dreamer_conv import DreamerConvEncoder
from rl_garden.observations import ObservationContractError, ObservationSchema
from rl_garden.world_models.decoder import ObservationDecoder


def _multi_modal_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "state": spaces.Box(-1, 1, shape=(6,), dtype=np.float32),
            "rgb_front": spaces.Box(0, 255, shape=(32, 32, 3), dtype=np.uint8),
            "rgb_wrist": spaces.Box(0, 255, shape=(32, 32, 3), dtype=np.uint8),
        }
    )


# ---------------------------------------------------------------------------
# DreamerConvEncoder
# ---------------------------------------------------------------------------


def test_encoder_channel_concatenates_multiple_rgb_keys():
    obs_space = _multi_modal_space()
    schema = ObservationSchema.from_space(obs_space)
    enc = DreamerConvEncoder(obs_space, schema, depth=4, units=16, state_layers=1)
    obs = {
        "state": torch.randn(3, 6),
        "rgb_front": torch.randint(0, 255, (3, 32, 32, 3), dtype=torch.uint8),
        "rgb_wrist": torch.randint(0, 255, (3, 32, 32, 3), dtype=torch.uint8),
    }
    out = enc(obs)
    assert out.shape == (3, enc.features_dim)


def test_encoder_supports_leading_time_and_batch_dims():
    obs_space = _multi_modal_space()
    schema = ObservationSchema.from_space(obs_space)
    enc = DreamerConvEncoder(obs_space, schema, depth=4, units=16, state_layers=1)
    obs = {
        "state": torch.randn(5, 3, 6),
        "rgb_front": torch.randint(0, 255, (5, 3, 32, 32, 3), dtype=torch.uint8),
        "rgb_wrist": torch.randint(0, 255, (5, 3, 32, 32, 3), dtype=torch.uint8),
    }
    out = enc(obs)
    assert out.shape == (5, 3, enc.features_dim)


def test_encoder_rejects_mismatched_image_sizes():
    obs_space = spaces.Dict(
        {
            "rgb_a": spaces.Box(0, 255, shape=(32, 32, 3), dtype=np.uint8),
            "rgb_b": spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
        }
    )
    schema = ObservationSchema.from_space(obs_space)
    with pytest.raises(ObservationContractError):
        DreamerConvEncoder(obs_space, schema)


def test_encoder_rejects_frame_stack():
    obs_space = spaces.Dict({"rgb_front": spaces.Box(0, 255, shape=(2, 32, 32, 3), dtype=np.uint8)})
    schema = ObservationSchema.from_space(obs_space, frame_stack=2)
    with pytest.raises(ObservationContractError):
        DreamerConvEncoder(obs_space, schema)


def test_encoder_rejects_depth_keys():
    obs_space = spaces.Dict({"depth_front": spaces.Box(0, 1, shape=(32, 32, 1), dtype=np.float32)})
    schema = ObservationSchema.from_space(obs_space)
    with pytest.raises(ObservationContractError):
        DreamerConvEncoder(obs_space, schema)


def test_encoder_state_only_schema_still_works():
    obs_space = spaces.Dict({"state": spaces.Box(-1, 1, shape=(6,), dtype=np.float32)})
    schema = ObservationSchema.from_space(obs_space)
    enc = DreamerConvEncoder(obs_space, schema, units=16, state_layers=1)
    out = enc({"state": torch.randn(4, 6)})
    assert out.shape == (4, enc.features_dim)
    assert enc.cnn is None


def test_dreamer_conv_registered_in_registry():
    from rl_garden.encoders.config import EncoderConfig
    from rl_garden.encoders.registry import ENCODER_REGISTRY

    assert "dreamer_conv" in ENCODER_REGISTRY
    factory = EncoderConfig(backbone="dreamer_conv", dreamer_depth=4).image_encoder_factory()
    img_space = spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32)
    encoder = factory(img_space)
    out = encoder(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, encoder.features_dim)


# ---------------------------------------------------------------------------
# ObservationDecoder
# ---------------------------------------------------------------------------


def test_decoder_keys_match_schema_keys():
    obs_space = _multi_modal_space()
    schema = ObservationSchema.from_space(obs_space)
    deter_dim, stoch_dim = 32, 16
    decoder = ObservationDecoder(schema, deter_dim, stoch_dim, depth=4, units=16, state_layers=1)
    dists = decoder(torch.randn(4, 3, deter_dim), torch.randn(4, 3, stoch_dim))
    assert set(dists.keys()) == set(schema.keys)


def test_decoder_image_dist_mode_is_bounded_in_unit_interval():
    obs_space = spaces.Dict({"rgb_front": spaces.Box(0, 255, shape=(32, 32, 3), dtype=np.uint8)})
    schema = ObservationSchema.from_space(obs_space)
    decoder = ObservationDecoder(schema, 32, 16, depth=4, units=16)
    dists = decoder(torch.randn(2, 3, 32), torch.randn(2, 3, 16))
    mode = dists["rgb_front"].mode
    assert mode.shape == (2, 3, 32, 32, 3)
    assert bool((mode >= 0.0).all()) and bool((mode <= 1.0).all())


def test_decoder_log_prob_shape_drops_only_feature_dims():
    obs_space = spaces.Dict({"state": spaces.Box(-1, 1, shape=(6,), dtype=np.float32)})
    schema = ObservationSchema.from_space(obs_space)
    decoder = ObservationDecoder(schema, 32, 16, units=16, state_layers=1)
    dists = decoder(torch.randn(5, 4, 32), torch.randn(5, 4, 16))
    log_prob = dists["state"].log_prob(torch.randn(5, 4, 6))
    assert log_prob.shape == (5, 4)
    assert torch.isfinite(log_prob).all()


def test_decoder_rejects_depth_keys():
    obs_space = spaces.Dict({"depth_front": spaces.Box(0, 1, shape=(32, 32, 1), dtype=np.float32)})
    schema = ObservationSchema.from_space(obs_space)
    with pytest.raises(ObservationContractError):
        ObservationDecoder(schema, 32, 16)


def test_decoder_rejects_mismatched_image_sizes():
    obs_space = spaces.Dict(
        {
            "rgb_a": spaces.Box(0, 255, shape=(32, 32, 3), dtype=np.uint8),
            "rgb_b": spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
        }
    )
    schema = ObservationSchema.from_space(obs_space)
    with pytest.raises(ObservationContractError):
        ObservationDecoder(schema, 32, 16)


def test_decoder_requires_at_least_one_key():
    empty_schema = ObservationSchema(entries={})
    with pytest.raises(ValueError):
        ObservationDecoder(empty_schema, 32, 16, units=16, state_layers=1)
