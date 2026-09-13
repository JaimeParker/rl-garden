"""Tests for the Layer B encoder factory (build_observation_encoder) and
CombinedExtractor's schema-driven constructor.

See ``.claude/plans/observation-redesign.md`` for the overall design: this
file covers EncoderConfig, CombinedExtractor's single
``(observation_space, schema, encoder_config)`` constructor, and
build_observation_encoder's dispatch.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import pytest

from rl_garden.encoders import (
    CombinedExtractor,
    EncoderConfig,
    FlattenExtractor,
    build_observation_encoder,
)
from rl_garden.observations import ObservationSchema
from rl_garden.observations.schema import ObservationContractError

STATE_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
RGB_SPACE = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
STACKED_RGB_SPACE = spaces.Box(low=0, high=255, shape=(4, 64, 64, 3), dtype=np.uint8)


def _rgb_state_space() -> spaces.Dict:
    return spaces.Dict({"rgb_cam1": RGB_SPACE, "state": STATE_SPACE})


def test_box_builds_flatten_extractor():
    ext = build_observation_encoder(STATE_SPACE)
    assert isinstance(ext, FlattenExtractor)
    assert ext.features_dim == 8
    out = ext(torch.zeros(2, 8))
    assert out.shape == (2, 8)


def test_dict_state_only_builds_flatten_extractor():
    space = spaces.Dict({"state": STATE_SPACE})
    ext = build_observation_encoder(space)
    assert isinstance(ext, FlattenExtractor)
    assert ext.features_dim == 8
    out = ext({"state": torch.zeros(2, 8)})
    assert out.shape == (2, 8)


def test_dict_rgb_and_state_builds_combined_extractor_with_proprio_branch():
    ext = build_observation_encoder(_rgb_state_space())
    assert isinstance(ext, CombinedExtractor)
    assert ext.has_state
    assert ext.image_keys == ("rgb_cam1",)
    out = ext(
        {
            "rgb_cam1": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
            "state": torch.zeros(2, 8),
        }
    )
    assert out.shape == (2, ext.features_dim)


def test_without_state_drops_proprio_branch():
    space = spaces.Dict({"rgb_cam1": RGB_SPACE})
    ext = build_observation_encoder(space)
    assert isinstance(ext, CombinedExtractor)
    assert not ext.has_state
    assert ext.proprio is None


def test_keys_subset_drops_state():
    ext = build_observation_encoder(_rgb_state_space(), keys=("rgb_cam1",))
    assert isinstance(ext, CombinedExtractor)
    assert not ext.has_state


def test_stacked_image_input():
    space = spaces.Dict({"rgb_cam1": STACKED_RGB_SPACE, "state": STATE_SPACE})
    ext = build_observation_encoder(space)
    assert ext.enable_stacking
    out = ext(
        {
            "rgb_cam1": torch.zeros(2, 4, 64, 64, 3, dtype=torch.uint8),
            "state": torch.zeros(2, 8),
        }
    )
    assert out.shape == (2, ext.features_dim)


def test_per_key_fusion_mode():
    space = spaces.Dict(
        {"rgb_cam1": RGB_SPACE, "rgb_cam2": RGB_SPACE, "state": STATE_SPACE}
    )
    ext = build_observation_encoder(space, EncoderConfig(image_fusion_mode="per_key"))
    assert ext.fusion_mode == "per_key"
    assert set(ext.image_encoders.keys()) == {"rgb_cam1", "rgb_cam2"}
    assert ext.image_encoder is None


def test_stack_channels_fusion_mode():
    space = spaces.Dict(
        {"rgb_cam1": RGB_SPACE, "rgb_cam2": RGB_SPACE, "state": STATE_SPACE}
    )
    ext = build_observation_encoder(
        space, EncoderConfig(image_fusion_mode="stack_channels")
    )
    assert ext.fusion_mode == "stack_channels"
    assert ext.image_encoder is not None
    assert len(ext.image_encoders) == 0


def test_state_only_rejects_non_default_image_field():
    space = spaces.Dict({"state": STATE_SPACE})
    with pytest.raises(ObservationContractError, match="backbone"):
        build_observation_encoder(space, EncoderConfig(backbone="resnet18"))


def test_state_only_allows_normalize_obs_alone():
    space = spaces.Dict({"state": STATE_SPACE})
    ext = build_observation_encoder(space, EncoderConfig(normalize_obs=True))
    assert isinstance(ext, FlattenExtractor)
    assert ext.normalizer is not None


def test_asymmetric_actor_images_critic_state_only_does_not_raise():
    """resolve_observation_encoders calls build_observation_encoder once per
    consumer with a (possibly state-only) SUBSET schema; encoder_config's
    image fields must be judged against the full space, not that subset, or
    an asymmetric actor(images)/critic(state-only) split would incorrectly
    raise on the critic's own call."""
    from rl_garden.algorithms._observation import resolve_observation_encoders
    from rl_garden.observations import ObsGroups

    space = _rgb_state_space()
    encoders = resolve_observation_encoders(
        space,
        EncoderConfig(backbone="resnet18"),
        ObsGroups(actor=("rgb_cam1",), critic=("state",)),
        "separate",
    )
    assert isinstance(encoders.actor, CombinedExtractor)
    assert isinstance(encoders.critic, FlattenExtractor)


EXTRA_STATE_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


def test_flatten_extractor_concatenates_multiple_state_keys():
    space = spaces.Dict({"state": STATE_SPACE, "state_object_pose": EXTRA_STATE_SPACE})
    ext = build_observation_encoder(space)
    assert isinstance(ext, FlattenExtractor)
    assert ext.features_dim == 8 + 3
    out = ext({"state": torch.zeros(2, 8), "state_object_pose": torch.zeros(2, 3)})
    assert out.shape == (2, 11)


def test_combined_extractor_proprio_branch_concatenates_state_keys():
    space = spaces.Dict(
        {"rgb_cam1": RGB_SPACE, "state": STATE_SPACE, "state_object_pose": EXTRA_STATE_SPACE}
    )
    ext = build_observation_encoder(space)
    assert isinstance(ext, CombinedExtractor)
    assert ext.state_keys == ("state", "state_object_pose")
    out = ext(
        {
            "rgb_cam1": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
            "state": torch.zeros(2, 8),
            "state_object_pose": torch.zeros(2, 3),
        }
    )
    assert out.shape == (2, ext.features_dim)


def test_asymmetric_critic_sees_extra_state_actor_does_not():
    from rl_garden.algorithms._observation import resolve_observation_encoders
    from rl_garden.observations import ObsGroups

    space = spaces.Dict(
        {"rgb_cam1": RGB_SPACE, "state": STATE_SPACE, "state_object_pose": EXTRA_STATE_SPACE}
    )
    encoders = resolve_observation_encoders(
        space,
        EncoderConfig(),
        ObsGroups(actor=("rgb_cam1", "state"), critic=("rgb_cam1", "state", "state_object_pose")),
        "separate",
    )
    assert "state_object_pose" not in encoders.actor.state_keys
    assert "state_object_pose" in encoders.critic.state_keys


def test_combined_extractor_direct_construction():
    """CombinedExtractor's only constructor: (observation_space, schema,
    encoder_config)."""
    space = _rgb_state_space()
    schema = ObservationSchema.from_space(space)
    ext = CombinedExtractor(space, schema, EncoderConfig())
    assert ext.image_keys == ("rgb_cam1",)
    assert ext.has_state
    out = ext(
        {"rgb_cam1": torch.zeros(2, 64, 64, 3, dtype=torch.uint8), "state": torch.zeros(2, 8)}
    )
    assert out.shape == (2, ext.features_dim)
