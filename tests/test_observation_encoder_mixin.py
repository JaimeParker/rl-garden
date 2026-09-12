"""Tests for rl_garden/algorithms/_observation.py: the Layer C observation-
encoder mixin every algorithm inherits from BaseAlgorithm (see the
observation-redesign plan and .claude/plans/observation-redesign-phase2-
recipe.md). Covers the three encoder_sharing modes, the asymmetric-obs-
groups/critic-encoder-config validation, and checkpoint-metadata round-
tripping via SAC (the W3 reference migration).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.algorithms._observation import (
    ObservationEncoderMixin,
    ObservationEncoders,
    resolve_observation_encoders,
)
from rl_garden.encoders.combined import CombinedExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.observations import ObservationContractError, ObsGroups


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = action_space


STATE_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
RGBD_SPACE = spaces.Dict(
    {
        "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        "depth_cam": spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }
)


# --- resolve_observation_encoders: the module-level entry point ---


def test_resolve_box_space_builds_flatten_extractor_actor_only():
    result = resolve_observation_encoders(STATE_SPACE, None, None, "shared_critic_grad")
    assert isinstance(result, ObservationEncoders)
    assert isinstance(result.actor, FlattenExtractor)
    assert result.critic is None
    assert result.critic_or_actor is result.actor
    assert result.sharing == "shared_critic_grad"


def test_resolve_dict_space_shared_builds_one_combined_extractor():
    result = resolve_observation_encoders(RGBD_SPACE, None, None, "shared_critic_grad")
    assert isinstance(result.actor, CombinedExtractor)
    assert result.critic is None
    assert result.critic_or_actor is result.actor


def test_resolve_separate_sharing_builds_two_independent_extractors():
    result = resolve_observation_encoders(RGBD_SPACE, None, None, "separate")
    assert isinstance(result.actor, CombinedExtractor)
    assert isinstance(result.critic, CombinedExtractor)
    assert result.critic is not result.actor
    assert result.critic_or_actor is result.critic


def test_resolve_separate_uses_critic_encoder_config_when_given():
    critic_cfg = EncoderConfig(features_dim=17)
    result = resolve_observation_encoders(
        RGBD_SPACE, EncoderConfig(features_dim=31), None, "separate", critic_encoder_config=critic_cfg
    )
    assert result.actor.image_encoder.features_dim == 31
    assert result.critic.image_encoder.features_dim == 17


def test_resolve_asymmetric_obs_groups_requires_separate_sharing():
    groups = ObsGroups(actor=("rgb_cam", "state"), critic=("rgb_cam", "depth_cam", "state"))
    with pytest.raises(ValueError, match="encoder_sharing='separate'"):
        resolve_observation_encoders(RGBD_SPACE, None, groups, "shared_critic_grad")


def test_resolve_asymmetric_obs_groups_with_separate_sharing_builds_distinct_schemas():
    groups = ObsGroups(actor=("rgb_cam", "state"), critic=("rgb_cam", "depth_cam", "state"))
    result = resolve_observation_encoders(RGBD_SPACE, None, groups, "separate")
    assert set(result.actor.image_keys) == {"rgb_cam"}
    assert set(result.critic.image_keys) == {"rgb_cam", "depth_cam"}


def test_resolve_critic_encoder_config_without_separate_sharing_raises():
    with pytest.raises(ValueError, match="encoder_sharing='separate'"):
        resolve_observation_encoders(
            RGBD_SPACE, None, None, "shared", critic_encoder_config=EncoderConfig()
        )


def test_resolve_unknown_obs_group_key_raises_contract_error():
    groups = ObsGroups(actor=("not_a_real_key",))
    with pytest.raises(ObservationContractError):
        resolve_observation_encoders(RGBD_SPACE, None, groups, "shared_critic_grad")


# --- ObservationEncoderMixin: the per-algorithm helper ---


class _FakeAlgo(ObservationEncoderMixin):
    """Minimal object exercising the mixin without a full BaseAlgorithm."""

    def __init__(self, encoder_sharing: str = "shared_critic_grad", **kwargs) -> None:
        self.encoder_sharing = encoder_sharing
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_mixin_default_encoder_sharing_is_shared_critic_grad():
    assert ObservationEncoderMixin.encoder_sharing == "shared_critic_grad"


def test_mixin_resolve_observation_encoders_reads_optional_attrs_with_none_default():
    algo = _FakeAlgo()  # no encoder_config/obs_groups/critic_encoder_config set at all
    result = algo._resolve_observation_encoders(STATE_SPACE)
    assert algo.observation_encoders is result
    assert isinstance(result.actor, FlattenExtractor)


def test_mixin_actor_features_stop_gradients_under_shared_critic_grad():
    algo = _FakeAlgo(encoder_sharing="shared_critic_grad")
    algo._resolve_observation_encoders(RGBD_SPACE)
    obs = {
        "rgb_cam": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).float(),
        "depth_cam": torch.rand(2, 64, 64, 1),
        "state": torch.randn(2, 4, requires_grad=False),
    }
    features = algo._actor_features(obs)
    loss = features.sum()
    # stop_gradient only detaches the image branch (CombinedExtractor's
    # established Q-loss-only image-encoder convention); the proprio branch
    # still trains from the actor loss -- see CombinedExtractor.extract.
    image_grads = torch.autograd.grad(
        loss, list(algo.observation_encoders.actor.image_encoder.parameters()), allow_unused=True
    )
    assert all(g is None for g in image_grads)


def test_mixin_actor_features_do_not_stop_gradients_under_shared():
    algo = _FakeAlgo(encoder_sharing="shared")
    algo._resolve_observation_encoders(RGBD_SPACE)
    obs = {
        "rgb_cam": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).float(),
        "depth_cam": torch.rand(2, 64, 64, 1),
        "state": torch.randn(2, 4),
    }
    features = algo._actor_features(obs)
    loss = features.sum()
    grads = torch.autograd.grad(
        loss, list(algo.observation_encoders.actor.parameters()), allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_mixin_critic_features_uses_critic_or_actor_encoder():
    algo = _FakeAlgo(encoder_sharing="separate")
    algo._resolve_observation_encoders(RGBD_SPACE)
    obs = {
        "rgb_cam": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).float(),
        "depth_cam": torch.rand(2, 64, 64, 1),
        "state": torch.randn(2, 4),
    }
    critic_features = algo._critic_features(obs)
    assert critic_features.shape[0] == 2
    assert algo.observation_encoders.critic is not algo.observation_encoders.actor


# --- SAC end-to-end: the reference migration, checkpoint round-trip ---


def _sac_kwargs(**overrides) -> dict:
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 8,
        "batch_size": 2,
        "eval_freq": 0,
    }
    params.update(overrides)
    return params


def test_sac_box_default_encoder_sharing_and_metadata_round_trip():
    agent = SAC(env=DummyVecEnv(STATE_SPACE, ACT_SPACE), **_sac_kwargs())
    assert agent.encoder_sharing == "shared_critic_grad"
    meta = agent._checkpoint_metadata()
    assert meta["encoder_sharing"] == "shared_critic_grad"
    assert meta["encoder_config"] is None
    assert meta["obs_groups"] is None
    assert meta["critic_encoder_config"] is None


def test_sac_dict_encoder_config_round_trips_through_checkpoint_metadata():
    cfg = EncoderConfig(backbone="plain_conv", features_dim=42, plain_conv_pooling="gap")
    agent = SAC(env=DummyVecEnv(RGBD_SPACE, ACT_SPACE), encoder_config=cfg, **_sac_kwargs())
    meta = agent._checkpoint_metadata()
    assert meta["encoder_config"] == dataclasses.asdict(cfg)
    assert isinstance(agent.policy.features_extractor, CombinedExtractor)
    # features_dim = image sub-encoder's 42 + the proprio branch's default 64.
    assert agent.policy.features_extractor.features_dim == 42 + 64
    assert agent.policy.features_extractor.image_encoder.features_dim == 42


def test_sac_obs_groups_round_trips_through_checkpoint_metadata():
    groups = ObsGroups(actor=("rgb_cam", "state"), critic=("rgb_cam", "depth_cam", "state"))
    agent = SAC(
        env=DummyVecEnv(RGBD_SPACE, ACT_SPACE),
        obs_groups=groups,
        encoder_sharing="separate",
        **_sac_kwargs(),
    )
    meta = agent._checkpoint_metadata()
    assert meta["obs_groups"] == dataclasses.asdict(groups)
    assert set(agent.policy.features_extractor.image_keys) == {"rgb_cam"}
    assert set(agent.policy.critic_features_extractor.image_keys) == {"rgb_cam", "depth_cam"}


def test_sac_asymmetric_obs_groups_without_separate_sharing_raises():
    groups = ObsGroups(actor=("rgb_cam", "state"), critic=("rgb_cam", "depth_cam", "state"))
    with pytest.raises(ValueError, match="encoder_sharing='separate'"):
        SAC(env=DummyVecEnv(RGBD_SPACE, ACT_SPACE), obs_groups=groups, **_sac_kwargs())
