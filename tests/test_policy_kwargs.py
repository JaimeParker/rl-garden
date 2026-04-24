from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import RGBDSAC, SAC
from rl_garden.encoders import BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = action_space


class RecordingExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 13,
        marker: str = "default",
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.marker = marker

    def forward(self, obs):
        batch = obs.shape[0] if isinstance(obs, torch.Tensor) else next(iter(obs.values())).shape[0]
        return torch.zeros(batch, self.features_dim)


class RecordingImageExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 17,
        marker: str = "image",
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.marker = marker

    def forward(self, obs):
        return torch.zeros(obs.shape[0], self.features_dim)


def _state_env() -> DummyVecEnv:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _rgbd_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _agent_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 8,
        "batch_size": 2,
        "eval_freq": 0,
    }


def test_sac_uses_flatten_extractor_by_default():
    agent = SAC(env=_state_env(), **_agent_kwargs())
    assert isinstance(agent.policy.features_extractor, FlattenExtractor)


def test_sac_policy_kwargs_can_build_custom_extractor():
    agent = SAC(
        env=_state_env(),
        **_agent_kwargs(),
        policy_kwargs={
            "features_extractor_class": RecordingExtractor,
            "features_extractor_kwargs": {"features_dim": 23, "marker": "state-custom"},
        },
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, RecordingExtractor)
    assert extractor.features_dim == 23
    assert extractor.marker == "state-custom"


def test_rgbdsac_uses_combined_extractor_by_default():
    agent = RGBDSAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        image_keys=("rgb", "depth"),
    )
    assert isinstance(agent.policy.features_extractor, CombinedExtractor)


def test_rgbdsac_legacy_image_encoder_factory_still_works():
    def factory(img_space):
        return RecordingImageExtractor(img_space, features_dim=19, marker="legacy")

    agent = RGBDSAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        image_keys=("rgb",),
        image_encoder_factory=factory,
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert isinstance(extractor.image_encoder, RecordingImageExtractor)
    assert extractor.image_encoder.marker == "legacy"
    assert extractor.image_keys == ("rgb",)


def test_rgbdsac_policy_kwargs_can_override_with_custom_extractor():
    agent = RGBDSAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        policy_kwargs={
            "features_extractor_class": RecordingExtractor,
            "features_extractor_kwargs": {"features_dim": 29, "marker": "rgbd-custom"},
        },
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, RecordingExtractor)
    assert extractor.features_dim == 29
    assert extractor.marker == "rgbd-custom"


def test_rgbdsac_policy_kwargs_win_over_legacy_extractor_args():
    def legacy_factory(img_space):
        return RecordingImageExtractor(img_space, features_dim=19, marker="legacy")

    def new_factory(img_space):
        return RecordingImageExtractor(img_space, features_dim=31, marker="policy")

    agent = RGBDSAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        image_keys=("rgb", "depth"),
        image_encoder_factory=legacy_factory,
        policy_kwargs={
            "features_extractor_kwargs": {
                "image_keys": ("rgb",),
                "image_encoder_factory": new_factory,
            }
        },
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert extractor.image_keys == ("rgb",)
    assert isinstance(extractor.image_encoder, RecordingImageExtractor)
    assert extractor.image_encoder.marker == "policy"


def test_unknown_policy_kwargs_raise_clear_error():
    with pytest.raises(ValueError, match="Unsupported policy_kwargs keys"):
        SAC(
            env=_state_env(),
            **_agent_kwargs(),
            policy_kwargs={"unknown_key": 1},
        )
