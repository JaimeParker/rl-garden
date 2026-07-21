# tests/test_asymmetric_ppo_policy.py
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders import FlattenExtractor
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.ppo_policy import PPOPolicy


def _obs_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


class _DimExtractor(BaseFeaturesExtractor):
    """Linear-projection stub with a caller-chosen output dim.

    ``FlattenExtractor`` derives its features_dim from the observation shape
    and takes no override, so it can't produce a value encoder whose
    features_dim differs from the actor's over the same Box space.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.proj = torch.nn.Linear(in_dim, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(obs)


def _policy(**kwargs) -> PPOPolicy:
    obs_space = _obs_space()
    actor_ext = kwargs.pop("features_extractor", FlattenExtractor(obs_space))
    return PPOPolicy(
        observation_space=obs_space,
        action_space=_action_space(),
        features_extractor=actor_ext,
        net_arch={"pi": [8], "vf": [8]},
        **kwargs,
    )


def test_default_state_dict_unchanged_when_no_value_encoder():
    policy = _policy()
    assert policy.value_features_extractor is None
    assert "value_features_extractor" not in policy.state_dict()


def test_value_encoder_registered_and_used_in_forward():
    obs_space = _obs_space()
    actor_ext = FlattenExtractor(obs_space)
    value_ext = _DimExtractor(obs_space, features_dim=9)
    policy = _policy(features_extractor=actor_ext, value_features_extractor=value_ext)
    assert any(k.startswith("value_features_extractor.") for k in policy.state_dict())

    obs = torch.randn(3, 5)
    actions, values, log_prob, entropy = policy.forward(obs)
    assert values.shape[0] == 3  # value_net sized off value_ext.features_dim (9)


def test_predict_values_uses_value_encoder():
    obs_space = _obs_space()
    actor_ext = FlattenExtractor(obs_space)
    value_ext = _DimExtractor(obs_space, features_dim=9)
    policy = _policy(features_extractor=actor_ext, value_features_extractor=value_ext)
    obs = torch.randn(4, 5)
    values = policy.predict_values(obs)
    assert values.shape == (4, 1)
