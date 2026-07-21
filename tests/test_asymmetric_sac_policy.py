# tests/test_asymmetric_sac_policy.py
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders import CombinedExtractor, FlattenExtractor
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.sac_policy import SACPolicy


def _box_obs_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _dict_obs_space(with_privileged: bool = False) -> spaces.Dict:
    d = {
        "rgb": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }
    if with_privileged:
        d["privileged"] = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return spaces.Dict(d)


class _DimExtractor(BaseFeaturesExtractor):
    """Linear-projection stub extractor with a caller-chosen output dim.

    ``FlattenExtractor`` takes no ``features_dim`` override -- it always
    derives its output dim from ``prod(observation_space.shape)`` -- so it
    cannot produce a features_dim that differs from the input shape. Tests
    below need exactly that (e.g. a critic encoder with a different output
    dim than the actor's, over the *same* observation space), hence this
    small stub instead.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.proj = torch.nn.Linear(in_dim, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(obs)


def _policy(**kwargs) -> SACPolicy:
    obs_space = kwargs.pop("observation_space", _box_obs_space())
    extractor = kwargs.pop("features_extractor", FlattenExtractor(obs_space))
    return SACPolicy(
        observation_space=obs_space,
        action_space=_action_space(),
        features_extractor=extractor,
        net_arch={"pi": [8], "qf": [8]},
        **kwargs,
    )


def test_default_state_dict_unchanged_when_no_critic_encoder():
    policy = _policy()
    assert policy.critic_features_extractor is None
    module_names = {name for name, _ in policy.named_modules()}
    assert "critic_features_extractor" not in module_names
    assert "critic_features_extractor" not in policy.state_dict()


def test_critic_encoder_registered_as_submodule_when_configured():
    obs_space = _box_obs_space()
    actor_ext = FlattenExtractor(obs_space)
    critic_ext = FlattenExtractor(obs_space)
    policy = _policy(features_extractor=actor_ext, critic_features_extractor=critic_ext)
    assert policy.critic_features_extractor is critic_ext
    assert policy.has_separate_critic_encoder is True
    module_names = {name for name, _ in policy.named_modules()}
    # FlattenExtractor has no learnable parameters, so state_dict() alone
    # wouldn't show it -- named_modules() reflects module-tree registration
    # regardless of whether a module holds any parameters/buffers.
    assert "critic_features_extractor" in module_names


def test_critic_encoder_sizes_critic_head_from_its_own_features_dim():
    obs_space = _box_obs_space()
    actor_ext = FlattenExtractor(obs_space)
    critic_ext = _DimExtractor(obs_space, features_dim=9)
    policy = _policy(features_extractor=actor_ext, critic_features_extractor=critic_ext)
    # First critic trunk Linear's in_features must be sized off the critic's
    # own features_dim (9 + action_dim), not the actor's (5 + action_dim).
    first_linear = policy.critic.ens_p_trunk__0__weight
    assert first_linear.shape[-1] == 9 + 2


def test_critic_encoder_falls_back_to_shared_when_unset():
    policy = _policy()
    assert policy._critic_encoder() is policy.features_extractor


def test_critic_extract_features_uses_critic_encoder():
    obs_space = _box_obs_space()
    actor_ext = FlattenExtractor(obs_space)
    critic_ext = _DimExtractor(obs_space, features_dim=9)
    policy = _policy(features_extractor=actor_ext, critic_features_extractor=critic_ext)
    obs = torch.randn(3, 5)
    features = policy.critic_extract_features(obs)
    assert features.shape == (3, 9)


def test_actor_and_critic_parameters_partition_disjointly_with_separate_encoders():
    obs_space = _box_obs_space()
    actor_ext = FlattenExtractor(obs_space)
    critic_ext = _DimExtractor(obs_space, features_dim=9)
    policy = _policy(features_extractor=actor_ext, critic_features_extractor=critic_ext)

    actor_params = set(id(p) for p in policy.actor_parameters())
    critic_params = set(id(p) for p in policy.critic_and_encoder_parameters())
    assert actor_params.isdisjoint(critic_params)
    # Actor's own encoder must now be trained by the actor optimizer (it is
    # no longer shared with, or owned by, the critic optimizer).
    actor_encoder_param_ids = set(id(p) for p in actor_ext.parameters())
    assert actor_encoder_param_ids <= actor_params
    critic_encoder_param_ids = set(id(p) for p in critic_ext.parameters())
    assert critic_encoder_param_ids <= critic_params
    assert critic_encoder_param_ids.isdisjoint(actor_params)


def test_actor_and_critic_parameters_unchanged_when_no_critic_encoder():
    policy = _policy()
    actor_params = set(id(p) for p in policy.actor_parameters())
    critic_params = set(id(p) for p in policy.critic_and_encoder_parameters())
    shared_encoder_param_ids = set(id(p) for p in policy.features_extractor.parameters())
    # Default behavior preserved exactly: shared encoder trained only via
    # the critic optimizer, never the actor optimizer.
    assert shared_encoder_param_ids <= critic_params
    assert shared_encoder_param_ids.isdisjoint(actor_params)


def test_structured_feature_config_prefers_critic_encoder():
    class _TokenExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=6):
            super().__init__(observation_space, features_dim)

        def forward(self, obs):
            return torch.zeros(obs.shape[0], self.features_dim)

        def structured_feature_config(self):
            return {
                "layout": "token_and_prop",
                "num_patches": 1,
                "patch_dim": 4,
                "prop_dim": 2,
            }

    obs_space = _box_obs_space()
    actor_ext = FlattenExtractor(obs_space)
    critic_ext = _TokenExtractor(obs_space)
    policy = _policy(features_extractor=actor_ext, critic_features_extractor=critic_ext)

    from rl_garden.networks import SpatialEmbQEnsemble

    assert isinstance(policy.critic, SpatialEmbQEnsemble)
