from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.rlpd_policy import RLPDPolicy


class StructuredFeaturesExtractor(BaseFeaturesExtractor):
    """Minimal fake extractor declaring a token_and_prop layout, purely to
    exercise RLPDPolicy's use_pnorm rejection for the SpatialEmbQEnsemble path."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=16)

    def structured_feature_config(self):
        return {"layout": "token_and_prop", "num_patches": 4, "patch_dim": 4, "prop_dim": 0}

    def extract(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        return torch.randn(obs.shape[0], 16)


def _obs_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype="float32")


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype="float32")


def test_rlpd_policy_default_matches_plain_sac_policy_shapes():
    obs_space = _obs_space()
    fe = FlattenExtractor(obs_space)
    policy = RLPDPolicy(
        observation_space=obs_space,
        action_space=_action_space(),
        features_extractor=fe,
        net_arch=[16],
        n_critics=3,
        critic_subsample_size=2,
    )
    features = torch.randn(5, obs_space.shape[0])
    actions = torch.randn(5, 2).clamp(-1, 1)
    action, log_prob = policy.actor.action_log_prob(features)
    assert action.shape == (5, 2)
    q_all = policy.critic.forward_all(features, actions)
    assert q_all.shape == (3, 5, 1)


def test_rlpd_policy_use_pnorm_normalizes_actor_and_critic_trunks():
    obs_space = _obs_space()
    fe = FlattenExtractor(obs_space)
    policy = RLPDPolicy(
        observation_space=obs_space,
        action_space=_action_space(),
        features_extractor=fe,
        net_arch=[16],
        n_critics=3,
        critic_subsample_size=2,
        use_pnorm=True,
    )
    features = torch.randn(5, obs_space.shape[0])
    actions = torch.randn(5, 2).clamp(-1, 1)

    actor_trunk_out = policy.actor.trunk(features)
    torch.testing.assert_close(actor_trunk_out.norm(dim=-1), torch.ones(5))

    critic_trunk_out = policy.critic.trunk_features_first(features, actions)
    torch.testing.assert_close(critic_trunk_out.norm(dim=-1), torch.ones(5))

    # critic_target must be independently initialized, then synced -- same
    # invariant SACPolicy.__init__ enforces for the non-pnorm path.
    for p_a, p_b in zip(policy.critic.parameters(), policy.critic_target.parameters()):
        torch.testing.assert_close(p_a, p_b)
        assert not p_b.requires_grad


def test_rlpd_policy_rejects_use_pnorm_with_token_and_prop_features():
    obs_space = _obs_space()
    fe = StructuredFeaturesExtractor(obs_space)
    with pytest.raises(ValueError, match="token_and_prop"):
        RLPDPolicy(
            observation_space=obs_space,
            action_space=_action_space(),
            features_extractor=fe,
            net_arch=[16],
            n_critics=3,
            critic_subsample_size=2,
            actor_feature_dim=8,
            use_pnorm=True,
        )
