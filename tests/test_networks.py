from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from rl_garden.networks import (
    EnsembleQCritic,
    SquashedGaussianActor,
    create_mlp,
    get_actor_critic_arch,
)


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)


def test_get_actor_critic_arch_from_list():
    pi, qf = get_actor_critic_arch([128, 64])
    assert pi == [128, 64]
    assert qf == [128, 64]


def test_get_actor_critic_arch_from_dict():
    pi, qf = get_actor_critic_arch({"pi": [64, 32], "qf": [256, 256]})
    assert pi == [64, 32]
    assert qf == [256, 256]


def test_get_actor_critic_arch_missing_key_raises():
    with pytest.raises(ValueError, match="must contain both 'pi' and 'qf'"):
        get_actor_critic_arch({"pi": [64, 64]})


def test_create_mlp_basic_forward():
    mlp = create_mlp(8, 4, [16, 16], use_layer_norm=True)
    x = torch.randn(5, 8)
    y = mlp(x)
    assert y.shape == (5, 4)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in mlp.modules())


def test_create_mlp_without_output_layer():
    mlp = create_mlp(8, -1, [16, 16])
    x = torch.randn(5, 8)
    y = mlp(x)
    assert y.shape == (5, 16)


def test_actor_tanh_log_std_mode_matches_sac_bounds():
    actor = SquashedGaussianActor(
        features_dim=10,
        action_space=_action_space(),
        hidden_dims=[32, 32],
        log_std_mode="tanh",
        log_std_min=-5.0,
        log_std_max=2.0,
    )
    features = torch.randn(7, 10)
    mean, log_std = actor(features)
    assert mean.shape == (7, 3)
    assert log_std.shape == (7, 3)
    assert torch.all(log_std >= -5.0)
    assert torch.all(log_std <= 2.0)


def test_actor_uniform_std_parameterization():
    actor = SquashedGaussianActor(
        features_dim=6,
        action_space=_action_space(),
        hidden_dims=[16],
        std_parameterization="uniform",
        log_std_mode="clamp",
        log_std_min=-20.0,
        log_std_max=2.0,
    )
    features = torch.randn(4, 6)
    action, log_prob = actor.action_log_prob(features)
    assert action.shape == (4, 3)
    assert log_prob.shape == (4, 1)


def test_ensemble_q_critic_forward_all_shape():
    critic = EnsembleQCritic(
        features_dim=11,
        action_space=_action_space(),
        hidden_dims=[32, 32],
        n_critics=5,
    )
    features = torch.randn(9, 11)
    actions = torch.randn(9, 3)
    q_all = critic.forward_all(features, actions)
    assert q_all.shape == (5, 9, 1)
