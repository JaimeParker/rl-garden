from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from rl_garden.networks import (
    EnsembleQCritic,
    MLPResNet,
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


# --- New: GroupNorm / Dropout / kernel_init / MLPResNet ---

def test_create_mlp_layer_and_group_norm_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot both be True"):
        create_mlp(8, 4, [16, 16], use_layer_norm=True, use_group_norm=True)


def test_create_mlp_with_group_norm():
    mlp = create_mlp(8, 4, [32, 32], use_group_norm=True, num_groups=8)
    assert any(isinstance(m, torch.nn.GroupNorm) for m in mlp.modules())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in mlp.modules())
    y = mlp(torch.randn(5, 8))
    assert y.shape == (5, 4)


def test_create_mlp_dropout_active_only_in_train():
    torch.manual_seed(0)
    mlp = create_mlp(8, 4, [32, 32], dropout_rate=0.5)
    assert any(isinstance(m, torch.nn.Dropout) for m in mlp.modules())
    x = torch.randn(64, 8)
    mlp.train()
    y_train_a = mlp(x)
    y_train_b = mlp(x)
    # In train mode dropout produces stochastic outputs
    assert not torch.allclose(y_train_a, y_train_b)
    mlp.eval()
    assert torch.allclose(mlp(x), mlp(x))


def test_create_mlp_kernel_init_orthogonal():
    mlp = create_mlp(32, 4, [16], kernel_init="orthogonal")
    linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
    # First linear has shape (16, 32) so rows are orthogonal: W @ W.T ~ I
    w = linears[0].weight.detach()
    gram = w @ w.T
    assert torch.allclose(gram, torch.eye(gram.shape[0]), atol=1e-5)
    # Bias should be zeroed by our init helper
    assert torch.allclose(linears[0].bias.detach(), torch.zeros_like(linears[0].bias))


def test_mlp_resnet_forward_shape():
    net = MLPResNet(
        input_dim=12,
        output_dim=4,
        hidden_dim=64,
        num_blocks=3,
        use_layer_norm=True,
    )
    x = torch.randn(8, 12)
    y = net(x)
    assert y.shape == (8, 4)


def test_mlp_resnet_as_trunk():
    net = MLPResNet(input_dim=12, output_dim=-1, hidden_dim=64, num_blocks=2)
    y = net(torch.randn(8, 12))
    assert y.shape == (8, 64)


def test_actor_with_mlp_resnet_backbone():
    actor = SquashedGaussianActor(
        features_dim=10,
        action_space=_action_space(),
        hidden_dims=[64, 64],
        backbone_type="mlp_resnet",
        use_layer_norm=True,
    )
    features = torch.randn(7, 10)
    action, log_prob = actor.action_log_prob(features)
    assert action.shape == (7, 3)
    assert log_prob.shape == (7, 1)


def test_actor_mlp_resnet_rejects_uneven_hidden_dims():
    with pytest.raises(ValueError, match="identical widths"):
        SquashedGaussianActor(
            features_dim=10,
            action_space=_action_space(),
            hidden_dims=[64, 128],
            backbone_type="mlp_resnet",
        )


def test_critic_with_mlp_resnet_backbone():
    critic = EnsembleQCritic(
        features_dim=11,
        action_space=_action_space(),
        hidden_dims=[64, 64],
        n_critics=3,
        backbone_type="mlp_resnet",
        use_group_norm=True,
        num_groups=8,
    )
    features = torch.randn(5, 11)
    actions = torch.randn(5, 3)
    q_all = critic.forward_all(features, actions)
    assert q_all.shape == (3, 5, 1)


def test_actor_dropout_changes_outputs_in_train_mode():
    torch.manual_seed(0)
    actor = SquashedGaussianActor(
        features_dim=10,
        action_space=_action_space(),
        hidden_dims=[64, 64],
        dropout_rate=0.5,
    )
    actor.train()
    features = torch.randn(8, 10)
    mean_a, _ = actor(features)
    mean_b, _ = actor(features)
    assert not torch.allclose(mean_a, mean_b)
    actor.eval()
    mean_c, _ = actor(features)
    mean_d, _ = actor(features)
    assert torch.allclose(mean_c, mean_d)
