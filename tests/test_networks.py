from __future__ import annotations

import numpy as np
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


def test_mlp_resnet_block_matches_wsrl_expansion():
    net = MLPResNet(input_dim=12, output_dim=4, hidden_dim=64, num_blocks=2)
    first_block = net.blocks[0]
    assert isinstance(first_block.fc1, torch.nn.Linear)
    assert isinstance(first_block.fc2, torch.nn.Linear)
    assert first_block.fc1.in_features == 64
    assert first_block.fc1.out_features == 64 * 4
    assert first_block.fc2.in_features == 64 * 4
    assert first_block.fc2.out_features == 64
    assert isinstance(first_block.act, torch.nn.SiLU)


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


# ----------------------------------------------------------------------------
# vmap-fused EnsembleQCritic: numerical parity + checkpoint migration
# ----------------------------------------------------------------------------

class _LegacyEnsembleQCritic(torch.nn.Module):
    """Reference ModuleList implementation (pre-vmap) used to verify parity.

    Mirrors the old structure: ``q_nets[i]`` is an independent _QHead, forward
    iterates sequentially. We construct one with the same per-head random init
    as the vmap version, then check that both produce identical outputs.
    """

    def __init__(self, features_dim, action_space, hidden_dims, *, n_critics):
        from rl_garden.networks.actor_critic import _QHead
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.q_nets = torch.nn.ModuleList(
            [
                _QHead(
                    features_dim=features_dim,
                    act_dim=act_dim,
                    hidden_dims=hidden_dims,
                    backbone_type="mlp",
                    use_layer_norm=False,
                    use_group_norm=False,
                    num_groups=32,
                    dropout_rate=None,
                    kernel_init=None,
                )
                for _ in range(n_critics)
            ]
        )

    def forward_all(self, features, actions):
        return torch.stack([q(features, actions) for q in self.q_nets], dim=0)


def test_ensemble_vmap_numerical_parity():
    """vmap critic with weights copied from legacy ModuleList must match exactly."""
    from rl_garden.networks.actor_critic import _PARAM_PREFIX, _safe_name

    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
    torch.manual_seed(0)
    legacy = _LegacyEnsembleQCritic(features_dim=8, action_space=act_space,
                                     hidden_dims=[32, 32], n_critics=4)

    torch.manual_seed(99)  # different init seed for vmap version
    vmap_critic = EnsembleQCritic(features_dim=8, action_space=act_space,
                                  hidden_dims=[32, 32], n_critics=4)

    # Copy legacy weights into vmap's stacked parameters.
    legacy_states = [head.state_dict() for head in legacy.q_nets]
    for dotted in vmap_critic._dotted_param_names:
        stacked = torch.stack([s[dotted] for s in legacy_states], dim=0)
        safe = _safe_name(dotted, _PARAM_PREFIX)
        getattr(vmap_critic, safe).data.copy_(stacked)

    features = torch.randn(7, 8)
    actions = torch.randn(7, 3)
    legacy_out = legacy.forward_all(features, actions)  # (4, 7, 1)
    vmap_out = vmap_critic.forward_all(features, actions)  # (4, 7, 1)
    # Same parameters, deterministic forward → bit-identical or near-identical.
    torch.testing.assert_close(vmap_out, legacy_out, atol=1e-5, rtol=1e-5)


def test_ensemble_vmap_forward_shape():
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
    critic = EnsembleQCritic(
        features_dim=11, action_space=act_space, hidden_dims=[32, 32], n_critics=7
    )
    features = torch.randn(5, 11)
    actions = torch.randn(5, 3)
    q_all = critic.forward_all(features, actions)
    assert q_all.shape == (7, 5, 1)
    q_tuple = critic(features, actions)
    assert len(q_tuple) == 7
    assert all(q.shape == (5, 1) for q in q_tuple)


def test_ensemble_vmap_polyak_update():
    """polyak_update treats stacked params as one tensor → broadcasts naturally."""
    from rl_garden.common.utils import polyak_update

    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    src = EnsembleQCritic(features_dim=6, action_space=act_space,
                          hidden_dims=[16], n_critics=3)
    tgt = EnsembleQCritic(features_dim=6, action_space=act_space,
                          hidden_dims=[16], n_critics=3)
    # Make src very different from tgt initially.
    for p in src.parameters():
        p.data.fill_(1.0)
    for p in tgt.parameters():
        p.data.fill_(0.0)
    # Polyak with tau=0.5 → tgt should average toward 0.5 after one step.
    polyak_update(src.parameters(), tgt.parameters(), tau=0.5)
    for p in tgt.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 0.5))


def test_ensemble_load_legacy_modulelist_state_dict():
    """Old ``q_nets.<i>.<...>`` keys should migrate transparently on load."""
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

    # 1) Build a legacy ModuleList critic; capture its state_dict.
    torch.manual_seed(0)
    legacy = _LegacyEnsembleQCritic(features_dim=5, action_space=act_space,
                                     hidden_dims=[8], n_critics=3)
    features = torch.randn(4, 5)
    actions = torch.randn(4, 2)
    legacy_out = legacy.forward_all(features, actions)

    # 2) Strip the legacy state_dict to the format that would appear inside
    #    a checkpoint of the OLD EnsembleQCritic (keys start with "q_nets.").
    legacy_sd = legacy.state_dict()

    # 3) Build a new vmap EnsembleQCritic (different init).
    torch.manual_seed(123)
    vmap_critic = EnsembleQCritic(features_dim=5, action_space=act_space,
                                  hidden_dims=[8], n_critics=3)

    # 4) Load: should detect q_nets.<i>.* keys and stack along axis 0.
    missing, unexpected = vmap_critic.load_state_dict(legacy_sd, strict=False)
    assert not missing, f"Migration missed keys: {missing}"
    assert not unexpected, f"Unexpected keys after migration: {unexpected}"

    # 5) After migration the vmap critic should match the legacy forward output.
    vmap_out = vmap_critic.forward_all(features, actions)
    torch.testing.assert_close(vmap_out, legacy_out, atol=1e-5, rtol=1e-5)


def test_ensemble_load_intermediate_flat_state_dict():
    """Intermediate ``ens_p_{N}__*`` keys (pre-trunk/head refactor) should
    migrate to the current ``ens_p_trunk__{N}__*`` / ``ens_p_head__*`` layout."""
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

    # 1) Build a current critic and snapshot its state_dict + forward output.
    torch.manual_seed(0)
    source = EnsembleQCritic(features_dim=5, action_space=act_space,
                             hidden_dims=[8, 8], n_critics=3)
    features = torch.randn(4, 5)
    actions = torch.randn(4, 2)
    source_out = source.forward_all(features, actions)

    # 2) Rebuild the intermediate flat-sequential key layout from the current
    #    state_dict. For hidden_dims=[8, 8] the old _QHead was
    #    Sequential(Linear_0, Act_1, Linear_2, Act_3, Linear_4) so linear
    #    params lived at flat indices 0, 2, 4 with index 4 as the output head.
    head_flat_index = 2 * len([8, 8])  # 4
    flat_sd: dict[str, torch.Tensor] = {}
    for k, v in source.state_dict().items():
        if k.startswith("ens_p_trunk__"):
            rest = k[len("ens_p_trunk__"):]  # e.g. "0__weight"
            flat_sd["ens_p_" + rest] = v.clone()
        elif k.startswith("ens_p_head__"):
            suffix = k[len("ens_p_head__"):]  # "weight" or "bias"
            flat_sd[f"ens_p_{head_flat_index}__" + suffix] = v.clone()
        else:
            flat_sd[k] = v.clone()

    # Sanity: the synthetic state_dict must use the OLD key shape.
    assert any(k.startswith("ens_p_0__") for k in flat_sd)
    assert any(k.startswith(f"ens_p_{head_flat_index}__") for k in flat_sd)
    assert not any(k.startswith("ens_p_trunk__") for k in flat_sd)
    assert not any(k.startswith("ens_p_head__") for k in flat_sd)

    # 3) Build a fresh critic with a DIFFERENT init so any leftover params
    #    would show up as numerical mismatch in the forward check below.
    torch.manual_seed(123)
    target = EnsembleQCritic(features_dim=5, action_space=act_space,
                             hidden_dims=[8, 8], n_critics=3)

    # 4) Load: migration should rename keys 1:1 with no leftovers.
    missing, unexpected = target.load_state_dict(flat_sd, strict=False)
    assert not missing, f"Migration missed keys: {missing}"
    assert not unexpected, f"Unexpected keys after migration: {unexpected}"

    # 5) After migration the target critic must reproduce source forward output.
    target_out = target.forward_all(features, actions)
    torch.testing.assert_close(target_out, source_out, atol=1e-5, rtol=1e-5)


def test_ensemble_state_dict_roundtrip_no_false_migration():
    """Current-format state_dict round-trip must NOT trigger either migration
    branch.  Regression guard: ensures the flat-sequential regex does not
    falsely match keys like ``ens_p_trunk__0__weight`` or ``ens_p_head__weight``."""
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

    torch.manual_seed(0)
    source = EnsembleQCritic(features_dim=6, action_space=act_space,
                             hidden_dims=[16, 16], n_critics=4)
    features = torch.randn(3, 6)
    actions = torch.randn(3, 2)
    source_out = source.forward_all(features, actions)
    sd = {k: v.clone() for k, v in source.state_dict().items()}

    torch.manual_seed(999)  # different init so leftover params would diverge
    target = EnsembleQCritic(features_dim=6, action_space=act_space,
                             hidden_dims=[16, 16], n_critics=4)
    missing, unexpected = target.load_state_dict(sd, strict=False)
    assert not missing, f"Round-trip missed keys: {missing}"
    assert not unexpected, f"Round-trip produced unexpected keys: {unexpected}"

    # With identical params, forward output should be bit-identical.
    target_out = target.forward_all(features, actions)
    torch.testing.assert_close(target_out, source_out, atol=1e-7, rtol=1e-7)
