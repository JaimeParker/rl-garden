from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.qam_policy import QAMPolicy

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


def _make_policy(**kwargs) -> QAMPolicy:
    fe = FlattenExtractor(observation_space=OBS_SPACE)
    defaults = dict(net_arch=[16, 16], flow_steps=4)
    defaults.update(kwargs)
    return QAMPolicy(OBS_SPACE, ACT_SPACE, fe, **defaults)


def _obs(batch=4):
    return torch.randn(batch, *OBS_SPACE.shape)


def test_mutually_exclusive_bolt_ons_rejected():
    with pytest.raises(ValueError, match="Only one of"):
        _make_policy(fql_alpha=1.0, edit_scale=0.1)


def test_predict_plain_ddpg_in_bounds():
    policy = _make_policy(critic_loss_type="ddpg")
    action = policy.predict(_obs())
    assert action.shape == (4, 3)
    assert torch.all(action >= policy.action_low)
    assert torch.all(action <= policy.action_high)


def test_predict_iql_critic_mode_builds_value_net():
    policy = _make_policy(critic_loss_type="iql")
    assert policy.value is not None
    action = policy.predict(_obs())
    assert action.shape == (4, 3)


def test_predict_fql_alpha_bolt_on():
    policy = _make_policy(fql_alpha=1.0)
    assert policy.one_step_actor is not None
    action = policy.predict(_obs())
    assert action.shape == (4, 3)
    assert torch.all(action >= policy.action_low)
    assert torch.all(action <= policy.action_high)


def test_predict_edit_scale_bolt_on():
    policy = _make_policy(edit_scale=0.1)
    assert policy.edit_actor is not None
    assert policy.edit_alpha is not None
    action = policy.predict(_obs())
    assert action.shape == (4, 3)
    assert torch.all(action >= policy.action_low)
    assert torch.all(action <= policy.action_high)


def test_predict_best_of_n_reranks():
    policy = _make_policy(best_of_n=8)
    action = policy.predict(_obs())
    assert action.shape == (4, 3)


def test_predict_leaves_no_grad_on_any_parameter():
    policy = _make_policy()
    policy.predict(_obs())
    for p in policy.parameters():
        assert p.grad is None


def test_horizon_chunked_action_space():
    fe = FlattenExtractor(observation_space=OBS_SPACE)
    chunked_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
    policy = QAMPolicy(OBS_SPACE, chunked_space, fe, net_arch=[16, 16], flow_steps=4)
    action = policy.predict(_obs())
    assert action.shape == (4, 9)


def test_compute_flow_actions_residual_sums_both_networks():
    policy = _make_policy(residual=True)
    features = policy.extract_features(_obs())
    noises = torch.randn(4, 3)
    with torch.no_grad():
        combined = policy.compute_flow_actions(features, noises, 4, model="slow,fast")
        slow_only = policy.compute_flow_actions(features, noises, 4, model="slow")
    assert not torch.allclose(combined, slow_only)


# ---------------------------------------------------------------------------
# adj_matching -- the highest-risk piece, tested in isolation per the plan.
# ---------------------------------------------------------------------------


def test_adj_matching_shapes_and_finite():
    policy = _make_policy(flow_steps=4)
    features = policy.extract_features(_obs())
    xs, adjs, ts, info = policy.adj_matching(features)
    assert xs.shape == (4, 4, 3)
    assert adjs.shape == (4, 4, 3)
    assert ts.shape == (4, 4, 1)
    assert torch.isfinite(xs).all()
    assert torch.isfinite(adjs).all()
    for v in info.values():
        assert np.isfinite(v)


def test_adj_matching_zero_critic_gradient_reduces_adjoint_to_zero():
    """If the critic's output doesn't depend on the action at all, the
    Q-gradient at the trajectory endpoint is exactly zero, and since the
    backward VJP recursion is linear in the adjoint state
    (adj_new = adj + h*vjp(...)(adj)), a zero adjoint stays exactly zero
    through every reverse step. The single highest-value correctness check
    for the whole adjoint-matching mechanism, mirroring QGF's
    guidance_weight=0 reduction test."""
    policy = _make_policy(flow_steps=4)

    class _ConstantCritic(torch.nn.Module):
        def forward_all(self, features, actions):
            # Same (n_critics, batch, 1) shape as EnsembleQCritic.forward_all,
            # but literally independent of `actions`.
            return torch.zeros(2, features.shape[0], 1, requires_grad=False).expand(
                2, features.shape[0], 1
            ) + features.sum(-1, keepdim=True).unsqueeze(0)

    policy.critic = _ConstantCritic()
    policy.target_critic = _ConstantCritic()

    features = policy.extract_features(_obs())
    _, adjs, _, info = policy.adj_matching(features)
    assert torch.allclose(adjs, torch.zeros_like(adjs), atol=1e-6)
    assert info["adj_max"] == pytest.approx(0.0, abs=1e-6)


def test_adj_matching_backward_vjp_matches_manual_autograd():
    """Cross-checks torch.func.vjp's per-step result (as used in
    adj_matching) against an independently-constructed torch.autograd.grad
    call for the exact same function -- two different mechanisms computing
    the same quantity, agreeing, is strong evidence the vjp wiring is
    correct."""
    policy = _make_policy(flow_steps=4)
    features = policy.extract_features(_obs())
    actor_slow = policy._effective_actor_slow()
    h = 1.0 / policy.flow_steps

    xi = torch.randn(4, 3)
    t = torch.full((4, 1), 0.5)
    cotangent = torch.randn(4, 3)

    def fn(x):
        return 2 * actor_slow(features, x, t + h) - x / (t + h)

    _, vjp_fn = torch.func.vjp(fn, xi)
    (functorch_result,) = vjp_fn(cotangent)

    xi_leaf = xi.clone().requires_grad_(True)
    out = fn(xi_leaf)
    (manual_result,) = torch.autograd.grad(out, xi_leaf, grad_outputs=cotangent)

    assert torch.allclose(functorch_result, manual_result, atol=1e-5)
