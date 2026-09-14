"""Tests for the CEM/MPPI planner (``rl_garden.planners.mppi``) in isolation
from the full agent/env loop.

Moved from ``tests/test_tdmpc2_planner.py`` (model-based-base plan 1.4): the
planner's world-model argument is now a bare ``LatentConsistencyModel`` (no
``pi``/``Q`` -- those moved to ``TDMPC2Policy``, plan 1.3), and
``policy_prior``/``value_fn`` are injected callbacks instead of
``world_model.pi``/``world_model.Q`` calls.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.planners import mppi
from rl_garden.planners.mppi import PlannerConfig
from rl_garden.policies.tdmpc2_policy import TDMPC2Policy
from rl_garden.world_models.latent_consistency import LatentConsistencyModel


def _make_world_model(latent_dim=8, mlp_dim=8, num_bins=11) -> LatentConsistencyModel:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    encoder = FlattenExtractor(obs_space)
    model = LatentConsistencyModel(
        encoder=encoder, action_dim=2, latent_dim=latent_dim, mlp_dim=mlp_dim, num_bins=num_bins
    )
    model.apply_init()
    return model.eval()


def _make_policy(cfg: PlannerConfig, **model_kwargs) -> TDMPC2Policy:
    world_model = _make_world_model(**model_kwargs)
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    policy = TDMPC2Policy(obs_space, action_space, world_model, cfg, mlp_dim=8, num_q=2, dropout=0.0)
    return policy.eval()


def test_plan_returns_action_within_bounds_and_new_prev_mean():
    torch.manual_seed(0)
    cfg = PlannerConfig(
        action_dim=2, discount=0.9, horizon=2, num_samples=16, num_elites=4, num_pi_trajs=2, iterations=2
    )
    policy = _make_policy(cfg)
    obs = torch.zeros(1, 4)
    state = policy.world_model.observe(None, None, policy.world_model.encode(obs), torch.tensor([True]))

    action, prev_mean = policy.planner.plan(
        policy.world_model, state, policy_prior=policy.policy_prior, value_fn=policy.value_fn,
        prev_mean=None, t0=True,
    )

    assert action.shape == (2,)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert prev_mean.shape == (cfg.horizon, cfg.action_dim)


def test_plan_warm_starts_from_prev_mean_when_not_t0():
    torch.manual_seed(0)
    cfg = PlannerConfig(
        action_dim=2, discount=0.9, horizon=3, num_samples=16, num_elites=4, num_pi_trajs=2, iterations=1
    )
    policy = _make_policy(cfg)
    obs = torch.zeros(1, 4)
    state = policy.world_model.observe(None, None, policy.world_model.encode(obs), torch.tensor([True]))
    prev_mean = torch.tensor([[0.5, 0.5], [0.3, 0.3], [0.1, 0.1]])

    torch.manual_seed(1)
    action_t0, _ = policy.planner.plan(
        policy.world_model, state, policy_prior=policy.policy_prior, value_fn=policy.value_fn,
        prev_mean=None, t0=True, eval_mode=True,
    )
    torch.manual_seed(1)
    action_warm, _ = policy.planner.plan(
        policy.world_model, state, policy_prior=policy.policy_prior, value_fn=policy.value_fn,
        prev_mean=prev_mean, t0=False, eval_mode=True,
    )
    assert action_t0.shape == action_warm.shape == (2,)


def test_estimate_value_accumulates_discounted_reward_and_calls_value_fn_once_at_bootstrap():
    model = _make_world_model()
    horizon, num_samples = 2, 4
    state = {"z": torch.zeros(num_samples, model.latent_dim)}
    actions = torch.zeros(horizon, num_samples, 2)

    calls: list = []

    def policy_prior(s):
        return torch.zeros(s["z"].shape[0], 2)

    def value_fn(s, a):
        calls.append((s["z"].shape, a.shape))
        return torch.full((s["z"].shape[0], 1), 10.0)

    value = mppi._estimate_value(
        model, state, actions, discount=0.5, policy_prior=policy_prior, value_fn=value_fn
    )
    assert value.shape == (num_samples, 1)
    assert torch.isfinite(value).all()
    # The planner's bootstrap value comes from exactly one value_fn call at
    # the end of the horizon, not per-step -- the mid-rollout reward comes
    # from model.reward(), never from value_fn.
    assert len(calls) == 1


def test_estimate_value_uses_online_q_and_accumulates_discounted_reward():
    """Bootstraps with the ONLINE critic (no target=True), matching upstream
    TDMPC2._estimate_value (``3rd_party/tdmpc2/tdmpc2/tdmpc2.py:136``); only
    the TD-target used to train the critic uses the target critic. Perturbing
    ``policy.critic_target`` must not change ``_estimate_value``'s output;
    perturbing ``policy.critic`` (what ``policy.value_fn`` reads) must."""
    torch.manual_seed(0)
    cfg = PlannerConfig(action_dim=2, discount=0.5, horizon=2, num_samples=4)
    policy = _make_policy(cfg)
    model = policy.world_model
    horizon, num_samples = 2, 4
    state = {"z": torch.zeros(num_samples, model.latent_dim)}
    actions = torch.zeros(horizon, num_samples, 2)

    torch.manual_seed(1)
    value = mppi._estimate_value(
        model, state, actions, discount=0.5, policy_prior=policy.policy_prior, value_fn=policy.value_fn
    )
    assert value.shape == (num_samples, 1)
    assert torch.isfinite(value).all()

    # Perturbation must be non-uniform: every hidden layer is a NormedLinear
    # whose LayerNorm cancels a uniform additive shift, and the final Q
    # layer is zero-initialized so a uniform shift there produces uniform
    # (still symmetric-around-0) logits -- either way a constant shift is
    # invisible to two_hot_inv and would false-pass this test.
    torch.manual_seed(2)
    with torch.no_grad():
        for p in policy.critic_target.parameters():
            p.add_(torch.randn_like(p))
    torch.manual_seed(1)
    value_after_target_change = mppi._estimate_value(
        model, state, actions, discount=0.5, policy_prior=policy.policy_prior, value_fn=policy.value_fn
    )
    assert torch.allclose(value, value_after_target_change), (
        "_estimate_value must not be affected by the target critic at all"
    )

    torch.manual_seed(3)
    with torch.no_grad():
        for p in policy.critic.parameters():
            p.add_(torch.randn_like(p))
    torch.manual_seed(1)
    value_after_online_change = mppi._estimate_value(
        model, state, actions, discount=0.5, policy_prior=policy.policy_prior, value_fn=policy.value_fn
    )
    assert not torch.allclose(value, value_after_online_change), (
        "_estimate_value must bootstrap with the online critic (policy.value_fn)"
    )


def test_plan_eval_mode_is_deterministic_given_same_seed():
    cfg = PlannerConfig(
        action_dim=2, discount=0.9, horizon=2, num_samples=16, num_elites=4, num_pi_trajs=2, iterations=2
    )
    policy = _make_policy(cfg)
    obs = torch.zeros(1, 4)
    state = policy.world_model.observe(None, None, policy.world_model.encode(obs), torch.tensor([True]))

    torch.manual_seed(42)
    a1, _ = policy.planner.plan(
        policy.world_model, state, policy_prior=policy.policy_prior, value_fn=policy.value_fn,
        prev_mean=None, t0=True, eval_mode=True,
    )
    torch.manual_seed(42)
    a2, _ = policy.planner.plan(
        policy.world_model, state, policy_prior=policy.policy_prior, value_fn=policy.value_fn,
        prev_mean=None, t0=True, eval_mode=True,
    )
    torch.testing.assert_close(a1, a2)
