"""Tests for BCPolicy/BC's tanh_squash toggle (default True preserves
existing behavior; False builds UnsquashedGaussianActor instead)."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.bc import BC
from rl_garden.networks import SquashedGaussianActor, UnsquashedGaussianActor


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> BC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        net_arch=[16, 16],
    )
    defaults.update(kwargs)
    return BC(**defaults)


def test_tanh_squash_defaults_to_true_and_squashed_actor():
    agent = _make_agent()
    assert agent.tanh_squash is True
    assert isinstance(agent.policy.actor, SquashedGaussianActor)


def test_tanh_squash_false_builds_unsquashed_actor():
    agent = _make_agent(tanh_squash=False)
    assert isinstance(agent.policy.actor, UnsquashedGaussianActor)


def test_unsquashed_actor_gives_finite_log_prob_at_action_bounds():
    """The correctness argument for tanh_squash=False: tanh-squashed
    evaluate_action_log_prob on expert actions sitting at exactly +/-1
    (near-binary gripper actions in real Franka demos) is numerically
    degenerate. UnsquashedGaussianActor hard-clamps with no Jacobian
    correction, sidestepping this."""
    agent = _make_agent(tanh_squash=False)
    obs = torch.zeros(4, 4)
    boundary_actions = torch.tensor(
        [[1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]
    )
    log_prob, _ = agent.policy.behavior_log_prob(obs, boundary_actions)
    assert torch.isfinite(log_prob).all()


def test_squashed_actor_is_degenerate_at_action_bounds():
    """Confirms the problem tanh_squash=False actually solves -- the
    default (tanh-squashed) actor's log-prob blows up (not necessarily to
    literal +/-inf, but to a numerically-useless magnitude) at the same
    boundary actions, via the tanh Jacobian correction diverging."""
    agent = _make_agent()  # tanh_squash=True (default)
    obs = torch.zeros(4, 4)
    boundary_actions = torch.tensor(
        [[1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]
    )
    log_prob, _ = agent.policy.behavior_log_prob(obs, boundary_actions)
    assert (log_prob.abs() > 1e6).all()


def test_tanh_squash_false_learns_without_crashing():
    agent = _make_agent(tanh_squash=False)
    for _ in range(8):
        obs = torch.randn(1, 4)
        action = torch.rand(1, 3) * 2 - 1
        agent.replay_buffer.add(obs, obs, action, torch.zeros(1), torch.zeros(1))
    metrics = agent.train(2, compute_info=True)
    assert np.isfinite(metrics["actor_loss"])
