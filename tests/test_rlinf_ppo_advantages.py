from __future__ import annotations

import torch

from rl_garden.buffers.gae import compute_gae
from rl_garden.integrations.rlinf.ppo_advantages import (
    compute_advantages_from_rollout_batch,
)


def test_compute_advantages_from_rollout_batch_matches_manual_slicing():
    """Regression test for the RLinf-batch-shape -> compute_gae reshape.

    ``rollout_batch["rewards"]`` is ``[T, B, 1]``; ``["terminations"]``/
    ``["prev_values"]`` carry one extra *trailing* bootstrap-step entry,
    ``[T+1, B, 1]`` (see ``ppo_advantages.py``'s docstring for the source
    citations). Distinct values at every (step, env) position, so an
    off-by-one or wrong-axis-squeeze bug in the conversion would produce a
    numerically different result from directly slicing the same underlying
    tensors and calling ``compute_gae`` -- the already-tested GAE math
    itself (see ``tests/test_ppo.py::test_rollout_buffer_computes_gae_returns``)
    is not what this test is checking.
    """
    torch.manual_seed(0)
    t_steps, num_envs, obs_dim = 4, 3, 5
    gamma, gae_lambda = 0.9, 0.8

    rewards = torch.rand(t_steps, num_envs, 1)
    terminations_full = torch.zeros(t_steps + 1, num_envs, 1)
    terminations_full[2, 1, 0] = 1.0  # env 1 terminates mid-rollout at step 2
    values_full = torch.rand(t_steps + 1, num_envs, 1)
    next_states = torch.randn(t_steps, num_envs, obs_dim)

    rollout_batch = {
        "rewards": rewards,
        "terminations": terminations_full,
        "prev_values": values_full,
        "next_obs": {"states": next_states},
    }

    def value_head_forward(states: torch.Tensor) -> torch.Tensor:
        # Deterministic, distinguishable function of the input so a
        # wrong-slice bug (e.g. using next_states[step] instead of
        # next_states[step-1]) changes the result.
        return states.sum(dim=-1, keepdim=True)

    advantages, returns = compute_advantages_from_rollout_batch(
        rollout_batch, value_head_forward, gamma, gae_lambda
    )
    assert advantages.shape == (t_steps, num_envs, 1)
    assert returns.shape == (t_steps, num_envs, 1)

    # Manually reproduce the documented slicing convention and call
    # compute_gae directly -- the two must agree exactly.
    dones = terminations_full[:-1].squeeze(-1).float()
    last_dones = terminations_full[-1].squeeze(-1)
    values = values_full[:-1].squeeze(-1)
    last_values = values_full[-1].squeeze(-1)
    rewards_2d = rewards.squeeze(-1)
    next_values = value_head_forward(
        next_states.reshape(t_steps * num_envs, obs_dim)
    ).reshape(t_steps, num_envs)
    final_values = dones * next_values

    expected_advantages, expected_returns = compute_gae(
        rewards_2d, dones, values, final_values, last_values, last_dones, gamma, gae_lambda
    )

    assert torch.allclose(advantages.squeeze(-1), expected_advantages)
    assert torch.allclose(returns.squeeze(-1), expected_returns)


def test_compute_advantages_from_rollout_batch_zero_termination_case_matches_no_bootstrap_fix():
    """With no terminations, final_values is always zero -- the auto-reset
    fix should be a pure no-op, and results should match plain GAE with a
    zero final_values array (sanity check on the "no termination" branch).
    """
    t_steps, num_envs, obs_dim = 3, 2, 4
    gamma, gae_lambda = 1.0, 1.0

    rewards = torch.ones(t_steps, num_envs, 1)
    terminations_full = torch.zeros(t_steps + 1, num_envs, 1)
    values_full = torch.zeros(t_steps + 1, num_envs, 1)
    next_states = torch.randn(t_steps, num_envs, obs_dim)

    rollout_batch = {
        "rewards": rewards,
        "terminations": terminations_full,
        "prev_values": values_full,
        "next_obs": {"states": next_states},
    }

    def value_head_forward(states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape[0], 1)

    advantages, returns = compute_advantages_from_rollout_batch(
        rollout_batch, value_head_forward, gamma, gae_lambda
    )

    # Matches test_ppo.py::test_rollout_buffer_computes_gae_returns exactly
    # (same rewards=1, gamma=gae_lambda=1, values=0, no termination).
    assert torch.allclose(returns[0], torch.full((num_envs, 1), 3.0))
    assert torch.allclose(returns[1], torch.full((num_envs, 1), 2.0))
    assert torch.allclose(returns[2], torch.full((num_envs, 1), 1.0))
