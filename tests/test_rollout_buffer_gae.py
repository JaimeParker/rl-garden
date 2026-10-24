"""GAE episode-boundary alignment of ``RolloutBuffer.compute_returns_and_advantage``.

``RolloutBuffer.dones[t]`` is the after-step flag (the episode ended as a
result of acting on ``obs[t]``), so step ``t``'s own target is cut by
``dones[t]``; ``compute_gae`` itself takes episode-start flags.
"""

from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import RolloutBuffer
from rl_garden.buffers.gae import compute_gae

FINITE_HORIZON_MODES = [False, True]


def _filled_buffer(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    final_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> RolloutBuffer:
    num_steps, num_envs = rewards.shape
    buffer = RolloutBuffer(
        spaces.Box(-1.0, 1.0, shape=(3,)),
        spaces.Box(-1.0, 1.0, shape=(2,)),
        num_steps=num_steps,
        num_envs=num_envs,
        device="cpu",
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    for step in range(num_steps):
        buffer.add(
            torch.zeros(num_envs, 3),
            torch.zeros(num_envs, 2),
            rewards[step],
            dones[step],
            values[step],
            torch.zeros(num_envs),
            final_values=final_values[step],
        )
    return buffer


def _returns(
    rewards,
    dones,
    values=None,
    final_values=None,
    last_values=None,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    finite_horizon_gae: bool = False,
) -> torch.Tensor:
    """Single-env returns for per-step lists; ``dones`` are after-step flags."""
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)
    zeros = torch.zeros_like(rewards)
    values = zeros if values is None else torch.tensor(values).float().unsqueeze(-1)
    final_values = (
        zeros if final_values is None else torch.tensor(final_values).float().unsqueeze(-1)
    )
    last_values = torch.tensor([0.0 if last_values is None else float(last_values)])
    buffer = _filled_buffer(
        rewards, dones, values, final_values, gamma=gamma, gae_lambda=gae_lambda
    )
    buffer.compute_returns_and_advantage(
        last_values, dones[-1], finite_horizon_gae=finite_horizon_gae
    )
    return buffer.returns[:, 0]


@pytest.mark.parametrize(
    "dones,standard,finite_horizon",
    [
        # Episode ends after step 0: step 0 is cut, steps 1-2 run to the window end.
        ([1.0, 0.0, 0.0], [1.0, 2.0, 1.0], [1.0, 1.5, 1.0]),
        # Episode ends after step 1: steps 0-1 form one episode, step 2 another.
        ([0.0, 1.0, 0.0], [2.0, 1.0, 1.0], [1.5, 1.0, 1.0]),
    ],
)
@pytest.mark.parametrize("finite_horizon_gae", FINITE_HORIZON_MODES)
def test_episode_end_cuts_its_own_step(dones, standard, finite_horizon, finite_horizon_gae):
    # gamma = lambda = 1, rewards 1, values 0, final_values 0. Finite-horizon GAE
    # at lambda=1 averages the n-step returns available inside the episode
    # window, hence 1.5 for a two-step tail.
    returns = _returns([1.0, 1.0, 1.0], dones, finite_horizon_gae=finite_horizon_gae)
    expected = finite_horizon if finite_horizon_gae else standard
    assert returns.tolist() == pytest.approx(expected)


@pytest.mark.parametrize("finite_horizon_gae", FINITE_HORIZON_MODES)
def test_terminated_step_one_step_target_is_its_reward(finite_horizon_gae):
    gamma = 0.9
    returns = _returns(
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        values=[0.5, 0.7, 0.9],
        final_values=[0.0, 0.0, 0.0],
        last_values=4.0,
        gamma=gamma,
        gae_lambda=0.0,
        finite_horizon_gae=finite_horizon_gae,
    )
    assert returns[0].item() == pytest.approx(1.0 + gamma * 0.7)
    assert returns[1].item() == pytest.approx(2.0)
    assert returns[2].item() == pytest.approx(3.0 + gamma * 4.0)


@pytest.mark.parametrize("finite_horizon_gae", FINITE_HORIZON_MODES)
def test_truncated_step_bootstraps_from_its_final_value(finite_horizon_gae):
    gamma = 0.9
    returns = _returns(
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        values=[0.5, 0.7, 0.9],
        final_values=[0.0, 5.0, 0.0],
        last_values=4.0,
        gamma=gamma,
        gae_lambda=0.0,
        finite_horizon_gae=finite_horizon_gae,
    )
    assert returns[0].item() == pytest.approx(1.0 + gamma * 0.7)
    assert returns[1].item() == pytest.approx(2.0 + gamma * 5.0)
    assert returns[2].item() == pytest.approx(3.0 + gamma * 4.0)


@pytest.mark.parametrize("finite_horizon_gae", FINITE_HORIZON_MODES)
def test_mid_window_boundary_does_not_leak_gae(finite_horizon_gae):
    dones = [0.0, 1.0, 0.0, 0.0]
    kwargs = {"gamma": 1.0, "gae_lambda": 1.0, "finite_horizon_gae": finite_horizon_gae}
    base = _returns([1.0, 1.0, 1.0, 1.0], dones, values=[0.0, 0.0, 0.0, 0.0], **kwargs)
    perturbed = _returns(
        [1.0, 1.0, 50.0, 70.0],
        dones,
        values=[0.0, 0.0, 30.0, 40.0],
        last_values=90.0,
        **kwargs,
    )
    expected_first_episode = [1.5, 1.0] if finite_horizon_gae else [2.0, 1.0]
    assert base[:2].tolist() == pytest.approx(expected_first_episode)
    assert perturbed[:2].tolist() == pytest.approx(expected_first_episode)


def test_standard_gae_matches_after_step_reference():
    torch.manual_seed(0)
    num_steps, num_envs, gamma, gae_lambda = 7, 5, 0.9, 0.8
    rewards = torch.randn(num_steps, num_envs)
    dones = (torch.rand(num_steps, num_envs) < 0.3).float()
    values = torch.randn(num_steps, num_envs)
    final_values = dones * torch.randn(num_steps, num_envs)
    last_values = torch.randn(num_envs)
    buffer = _filled_buffer(
        rewards, dones, values, final_values, gamma=gamma, gae_lambda=gae_lambda
    )
    buffer.compute_returns_and_advantage(last_values, dones[-1])

    expected = torch.zeros(num_steps, num_envs)
    running = torch.zeros(num_envs)
    for step in reversed(range(num_steps)):
        next_values = last_values if step == num_steps - 1 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = (
            rewards[step]
            + gamma * (not_done * next_values + final_values[step])
            - values[step]
        )
        running = delta + gamma * gae_lambda * not_done * running
        expected[step] = running
    assert torch.allclose(buffer.advantages, expected, atol=1e-6)
    assert torch.allclose(buffer.returns, expected + values, atol=1e-6)


def test_compute_gae_reads_episode_start_flags():
    # dones[1] = 1: obs[1] starts a new episode, so step 0 is cut.
    advantages, returns = compute_gae(
        rewards=torch.ones(3, 1),
        dones=torch.tensor([[0.0], [1.0], [0.0]]),
        values=torch.zeros(3, 1),
        final_values=torch.zeros(3, 1),
        last_values=torch.zeros(1),
        last_dones=torch.zeros(1),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert returns[:, 0].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert torch.equal(advantages, returns)
