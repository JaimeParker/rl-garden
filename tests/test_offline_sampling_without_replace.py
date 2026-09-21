"""Regression coverage for the `offline_sampling="without_replace"` fix.

Eleven offline algorithms' `_sample_train_batch` looked up a nonexistent
`ReplayBuffer.sample_without_replace` attribute instead of the real
`sample_without_repeat` (rl_garden/buffers/_sampling.py). This test builds
tiny CPU agents for a representative subset (FQL, IQL, AWAC, TD3BC), fills
their buffers, and drives `train()` under `offline_sampling="without_replace"`
to confirm the fix works end-to-end, plus a direct `_sample_train_batch`
shape check.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import AWAC, FQL, IQL, TD3BC, OfflineEnvSpec


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _fill(agent, steps: int = 32) -> None:
    # env.single_observation_space is Dict({"state": Box}) -- OfflineEnvSpec's
    # bare Box is boundary-normalized by BaseAlgorithm.__init__ (see
    # rl_garden.envs.wrappers.VectorizedDictStateWrapper).
    env = agent.env
    state_shape = env.single_observation_space["state"].shape
    for _ in range(steps):
        obs = {"state": torch.randn(env.num_envs, *state_shape)}
        next_obs = {"state": torch.randn_like(obs["state"])}
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _assert_finite_metrics(metrics: dict) -> None:
    assert metrics
    for key, value in metrics.items():
        assert np.isfinite(value), (key, value)


def test_fql_without_replace_train_step():
    agent = FQL(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        net_arch=[16, 16],
        flow_steps=4,
        offline_sampling="without_replace",
    )
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    _assert_finite_metrics(metrics)


def test_iql_without_replace_train_step():
    agent = IQL(
        env=_state_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=1000,
        batch_size=8,
        net_arch={"pi": [16], "qf": [16], "vf": [16]},
        n_critics=2,
        critic_subsample_size=2,
        offline_sampling="without_replace",
        std_log=False,
    )
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    _assert_finite_metrics(metrics)


def test_awac_without_replace_train_step():
    agent = AWAC(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        offline_sampling="without_replace",
    )
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    _assert_finite_metrics(metrics)


def test_td3_bc_without_replace_train_step():
    agent = TD3BC(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        offline_sampling="without_replace",
    )
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    _assert_finite_metrics(metrics)


def test_sample_train_batch_without_replace_returns_correct_action_shape():
    agent = FQL(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=4,
        device="cpu",
        net_arch=[16, 16],
        flow_steps=4,
        offline_sampling="without_replace",
    )
    _fill(agent)
    batch = agent._sample_train_batch(4)
    action_dim = agent.env.single_action_space.shape[0]
    assert batch.actions.shape == (4, action_dim)
