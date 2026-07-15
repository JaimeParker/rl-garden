from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import TD3BC, OfflineEnvSpec


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)}),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> TD3BC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
    )
    defaults.update(kwargs)
    return TD3BC(**defaults)


def _fill(agent: TD3BC, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_rejects_dict_observation_space():
    with pytest.raises(TypeError):
        TD3BC(env=_dict_env(), buffer_device="cpu", device="cpu")


def test_gradient_step_produces_finite_losses():
    agent = _make_agent(policy_freq=1)
    _fill(agent)
    agent.fit_obs_normalizer()
    metrics = agent.train(1, compute_info=True)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["bc_loss"])


def test_policy_freq_delays_actor_and_target_updates():
    agent = _make_agent(policy_freq=3)
    _fill(agent)
    agent.fit_obs_normalizer()

    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    target_before = [p.clone() for p in agent.policy.actor_target.parameters()]

    # Steps 1 and 2 (global_update % 3 != 0): critic updates, actor/target frozen.
    for _ in range(2):
        metrics = agent.train(1, compute_info=True)
        assert "actor_loss" not in metrics

    assert all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(target_before, agent.policy.actor_target.parameters())
    )

    # Step 3 (global_update % 3 == 0): actor and target update.
    metrics = agent.train(1, compute_info=True)
    assert "actor_loss" in metrics
    assert not all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(target_before, agent.policy.actor_target.parameters())
    )


def test_obs_normalizer_fits_dataset_mean_std():
    agent = _make_agent()
    env = agent.env
    obs = torch.randn(500, *env.single_observation_space.shape) * 3.0 + 5.0
    for i in range(obs.shape[0] - 1):
        agent.replay_buffer.add(
            obs[i : i + 1],
            obs[i + 1 : i + 2],
            torch.zeros(1, 3),
            torch.zeros(1),
            torch.zeros(1),
        )
    agent.fit_obs_normalizer()
    assert torch.allclose(agent.policy.obs_mean, obs[:-1].mean(0), atol=0.5)
    assert torch.allclose(agent.policy.obs_std, obs[:-1].std(0), atol=0.5)


def test_checkpoint_round_trip_preserves_normalizer_and_targets():
    agent = _make_agent(policy_freq=1)
    _fill(agent)
    agent.fit_obs_normalizer()
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent(policy_freq=1)
        loaded.load(path, load_replay_buffer=False)

    assert torch.allclose(agent.policy.obs_mean, loaded.policy.obs_mean)
    assert torch.allclose(agent.policy.obs_std, loaded.policy.obs_std)
    for a, b in zip(agent.policy.actor.parameters(), loaded.policy.actor.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(
        agent.policy.actor_target.parameters(), loaded.policy.actor_target.parameters()
    ):
        assert torch.allclose(a, b)
    for a, b in zip(
        agent.policy.critic_target.parameters(), loaded.policy.critic_target.parameters()
    ):
        assert torch.allclose(a, b)
