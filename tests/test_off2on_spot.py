"""Unit tests for Off2OnSPOT: online-switch optimizer/discount reset,
lambd cooling, exploration-noise rollout, and checkpoint compatibility with
the offline SPOT class.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.off2on_spot import Off2OnSPOT
from rl_garden.algorithms.spot import SPOT


@pytest.fixture
def simple_env():
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    env.action_space = spaces.Box(low=-1, high=1, shape=(2, 2), dtype=np.float32)
    return env


def _make_off2on(simple_env, **kwargs) -> Off2OnSPOT:
    defaults = dict(
        buffer_size=200,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=8,
        gamma=0.99,
        training_freq=4,
        utd=1.0,
        net_arch=[16, 16],
        vae_hidden_dim=16,
        vae_iterations=0,
        device="cpu",
        seed=42,
    )
    defaults.update(kwargs)
    return Off2OnSPOT(env=simple_env, **defaults)


def _fill_buffer(buffer, num_steps: int) -> None:
    n = buffer.num_envs
    obs_dim = buffer.obs.shape[-1]
    act_dim = buffer.actions.shape[-1]
    for step in range(num_steps):
        is_last = step == num_steps - 1
        buffer.add(
            torch.randn(n, obs_dim),
            torch.randn(n, obs_dim),
            torch.rand(n, act_dim) * 2 - 1,
            torch.randn(n),
            torch.ones(n) if is_last else torch.zeros(n),
        )


def test_online_switch_rebuilds_optimizers(simple_env):
    agent = _make_off2on(simple_env)
    old_actor_opt = agent.actor_optimizer
    old_critic_opt = agent.critic_optimizer

    agent.switch_to_online_mode()

    assert agent.actor_optimizer is not old_actor_opt
    assert agent.critic_optimizer is not old_critic_opt


def test_online_switch_sets_gamma_to_online_discount(simple_env):
    agent = _make_off2on(simple_env, gamma=0.99, online_discount=0.995)
    assert agent.gamma == 0.99
    agent.switch_to_online_mode()
    assert agent.gamma == 0.995


def test_online_switch_rebuilds_lr_schedulers_around_new_optimizers(simple_env):
    agent = _make_off2on(
        simple_env, lr_schedule="linear_warmup", lr_warmup_steps=100
    )
    critic_sched, actor_sched = agent._lr_schedulers
    assert critic_sched is not None and actor_sched is not None

    agent.switch_to_online_mode()

    new_critic_sched, new_actor_sched = agent._lr_schedulers
    assert new_critic_sched.optimizer is agent.critic_optimizer
    assert new_actor_sched.optimizer is agent.actor_optimizer


def test_current_lambd_without_cooling_stays_constant(simple_env):
    agent = _make_off2on(simple_env, lambd=1.0, lambd_cool=False)
    assert agent._current_lambd() == 1.0
    agent.switch_to_online_mode()
    agent._global_update = 500
    assert agent._current_lambd() == 1.0


def test_current_lambd_cools_only_after_online_switch(simple_env):
    agent = _make_off2on(
        simple_env, lambd=1.0, lambd_cool=True, lambd_end=0.2, max_online_updates=1000
    )
    # Offline: no online start recorded yet, no cooling.
    assert agent._current_lambd() == 1.0

    agent.switch_to_online_mode()
    assert agent._current_lambd() == pytest.approx(1.0)

    agent._global_update += 500  # halfway through max_online_updates
    assert agent._current_lambd() == pytest.approx(0.5)

    agent._global_update += 10_000  # past max_online_updates -> floors at lambd_end
    assert agent._current_lambd() == pytest.approx(0.2)


def test_rollout_action_adds_exploration_noise_after_learning_starts(simple_env):
    agent = _make_off2on(simple_env, expl_noise=0.5, noise_clip=1.0)
    obs = torch.zeros(agent.num_envs, 4)
    torch.manual_seed(0)
    action, _, _ = agent._rollout_action(obs, learning_has_started=True)

    with torch.no_grad():
        features = agent.policy.extract_features(obs)
        deterministic = agent.policy.actor.deterministic_action(features)
    assert not torch.allclose(action, deterministic)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_rollout_action_before_learning_starts_uses_explore_action(simple_env):
    agent = _make_off2on(simple_env)
    obs = torch.zeros(agent.num_envs, 4)
    action, _, info = agent._rollout_action(obs, learning_has_started=False)
    assert action.shape == (agent.num_envs, 2)
    assert info is None


def test_off2on_spot_one_update_smoke(simple_env):
    agent = _make_off2on(simple_env)
    _fill_buffer(agent.replay_buffer, 20)
    metrics = agent.train(4, compute_info=True)
    assert "critic_loss" in metrics
    assert np.isfinite(metrics["critic_loss"])


def test_offline_spot_checkpoint_loads_into_off2on_spot(simple_env, tmp_path):
    offline_env = OfflineEnvSpec(
        simple_env.single_observation_space,
        simple_env.single_action_space,
        num_envs=1,
    )
    offline_agent = SPOT(
        env=offline_env,
        buffer_size=100,
        buffer_device="cpu",
        batch_size=8,
        net_arch=[16, 16],
        vae_hidden_dim=16,
        vae_iterations=5,
        device="cpu",
    )
    obs = torch.randn(1, *offline_env.single_observation_space.shape)
    for _ in range(20):
        offline_agent.replay_buffer.add(
            obs,
            torch.randn_like(obs),
            torch.rand(1, *offline_env.single_action_space.shape) * 2 - 1,
            torch.randn(1),
            torch.zeros(1),
        )
    offline_agent.pretrain_vae()
    offline_agent.train(2)

    path = str(tmp_path / "spot_offline.pt")
    offline_agent.save(path, include_replay_buffer=False)

    online_agent = _make_off2on(simple_env)
    online_agent.load(path, load_replay_buffer=False)

    for key, value in offline_agent.policy.state_dict().items():
        assert torch.equal(value, online_agent.policy.state_dict()[key]), key
    assert online_agent._vae_pretrained is True
