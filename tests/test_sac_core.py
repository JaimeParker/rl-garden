from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC


class DummyVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def _agent(**kwargs) -> SAC:
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
    }
    params.update(kwargs)
    return SAC(
        env=DummyVecEnv(),
        **params,
    )


def _fill(agent: SAC, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_sac_redq_target_uses_subsampled_critics():
    agent = _agent(n_critics=10, critic_subsample_size=2)
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)
    next_action, _, next_features = agent.policy.actor_action_log_prob(data.next_obs)

    q_sub = agent.policy.q_values_subsampled(
        next_features,
        next_action,
        subsample_size=agent.critic_subsample_size,
        target=True,
    )
    target_q = agent._target_q(data)

    assert q_sub.shape == (2, agent.batch_size, 1)
    assert target_q.shape == (agent.batch_size, 1)
    assert torch.isfinite(target_q).all()


def test_sac_core_high_utd_update_runs():
    agent = _agent(n_critics=4, critic_subsample_size=2, utd=2.0, batch_size=8)
    _fill(agent)

    info = agent.train(gradient_steps=2)

    assert agent._global_update == 2
    assert info["utd_ratio"] == 2.0
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_sac_actor_loss_uses_all_critics():
    agent = _agent(n_critics=5, critic_subsample_size=2)
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    _, _, features = agent.policy.actor_action_log_prob(data.obs)
    q_all = agent.policy.q_values_subsampled(
        features, data.actions, subsample_size=None, target=False
    )
    loss, log_prob = agent._actor_loss(data.obs)

    assert q_all.shape[0] == 5
    assert loss.shape == ()
    assert log_prob.shape == (agent.batch_size, 1)
