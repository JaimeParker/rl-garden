from __future__ import annotations

import io

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid


class DummyVecEnv:
    """3D action space: 2 continuous ee dims + 1 discrete gripper dim."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, 3)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, 3)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        obs = torch.randn(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _agent(**overrides) -> RLPDHybrid:
    kwargs = dict(
        env=DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
    )
    kwargs.update(overrides)
    return RLPDHybrid(**kwargs)


def test_learns_without_crashing_and_updates_discrete_critic_independently():
    agent = _agent()
    initial_discrete_params = [p.clone() for p in agent.policy.discrete_critic.parameters()]

    agent.learn(total_timesteps=16)

    assert agent._global_step == 16
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_discrete_params, agent.policy.discrete_critic.parameters())
    ), "discrete_critic params never changed"


def test_predict_returns_concatenated_continuous_and_discrete_action():
    agent = _agent()
    action = agent.policy.predict(torch.zeros(2, 4), deterministic=True)
    assert action.shape == (2, 3)


def test_learns_with_demo_buffer_mixed_in():
    agent = _agent()
    agent.init_demo_buffer(buffer_size=32, demo_data_ratio=0.5)

    obs = torch.zeros(4)
    action = torch.zeros(3)
    reward = torch.tensor(1.0)
    done = torch.tensor(False)
    for _ in range(8):
        agent.add_demo_transition(obs, obs, action, reward, done)

    agent.learn(total_timesteps=16)

    assert agent._global_step == 16
    assert len(agent.offline_replay_buffer) == 8


def test_checkpoint_round_trips_discrete_critic_and_dqn_optimizer():
    agent = _agent()
    agent.learn(total_timesteps=8)

    buf = io.BytesIO()
    torch.save(agent.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=False)

    assert "dqn_optimizer" in state["optimizers"]
    assert any("discrete_critic" in k for k in state["policy"].keys())

    fresh = _agent()
    fresh.policy.load_state_dict(state["policy"])
    fresh.dqn_optimizer.load_state_dict(state["optimizers"]["dqn_optimizer"])
