from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import ResidualSAC
from rl_garden.buffers import ResidualDictReplayBuffer, ResidualTensorReplayBuffer
from rl_garden.common import ActionScaler


class ConstantBaseProvider:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action.float()
        self.reset_calls = 0

    def __call__(self, obs):
        if isinstance(obs, dict):
            n = next(iter(obs.values())).shape[0]
            device = next(iter(obs.values())).device
        else:
            n = obs.shape[0]
            device = obs.device
        return self.action.to(device).expand(n, -1)

    def reset(self, env_ids=None) -> None:
        del env_ids
        self.reset_calls += 1


class RawActionVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=np.array([-0.1, -0.1], dtype=np.float32),
            high=np.array([0.1, 0.1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.broadcast_to(
                self.single_action_space.low,
                (self.num_envs,) + self.single_action_space.shape,
            ),
            high=np.broadcast_to(
                self.single_action_space.high,
                (self.num_envs,) + self.single_action_space.shape,
            ),
            dtype=np.float32,
        )
        self.last_actions = None

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, *self.single_observation_space.shape), {}

    def step(self, actions):
        self.last_actions = actions.detach().clone()
        obs = torch.ones(self.num_envs, *self.single_observation_space.shape)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}


def _agent(env=None, base_action=None, **kwargs) -> ResidualSAC:
    env = env or RawActionVecEnv()
    provider = ConstantBaseProvider(
        torch.tensor([0.05, -0.05]) if base_action is None else base_action
    )
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 16,
        "batch_size": 4,
        "learning_starts": 100,
        "training_freq": 2,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [16],
        "residual_action_scale": 0.0,
    }
    params.update(kwargs)
    return ResidualSAC(env=env, base_action_provider=provider, **params)


def test_action_scaler_scales_and_unscales_raw_actions():
    action_space = spaces.Box(
        low=np.array([-0.1, -0.2], dtype=np.float32),
        high=np.array([0.1, 0.2], dtype=np.float32),
        dtype=np.float32,
    )
    scaler = ActionScaler.from_action_space(action_space)

    raw = torch.tensor([[-0.1, 0.0], [0.0, 0.2]])
    normalized = scaler.scale(raw)
    assert torch.allclose(normalized, torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(scaler.unscale(normalized), raw)


def test_residual_tensor_replay_buffer_samples_base_actions():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    rb = ResidualTensorReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")

    rb.add(
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.zeros(2, 2),
        torch.ones(2),
        torch.zeros(2),
        base_actions=torch.full((2, 2), 0.25),
        next_base_actions=torch.full((2, 2), -0.25),
    )

    batch = rb.sample(2)
    assert batch.base_actions.shape == (2, 2)
    assert batch.next_base_actions.shape == (2, 2)
    assert torch.all(batch.base_actions == 0.25)
    assert torch.all(batch.next_base_actions == -0.25)


def test_residual_dict_replay_buffer_samples_base_actions():
    obs_space = spaces.Dict(
        {"state": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)}
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    rb = ResidualDictReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")
    obs = {"state": torch.zeros(2, 3)}
    next_obs = {"state": torch.ones(2, 3)}

    rb.add(
        obs,
        next_obs,
        torch.zeros(2, 2),
        torch.ones(2),
        torch.zeros(2),
        base_actions=torch.full((2, 2), 0.5),
        next_base_actions=torch.full((2, 2), -0.5),
    )

    batch = rb.sample(2)
    assert batch.obs["state"].shape == (2, 3)
    assert batch.base_actions.shape == (2, 2)
    assert batch.next_base_actions.shape == (2, 2)


def test_residual_rollout_unscales_env_action_and_stores_normalized_action():
    env = RawActionVecEnv()
    agent = _agent(env=env, residual_action_scale=0.0)

    agent.learn(total_timesteps=2)

    expected_raw = torch.tensor([[0.05, -0.05], [0.05, -0.05]])
    expected_normalized = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    assert torch.allclose(env.last_actions, expected_raw)
    assert torch.allclose(agent.replay_buffer.actions[0], expected_normalized)
    assert torch.allclose(agent.replay_buffer.base_actions[0], expected_normalized)
    assert torch.allclose(agent.replay_buffer.next_base_actions[0], expected_normalized)


def test_residual_update_hooks_combine_base_and_residual_actions():
    agent = _agent(learning_starts=0)
    env = agent.env
    for _ in range(4):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        base = torch.full((env.num_envs, 2), 0.25)
        next_base = torch.full((env.num_envs, 2), -0.25)
        agent.replay_buffer.add(
            obs,
            next_obs,
            base,
            torch.ones(env.num_envs),
            torch.zeros(env.num_envs),
            base_actions=base,
            next_base_actions=next_base,
        )

    data = agent.replay_buffer.sample(agent.batch_size)
    target_action, _, _ = agent._target_action_log_prob(data)
    actor_loss, log_prob = agent._actor_loss_from_batch(data)
    train_info = agent.train(1)

    assert torch.allclose(target_action, data.next_base_actions)
    assert actor_loss.shape == ()
    assert log_prob.shape == (agent.batch_size, 1)
    assert torch.isfinite(torch.tensor(train_info["critic_loss"]))
