from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import ResidualSAC
from rl_garden.buffers import ResidualDictReplayBuffer, ResidualTensorReplayBuffer
from rl_garden.common import ActionScaler
from rl_garden.common.types import ResidualReplayBufferSample
from rl_garden.policies.base_policies import BasePolicyOutput, BasePolicyProvider


class ConstantBaseProvider(BasePolicyProvider):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        action: torch.Tensor,
    ) -> None:
        super().__init__(observation_space, action_space, device="cpu")
        self.action = action.float()
        self.reset_calls = 0

    def select_action(self, obs):
        if isinstance(obs, dict):
            n = next(iter(obs.values())).shape[0]
            device = next(iter(obs.values())).device
        else:
            n = obs.shape[0]
            device = obs.device
        return BasePolicyOutput(actions=self.action.to(device).expand(n, -1))

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
        env.single_observation_space,
        env.single_action_space,
        torch.tensor([0.05, -0.05]) if base_action is None else base_action,
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
    train_info = agent.train(1, compute_info=True)

    assert torch.allclose(target_action, data.next_base_actions)
    assert actor_loss.shape == ()
    assert log_prob.shape == (agent.batch_size, 1)
    assert torch.isfinite(torch.tensor(train_info["critic_loss"]))


def test_residual_critic_configuration_is_forwarded_to_policy():
    agent = _agent(n_critics=4, critic_subsample_size=2, critic_impl="legacy")

    assert agent.n_critics == 4
    assert agent.critic_subsample_size == 2
    assert agent.critic_impl == "legacy"
    assert agent.policy.n_critics == 4
    assert agent.policy.critic_subsample_size == 2
    assert agent.policy.critic.critic_impl == "legacy"


def test_residual_eval_q_mc_uses_normalized_final_action_for_critic():
    agent = _agent(residual_action_scale=0.0)
    obs, _ = agent.env.reset()

    env_action, critic_action = agent._eval_action_and_critic_action(obs)

    expected_env_action = torch.tensor([[0.05, -0.05], [0.05, -0.05]])
    expected_critic_action = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    torch.testing.assert_close(env_action, expected_env_action)
    torch.testing.assert_close(critic_action, expected_critic_action)


def test_residual_actor_diagnostics_use_base_actions_without_advancing_rng():
    agent = _agent()
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

    torch.manual_seed(123)
    expected = torch.rand(4)

    torch.manual_seed(123)
    diagnostics = agent._actor_diagnostics(data)
    actual = torch.rand(4)

    assert "action_saturation" in diagnostics
    assert "entropy_gaussian" in diagnostics
    torch.testing.assert_close(actual, expected)


def test_residual_sample_train_batch_mixes_online_and_offline_samples():
    agent = _agent()
    agent.offline_replay_buffer = agent._make_residual_replay_buffer(16)
    agent.offline_data_ratio = 0.5
    env = agent.env
    for replay_buffer in (agent.replay_buffer, agent.offline_replay_buffer):
        replay_buffer.add(
            torch.zeros(env.num_envs, 3),
            torch.ones(env.num_envs, 3),
            torch.zeros(env.num_envs, 2),
            torch.zeros(env.num_envs),
            torch.zeros(env.num_envs),
            base_actions=torch.zeros(env.num_envs, 2),
            next_base_actions=torch.zeros(env.num_envs, 2),
        )

    calls: list[tuple[str, int]] = []

    def sample_from(name: str, value: float):
        def _sample(batch_size: int) -> ResidualReplayBufferSample:
            calls.append((name, batch_size))
            return ResidualReplayBufferSample(
                obs=torch.full((batch_size, 3), value),
                next_obs=torch.full((batch_size, 3), value + 0.1),
                actions=torch.full((batch_size, 2), value),
                rewards=torch.full((batch_size,), value),
                dones=torch.zeros(batch_size),
                base_actions=torch.full((batch_size, 2), value + 0.2),
                next_base_actions=torch.full((batch_size, 2), value + 0.3),
            )

        return _sample

    agent.replay_buffer.sample = sample_from("online", 1.0)
    agent.offline_replay_buffer.sample = sample_from("offline", 2.0)

    batch = agent._sample_train_batch(4)

    assert calls == [("online", 2), ("offline", 2)]
    torch.testing.assert_close(batch.obs[:2], torch.full((2, 3), 1.0))
    torch.testing.assert_close(batch.obs[2:], torch.full((2, 3), 2.0))
    torch.testing.assert_close(batch.base_actions[:2], torch.full((2, 2), 1.2))
    torch.testing.assert_close(batch.base_actions[2:], torch.full((2, 2), 2.2))


def test_residual_offline_buffer_defaults_to_loadable_dataset_size(tmp_path):
    path = tmp_path / "residual_demo.h5"
    with h5py.File(path, "w") as f:
        f.attrs["dataset_type"] = "rl_garden_residual_offline"
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.ones((5, 3), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("base_actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset(
            "next_base_actions", data=np.ones((4, 2), dtype=np.float32)
        )
        group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, False, False, True]))
        group.create_dataset("truncated", data=np.array([False, False, False, False]))

    agent = _agent(buffer_size=128)

    loaded = agent.load_offline_replay_buffer(path)

    assert loaded == 4
    assert agent.offline_replay_buffer.num_envs == 1
    assert agent.offline_replay_buffer.buffer_size == 4
    assert agent.offline_replay_buffer.per_env_buffer_size == 4
    assert len(agent.offline_replay_buffer) == 4
