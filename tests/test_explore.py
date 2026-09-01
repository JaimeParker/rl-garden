from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import ExPLORe
from rl_garden.networks import RewardMaskRelabeler, RNDBonus


class DummyVecEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, 2)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, 2)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0 + 1e-4)
        assert torch.all(actions >= -1.0 - 1e-4)
        obs = torch.randn(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _agent(**overrides) -> ExPLORe:
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
    )
    kwargs.update(overrides)
    return ExPLORe(**kwargs)


def _write_offline_h5(path) -> None:
    with h5py.File(path, "w") as f:
        for traj_idx in range(2):
            group = f.create_group(f"traj_{traj_idx}")
            group.create_dataset(
                "obs", data=np.ones((3, 4), dtype=np.float32) * traj_idx
            )
            group.create_dataset("actions", data=np.ones((2, 2), dtype=np.float32))
            group.create_dataset(
                "rewards", data=np.array([1.0, -3.0], dtype=np.float32)
            )
            group.create_dataset("terminated", data=np.array([False, True]))
            group.create_dataset("truncated", data=np.array([False, False]))


def test_relabeler_reduces_prediction_error():
    relabeler = RewardMaskRelabeler(obs_dim=4, action_dim=2, net_arch=(16, 16))
    optimizer = torch.optim.Adam(relabeler.parameters(), lr=1e-2)
    obs = torch.randn(32, 4)
    action = torch.randn(32, 2)
    reward = torch.ones(32) * 2.0
    mask = torch.ones(32)

    first_loss, _ = relabeler.loss(obs, action, reward, mask)
    for _ in range(50):
        loss, _ = relabeler.loss(obs, action, reward, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    last_loss, _ = relabeler.loss(obs, action, reward, mask)
    assert last_loss.item() < first_loss.item()


def test_rnd_bonus_decreases_on_visited_states():
    rnd = RNDBonus(obs_dim=4, action_dim=2, feature_dim=8, net_arch=(16, 16))
    optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    visited_obs = torch.randn(1, 4)
    visited_action = torch.randn(1, 2)
    unvisited_obs = torch.randn(1, 4)
    unvisited_action = torch.randn(1, 2)

    bonus_before = rnd.bonus(visited_obs, visited_action).item()
    for _ in range(50):
        loss = rnd.loss(visited_obs, visited_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    bonus_after = rnd.bonus(visited_obs, visited_action).item()
    bonus_unvisited = rnd.bonus(unvisited_obs, unvisited_action).item()

    assert bonus_after < bonus_before
    assert bonus_after < bonus_unvisited


def test_relabel_modes_dispatch(tmp_path):
    path = tmp_path / "demo.h5"
    _write_offline_h5(path)

    gt_agent = _agent(offline_relabel_type="gt")
    assert gt_agent._relabeler is None
    gt_agent.load_offline_replay_buffer(path, backend="h5", buffer_size=16, offline_data_ratio=0.5)
    offline_sample = gt_agent.offline_replay_buffer.sample(4)
    relabeled = gt_agent._relabel(offline_sample)
    assert torch.equal(relabeled.rewards, offline_sample.rewards)
    assert torch.equal(relabeled.dones, offline_sample.dones)

    min_agent = _agent(offline_relabel_type="min")
    min_agent.load_offline_replay_buffer(path, backend="h5", buffer_size=16, offline_data_ratio=0.5)
    assert min_agent._offline_min_reward.item() == -3.0
    offline_sample = min_agent.offline_replay_buffer.sample(4)
    relabeled = min_agent._relabel(offline_sample)
    assert torch.all(relabeled.rewards == -3.0)
    assert relabeled.dones.shape == offline_sample.dones.shape

    pred_agent = _agent(offline_relabel_type="pred")
    pred_agent.load_offline_replay_buffer(path, backend="h5", buffer_size=16, offline_data_ratio=0.5)
    offline_sample = pred_agent.offline_replay_buffer.sample(4)
    relabeled = pred_agent._relabel(offline_sample)
    assert relabeled.rewards.shape == offline_sample.rewards.shape
    assert not torch.equal(relabeled.rewards, offline_sample.rewards)


def test_sample_train_batch_shapes(tmp_path):
    path = tmp_path / "demo.h5"
    _write_offline_h5(path)

    agent = _agent(offline_relabel_type="pred", use_rnd_online=True, use_rnd_offline=True)
    agent.load_offline_replay_buffer(path, backend="h5", buffer_size=16, offline_data_ratio=0.5)
    agent.replay_buffer.add(
        torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 2), torch.zeros(2), torch.zeros(2)
    )

    batch = agent._sample_train_batch(8)
    assert batch.obs.shape == (8, 4)
    assert batch.actions.shape == (8, 2)
    assert batch.rewards.shape == (8,)

    # offline_data_ratio == 0 falls back to a pure online sample.
    agent.offline_data_ratio = 0.0
    batch = agent._sample_train_batch(4)
    assert batch.obs.shape == (4, 4)


def test_checkpoint_roundtrip(tmp_path):
    agent = _agent(offline_relabel_type="pred", use_rnd_online=True)
    agent.learn(total_timesteps=16)
    path = tmp_path / "explore.pt"
    agent.save(path)

    loaded = _agent(offline_relabel_type="pred", use_rnd_online=True)
    loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
    for key, value in agent._relabeler.state_dict().items():
        assert torch.equal(value, loaded._relabeler.state_dict()[key]), key
    for key, value in agent._rnd.state_dict().items():
        assert torch.equal(value, loaded._rnd.state_dict()[key]), key
