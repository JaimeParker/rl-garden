from __future__ import annotations

import sys
import types

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import RLPD
from rl_garden.common.types import ReplayBufferSample


class DummyVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (2, 2)),
            high=np.broadcast_to(self.single_action_space.high, (2, 2)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        del actions
        obs = torch.ones(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _agent(**kwargs) -> RLPD:
    params = dict(
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=8,
        learning_starts=100,
        training_freq=2,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        n_critics=2,
        critic_subsample_size=None,
    )
    params.update(kwargs)
    return RLPD(env=DummyVecEnv(), **params)


def _tagged_sampler(value: float):
    def _sample(batch_size: int) -> ReplayBufferSample:
        return ReplayBufferSample(
            obs=torch.full((batch_size, 4), value),
            next_obs=torch.full((batch_size, 4), value + 0.1),
            actions=torch.full((batch_size, 2), value),
            rewards=torch.full((batch_size,), value),
            dones=torch.zeros(batch_size),
        )

    return _sample


def test_sample_train_batch_falls_back_to_online_when_no_offline_buffer():
    agent = _agent()
    agent.replay_buffer.sample = _tagged_sampler(1.0)
    batch = agent._sample_train_batch(8)
    assert torch.all(batch.obs == 1.0)


def test_sample_train_batch_falls_back_to_online_when_ratio_zero():
    agent = _agent()
    agent.offline_replay_buffer = agent._build_prior_data_buffer(16)
    agent.offline_data_ratio = 0.0
    agent.replay_buffer.sample = _tagged_sampler(1.0)
    agent.offline_replay_buffer.sample = _tagged_sampler(2.0)
    batch = agent._sample_train_batch(8)
    assert torch.all(batch.obs == 1.0)


def test_sample_train_batch_falls_back_to_offline_when_online_buffer_empty():
    agent = _agent()
    agent.offline_replay_buffer = agent._build_prior_data_buffer(16)
    agent.offline_data_ratio = 1.0
    agent.offline_replay_buffer.add(
        torch.zeros(1, 4), torch.ones(1, 4), torch.zeros(1, 2), torch.zeros(1), torch.zeros(1)
    )
    agent.offline_replay_buffer.sample = _tagged_sampler(2.0)
    # agent.replay_buffer has never had anything added -> len() == 0.
    batch = agent._sample_train_batch(8)
    assert torch.all(batch.obs == 2.0)


def test_sample_train_batch_mixes_ratio_and_shuffles():
    agent = _agent()
    agent.offline_replay_buffer = agent._build_prior_data_buffer(16)
    agent.offline_data_ratio = 0.5
    agent.replay_buffer.add(
        torch.zeros(agent.env.num_envs, 4),
        torch.ones(agent.env.num_envs, 4),
        torch.zeros(agent.env.num_envs, 2),
        torch.zeros(agent.env.num_envs),
        torch.zeros(agent.env.num_envs),
    )
    agent.offline_replay_buffer.add(
        torch.zeros(1, 4), torch.ones(1, 4), torch.zeros(1, 2), torch.zeros(1), torch.zeros(1)
    )
    agent.replay_buffer.sample = _tagged_sampler(1.0)
    agent.offline_replay_buffer.sample = _tagged_sampler(2.0)

    torch.manual_seed(0)
    batch1 = agent._sample_train_batch(8)
    torch.manual_seed(1)
    batch2 = agent._sample_train_batch(8)

    values1 = batch1.obs[:, 0].tolist()
    values2 = batch2.obs[:, 0].tolist()
    assert sorted(values1) == [1.0] * 4 + [2.0] * 4
    # A block-concat bug ([online]*4 + [offline]*4, unshuffled) would produce
    # the exact same order regardless of the RNG seed -- two different seeds
    # producing different orders proves the batch is actually shuffled, which
    # `train_high_utd`'s sequential minibatch slicing depends on.
    assert values1 != values2


def test_sample_train_batch_matches_nstep_buffer_shape():
    # Regression test: _build_prior_data_buffer must mirror SAC's own
    # nstep-aware buffer choice. Otherwise, with nstep>1, the online buffer
    # samples NStepReplayBufferSample (extra `discounts` field) while a plain
    # offline buffer samples ReplayBufferSample -- _concat_replay_samples
    # would then crash with AttributeError on `discounts`.
    agent = _agent(nstep=3, gamma=0.9)
    agent.offline_replay_buffer = agent._build_prior_data_buffer(16)
    agent.offline_data_ratio = 0.5
    for _ in range(5):
        agent.replay_buffer.add(
            torch.zeros(agent.env.num_envs, 4),
            torch.ones(agent.env.num_envs, 4),
            torch.zeros(agent.env.num_envs, 2),
            torch.zeros(agent.env.num_envs),
            torch.zeros(agent.env.num_envs),
            episode_end=torch.zeros(agent.env.num_envs),
        )
        agent.offline_replay_buffer.add(
            torch.zeros(1, 4),
            torch.ones(1, 4),
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.zeros(1),
            episode_end=torch.zeros(1),
        )

    batch = agent._sample_train_batch(8)
    assert batch.obs.shape == (8, 4)
    assert batch.discounts.shape == (8,)


def test_load_offline_replay_buffer_maniskill_h5(tmp_path):
    path = tmp_path / "demo_state.h5"
    with h5py.File(path, "w") as f:
        for traj_idx in range(2):
            group = f.create_group(f"traj_{traj_idx}")
            group.create_dataset(
                "obs", data=np.ones((3, 4), dtype=np.float32) * traj_idx
            )
            group.create_dataset("actions", data=np.ones((2, 2), dtype=np.float32))
            group.create_dataset("rewards", data=np.ones(2, dtype=np.float32))
            group.create_dataset("terminated", data=np.array([False, True]))
            group.create_dataset("truncated", data=np.array([False, False]))

    agent = _agent()
    loaded = agent.load_offline_replay_buffer(
        path, source="maniskill_h5", buffer_size=16, offline_data_ratio=0.5
    )

    assert loaded == 4
    assert agent.offline_replay_buffer is not None
    assert len(agent.offline_replay_buffer) == 4
    assert agent.offline_data_ratio == 0.5


class _FakeEpisodeData:
    def __init__(self, observations, actions, rewards, terminations, truncations):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations


class _FakeMinariDataset:
    def __init__(self, episodes, observation_space, action_space):
        self._episodes = episodes
        self.observation_space = observation_space
        self.action_space = action_space
        self.total_episodes = len(episodes)

    def iterate_episodes(self, episode_indices=None):
        indices = range(len(self._episodes)) if episode_indices is None else episode_indices
        for idx in indices:
            yield self._episodes[idx]


def test_load_offline_replay_buffer_minari(monkeypatch):
    observations = np.arange(0.0, 4.0, dtype=np.float32).reshape(4, 1)
    episode = _FakeEpisodeData(
        observations=np.tile(observations, (1, 4)),
        actions=np.ones((3, 2), dtype=np.float32),
        rewards=np.ones(3, dtype=np.float32),
        terminations=np.array([False, False, True]),
        truncations=np.array([False, False, False]),
    )
    dataset = _FakeMinariDataset(
        [episode],
        observation_space=spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )
    fake_module = types.SimpleNamespace(load_dataset=lambda dataset_id, download=True: dataset)
    monkeypatch.setitem(sys.modules, "minari", fake_module)

    agent = _agent()
    loaded = agent.load_offline_replay_buffer(
        "fake/dataset-v0", source="minari", buffer_size=16, offline_data_ratio=0.3
    )

    assert loaded == 3
    assert len(agent.offline_replay_buffer) == 3
    assert agent.offline_data_ratio == 0.3
