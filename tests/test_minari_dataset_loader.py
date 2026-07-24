"""Tests for loading Minari datasets into replay buffers."""
import sys
import types

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    MCTensorReplayBuffer,
    TensorReplayBuffer,
    infer_specs_from_minari,
    load_minari_dataset_to_replay_buffer,
)


class _FakeEpisodeData:
    def __init__(self, observations, actions, rewards, terminations, truncations, infos=None):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos


class _FakeMinariDataset:
    def __init__(self, episodes, observation_space, action_space):
        self._episodes = episodes
        self.observation_space = observation_space
        self.action_space = action_space
        self.total_episodes = len(episodes)
        self.total_steps = sum(len(ep.actions) for ep in episodes)

    def iterate_episodes(self, episode_indices=None):
        indices = range(len(self._episodes)) if episode_indices is None else episode_indices
        for idx in indices:
            yield self._episodes[idx]


def _install_fake_minari(monkeypatch, dataset: _FakeMinariDataset) -> None:
    fake_module = types.SimpleNamespace(load_dataset=lambda dataset_id, download=True: dataset)
    monkeypatch.setitem(sys.modules, "minari", fake_module)


def _make_episode(*, obs_start: float, true_terminal: bool) -> _FakeEpisodeData:
    # 3 transitions; observations has length steps+1 = 4.
    observations = np.arange(obs_start, obs_start + 4, dtype=np.float32).reshape(4, 1)
    actions = np.ones((3, 2), dtype=np.float32)
    rewards = np.ones(3, dtype=np.float32)
    terminations = np.array([False, False, true_terminal])
    # A timeout-truncated episode is truncated instead of terminated on its last step.
    truncations = np.array([False, False, not true_terminal])
    return _FakeEpisodeData(observations, actions, rewards, terminations, truncations)


def _make_box_dataset(episodes) -> _FakeMinariDataset:
    return _FakeMinariDataset(
        episodes,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    )


def test_infer_specs_canonicalizes_floating_observation_spaces(monkeypatch):
    dataset = _FakeMinariDataset(
        [],
        observation_space=spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
                ),
                "rgb": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
    )
    _install_fake_minari(monkeypatch, dataset)

    obs_space, action_space = infer_specs_from_minari("fake/dataset-v0")

    assert obs_space["observation"].dtype == np.float32
    assert obs_space["rgb"].dtype == np.uint8
    assert action_space.dtype == np.float64


def test_done_is_terminations_only_not_truncations(monkeypatch):
    terminal_episode = _make_episode(obs_start=0.0, true_terminal=True)
    timeout_episode = _make_episode(obs_start=10.0, true_terminal=False)
    dataset = _make_box_dataset([terminal_episode, timeout_episode])
    _install_fake_minari(monkeypatch, dataset)

    # num_envs=1 keeps buffer storage order == concatenation (insertion) order,
    # so positions can be read back deterministically.
    buffer = TensorReplayBuffer(
        observation_space=dataset.observation_space,
        action_space=dataset.action_space,
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_minari_dataset_to_replay_buffer(buffer, "fake/dataset-v0")
    assert loaded == 6
    dones = buffer.dones[:6, 0]
    # Only the last transition of the truly-terminated episode should have
    # done=1. The timeout-truncated episode must bootstrap through its
    # cutoff (done=0), even on its own last transition.
    assert torch.equal(dones, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))


def test_obs_next_obs_shift_by_one(monkeypatch):
    episode = _make_episode(obs_start=0.0, true_terminal=True)
    dataset = _make_box_dataset([episode])
    _install_fake_minari(monkeypatch, dataset)

    buffer = TensorReplayBuffer(
        observation_space=dataset.observation_space,
        action_space=dataset.action_space,
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_minari_dataset_to_replay_buffer(buffer, "fake/dataset-v0")
    assert loaded == 3
    assert torch.equal(buffer.obs[:3, 0].flatten(), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(buffer.next_obs[:3, 0].flatten(), torch.tensor([1.0, 2.0, 3.0]))


def test_reward_scale_and_bias_applied(monkeypatch):
    episode = _make_episode(obs_start=0.0, true_terminal=True)
    dataset = _make_box_dataset([episode])
    _install_fake_minari(monkeypatch, dataset)

    buffer = TensorReplayBuffer(
        observation_space=dataset.observation_space,
        action_space=dataset.action_space,
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    load_minari_dataset_to_replay_buffer(
        buffer, "fake/dataset-v0", reward_scale=2.0, reward_bias=-0.5
    )
    assert torch.allclose(buffer.rewards[:3, 0], torch.full((3,), 1.5))


def test_mc_table_populated_for_mc_buffer(monkeypatch):
    episode = _make_episode(obs_start=0.0, true_terminal=True)
    dataset = _make_box_dataset([episode])
    _install_fake_minari(monkeypatch, dataset)

    buffer = MCTensorReplayBuffer(
        observation_space=dataset.observation_space,
        action_space=dataset.action_space,
        num_envs=1,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    load_minari_dataset_to_replay_buffer(buffer, "fake/dataset-v0")
    assert buffer._mc_table is not None
    assert buffer._mc_table[:3, 0].tolist() == pytest.approx([2.71, 1.9, 1.0])


def test_num_episodes_caps_loaded_episodes(monkeypatch):
    episodes = [_make_episode(obs_start=float(i * 10), true_terminal=True) for i in range(3)]
    dataset = _make_box_dataset(episodes)
    _install_fake_minari(monkeypatch, dataset)

    buffer = TensorReplayBuffer(
        observation_space=dataset.observation_space,
        action_space=dataset.action_space,
        num_envs=1,
        buffer_size=20,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_minari_dataset_to_replay_buffer(buffer, "fake/dataset-v0", num_episodes=2)
    assert loaded == 6
