"""Tests for loading OGBench offline datasets into replay buffers."""

import sys
import types

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.buffers import (
    MCTensorReplayBuffer,
    TensorReplayBuffer,
    infer_specs_from_ogbench,
    load_ogbench_dataset_to_replay_buffer,
)

_OBS_DIM = 2
_ACTION_DIM = 2


class _FakeOGBenchEnv:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    def close(self):
        self.closed = True


def _singletask_dataset(*, traj1_success=True, traj2_success=False):
    # traj1: length 3, success at its last step -> masks[2] = 0.
    # traj2: length 2, never succeeds -> masks all 1.
    masks1 = [1.0, 1.0, 0.0 if traj1_success else 1.0]
    masks2 = [1.0, 0.0 if traj2_success else 1.0]
    rewards1 = [-1.0, -1.0, 0.0 if traj1_success else -1.0]
    rewards2 = [-1.0, 0.0 if traj2_success else -1.0]
    return {
        "observations": np.arange(5 * _OBS_DIM, dtype=np.float32).reshape(5, _OBS_DIM),
        "next_observations": np.arange(5 * _OBS_DIM, dtype=np.float32).reshape(5, _OBS_DIM) + 1,
        "actions": np.zeros((5, _ACTION_DIM), dtype=np.float32),
        "terminals": np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "rewards": np.array(rewards1 + rewards2, dtype=np.float32),
        "masks": np.array(masks1 + masks2, dtype=np.float32),
    }


def _goal_conditioned_dataset():
    return {
        "observations": np.zeros((5, _OBS_DIM), dtype=np.float32),
        "next_observations": np.zeros((5, _OBS_DIM), dtype=np.float32),
        "actions": np.zeros((5, _ACTION_DIM), dtype=np.float32),
        "terminals": np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }


def _install_fake_ogbench(monkeypatch, train_dataset, *, obs_space=None, action_space=None):
    obs_space = obs_space or spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float64)
    action_space = action_space or spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)
    captured = {}

    def _make_env_and_datasets(dataset_name, dataset_dir=None, env_only=False):
        captured["dataset_name"] = dataset_name
        env = _FakeOGBenchEnv(obs_space, action_space)
        if env_only:
            return env
        return env, train_dataset, {}

    fake_module = types.SimpleNamespace(make_env_and_datasets=_make_env_and_datasets)
    monkeypatch.setitem(sys.modules, "ogbench", fake_module)
    return captured


def test_infer_specs_from_ogbench_canonicalizes_float_obs(monkeypatch):
    _install_fake_ogbench(monkeypatch, _singletask_dataset())

    obs_space, action_space = infer_specs_from_ogbench("antmaze-large-navigate-singletask-v0")

    assert isinstance(obs_space, spaces.Box)
    assert obs_space.dtype == np.float32
    assert obs_space.shape == (_OBS_DIM,)
    assert action_space.shape == (_ACTION_DIM,)


def test_infer_specs_from_ogbench_passes_through_uint8_image_space(monkeypatch):
    image_space = spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)
    _install_fake_ogbench(monkeypatch, _singletask_dataset(), obs_space=image_space)

    obs_space, _ = infer_specs_from_ogbench("visual-antmaze-large-navigate-singletask-v0")

    assert obs_space.dtype == np.uint8
    assert obs_space.shape == (4, 4, 3)


def test_load_ogbench_dataset_maps_masks_to_dones_and_terminals_to_episode_ends(monkeypatch):
    _install_fake_ogbench(monkeypatch, _singletask_dataset())

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_ogbench_dataset_to_replay_buffer(buffer, "antmaze-large-navigate-singletask-v0")

    assert loaded == 5
    # done = 1 - masks: only traj1's last step (index 2) is a bootstrap-cut.
    # Raw (T, N) storage, not a shuffled .sample(), so insertion order is preserved.
    assert buffer.dones[:5, 0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_load_ogbench_dataset_episode_ends_track_terminals(monkeypatch):
    _install_fake_ogbench(monkeypatch, _singletask_dataset())

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_ogbench_dataset_to_replay_buffer(buffer, "antmaze-large-navigate-singletask-v0")

    assert loaded == 5
    # episode_end = terminals, independent of task-completion masks.
    assert buffer._episode_end[:5, 0].tolist() == [False, False, True, False, True]


def test_load_ogbench_dataset_num_traj_truncates_at_trajectory_boundary(monkeypatch):
    _install_fake_ogbench(monkeypatch, _singletask_dataset())

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_ogbench_dataset_to_replay_buffer(
        buffer, "antmaze-large-navigate-singletask-v0", num_traj=1
    )

    assert loaded == 3  # traj1 only (3 steps), not all 5.


def test_load_ogbench_dataset_rejects_goal_conditioned_dataset(monkeypatch):
    _install_fake_ogbench(monkeypatch, _goal_conditioned_dataset())

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    with pytest.raises(ValueError, match="rewards.*masks"):
        load_ogbench_dataset_to_replay_buffer(buffer, "antmaze-large-navigate-v0")


def test_load_ogbench_dataset_closes_throwaway_env(monkeypatch):
    captured = {}
    train_dataset = _singletask_dataset()

    def _make_env_and_datasets(dataset_name, dataset_dir=None, env_only=False):
        env = _FakeOGBenchEnv(
            spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float64),
            spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        )
        captured["env"] = env
        return env, train_dataset, {}

    monkeypatch.setitem(
        sys.modules, "ogbench", types.SimpleNamespace(make_env_and_datasets=_make_env_and_datasets)
    )

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )
    load_ogbench_dataset_to_replay_buffer(buffer, "antmaze-large-navigate-singletask-v0")

    assert captured["env"].closed
