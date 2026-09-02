from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.d4rl_legacy_dataset import (
    _official_calql_antmaze_dataset,
    _official_calql_kitchen_dataset,
    _standard_d4rl_dataset,
    infer_specs_from_d4rl_legacy,
    load_d4rl_legacy_dataset_to_replay_buffer,
)
from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer


class _FakeD4RLEnv:
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._max_episode_steps = 3
        self.spec = SimpleNamespace(name="antmaze-test-v2", max_episode_steps=3)
        self.closed = False

    def get_dataset(self):
        n = 7
        observations = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        return {
            "observations": observations,
            "next_observations": observations + 0.5,
            "actions": np.array(
                [
                    [1.2, -1.2],
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                    [0.9, 1.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "rewards": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
            "terminals": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
            "timeouts": np.array([0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        }

    def close(self):
        self.closed = True


def test_official_dataset_drops_timeouts_clips_actions_and_computes_sparse_mc():
    dataset = _official_calql_antmaze_dataset(
        _FakeD4RLEnv(),
        reward_scale=10.0,
        reward_bias=-5.0,
        clip_action=0.99999,
        gamma=0.99,
    )

    assert dataset["observations"].shape == (5, 4)
    assert dataset["rewards"].tolist() == [-5.0, -5.0, -5.0, -5.0, 5.0]
    assert dataset["terminals"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert np.max(np.abs(dataset["actions"])) <= 0.99999
    np.testing.assert_allclose(
        dataset["mc_returns"],
        [-500.0, -500.0, -5.0495, -0.05, 5.0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_legacy_loader_populates_replay_mc_table(monkeypatch):
    env = _FakeD4RLEnv()
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env", lambda env_id: env
    )
    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.99,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        sparse_negative_reward=-5.0,
        success_threshold=0.5,
    )

    loaded = load_d4rl_legacy_dataset_to_replay_buffer(
        buffer,
        "antmaze-test-v2",
        reward_scale=10.0,
        reward_bias=-5.0,
    )

    assert loaded == 5
    assert env.closed
    # Loader-provided MC values are stored as _external_mc (trusted as-is,
    # not derived from _build_mc_table()'s own recursion) -- see mc_buffer.py.
    assert buffer._externally_valid[:5, 0].all()
    torch.testing.assert_close(
        buffer._external_mc[:5, 0],
        torch.tensor([-500.0, -500.0, -5.0495, -0.05, 5.0]),
    )


def test_legacy_loader_populates_plain_replay_without_gamma_attr(monkeypatch):
    env = _FakeD4RLEnv()
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env", lambda env_id: env
    )
    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_d4rl_legacy_dataset_to_replay_buffer(
        buffer,
        "antmaze-test-v2",
        reward_scale=1.0,
        reward_bias=-1.0,
    )

    assert loaded == 5
    assert env.closed
    assert len(buffer) == 5
    assert not hasattr(buffer, "gamma")


def test_infer_specs_canonicalizes_observations_to_float32(monkeypatch):
    env = _FakeD4RLEnv()
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env", lambda env_id: env
    )

    obs_space, action_space = infer_specs_from_d4rl_legacy("antmaze-test-v2")

    assert obs_space.shape == (4,)
    assert obs_space.dtype == np.float32
    assert action_space.shape == (2,)
    assert env.closed


class _FakeDenseD4RLEnv(_FakeD4RLEnv):
    def __init__(self, name):
        super().__init__()
        self.spec = SimpleNamespace(name=name, max_episode_steps=3)

    def get_dataset(self):
        observations = np.arange(24, dtype=np.float32).reshape(6, 4)
        return {
            "observations": observations,
            "next_observations": observations + 0.5,
            "actions": np.full((6, 2), 1.5, dtype=np.float32),
            "rewards": np.array([1, 2, 99, 3, 4, 99], dtype=np.float32),
            "terminals": np.zeros(6, dtype=np.float32),
            "timeouts": np.array([0, 0, 1, 0, 0, 1], dtype=np.float32),
        }


def test_standard_adroit_uses_finite_episode_returns():
    dataset = _standard_d4rl_dataset(
        _FakeDenseD4RLEnv("pen-human-v1"),
        reward_scale=1.0,
        reward_bias=0.0,
        clip_action=0.99999,
        gamma=0.5,
    )

    assert dataset["rewards"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert dataset["episode_end"].tolist() == [0.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(dataset["mc_returns"], [2.0, 2.0, 5.0, 4.0])
    assert np.max(np.abs(dataset["actions"])) <= 0.99999


def test_kitchen_uses_infinite_horizon_tail_returns():
    dataset = _official_calql_kitchen_dataset(
        _FakeDenseD4RLEnv("kitchen-partial-v0"),
        reward_scale=1.0,
        reward_bias=0.0,
        clip_action=0.99999,
        gamma=0.5,
    )

    assert dataset["episode_end"].tolist() == [0.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(dataset["mc_returns"], [3.0, 4.0, 7.0, 8.0])


def test_legacy_loader_dispatches_supported_task_families(monkeypatch):
    env = _FakeDenseD4RLEnv("pen-human-v1")
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env", lambda env_id: env
    )
    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.5,
        storage_device="cpu",
        sample_device="cpu",
    )

    assert load_d4rl_legacy_dataset_to_replay_buffer(buffer, "pen-human-v1") == 4


def test_legacy_loader_dispatches_locomotion_family(monkeypatch):
    env = _FakeDenseD4RLEnv("halfcheetah-medium-v2")
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env", lambda env_id: env
    )
    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.5,
        storage_device="cpu",
        sample_device="cpu",
    )

    with pytest.warns(RuntimeWarning, match="finite-horizon backward sum"):
        assert (
            load_d4rl_legacy_dataset_to_replay_buffer(buffer, "halfcheetah-medium-v2")
            == 4
        )


def test_legacy_loader_rejects_unknown_family(monkeypatch):
    env = _FakeDenseD4RLEnv("totally-fake-env-v2")
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env", lambda env_id: env
    )
    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    with np.testing.assert_raises_regex(
        NotImplementedError, "AntMaze, Adroit, Kitchen, and Locomotion"
    ):
        load_d4rl_legacy_dataset_to_replay_buffer(buffer, "totally-fake-env-v2")
