from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.d4rl_legacy_dataset import (
    _official_calql_antmaze_dataset,
    infer_specs_from_d4rl_legacy,
    load_d4rl_legacy_dataset_to_replay_buffer,
)
from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer


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
    assert buffer._mc_table is not None
    torch.testing.assert_close(
        buffer._mc_table[:5, 0],
        torch.tensor([-500.0, -500.0, -5.0495, -0.05, 5.0]),
    )


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
