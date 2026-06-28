"""Tests for loading ManiSkill H5 trajectories into replay buffers."""

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    MCDictReplayBuffer,
    MCTensorReplayBuffer,
    ResidualTensorReplayBuffer,
    load_maniskill_h5_to_replay_buffer,
    load_residual_h5_to_replay_buffer,
)


def test_load_state_h5_to_tensor_replay_buffer(tmp_path):
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

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_maniskill_h5_to_replay_buffer(buffer, path)
    assert loaded == 4
    assert len(buffer) == 4
    sample = buffer.sample(4)
    assert sample.obs.shape == (4, 4)
    assert sample.next_obs.shape == (4, 4)
    assert sample.actions.shape == (4, 2)
    assert sample.mc_returns.shape == (4,)
    assert torch.all(sample.rewards == 1.0)


def test_load_dict_h5_to_dict_replay_buffer(tmp_path):
    path = tmp_path / "demo_dict.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.ones((5, 3), dtype=np.float32))
        obs.create_dataset("rgb", data=np.ones((5, 8, 8, 3), dtype=np.uint8))
        group.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        group.create_dataset("dones", data=np.array([False, False, False, True]))

    buffer = MCDictReplayBuffer(
        observation_space=spaces.Dict(
            {
                "state": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                "rgb": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_maniskill_h5_to_replay_buffer(buffer, path)
    assert loaded == 4
    sample = buffer.sample(4)
    assert sample.obs["state"].shape == (4, 3)
    assert sample.obs["rgb"].shape == (4, 8, 8, 3)
    assert sample.obs["rgb"].dtype == torch.uint8


def test_loader_preserves_mc_returns_from_trajectory_boundaries(tmp_path):
    path = tmp_path / "variable_lengths.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((3, 1), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(3, dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, False, True]))
        group.create_dataset("truncated", data=np.array([False, False, False]))

        group = f.create_group("traj_1")
        group.create_dataset("obs", data=np.ones((4, 2), dtype=np.float32) * 2)
        group.create_dataset("actions", data=np.ones((3, 1), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(3, dtype=np.float32) * 2)
        group.create_dataset("terminated", data=np.array([False, False, True]))
        group.create_dataset("truncated", data=np.array([False, False, False]))

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=2,
        buffer_size=8,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_maniskill_h5_to_replay_buffer(buffer, path)
    assert loaded == 6
    expected = torch.tensor(
        [
            [2.71, 1.9],
            [1.0, 5.42],
            [3.8, 2.0],
        ]
    )
    assert torch.allclose(buffer._mc_table[:3], expected)


def test_load_residual_h5_to_residual_replay_buffer(tmp_path):
    path = tmp_path / "residual_demo.h5"
    actions = np.arange(8, dtype=np.float32).reshape(4, 2) / 10.0
    base_actions = np.ones((4, 2), dtype=np.float32) * 0.25
    next_base_actions = np.ones((4, 2), dtype=np.float32) * -0.25
    with h5py.File(path, "w") as f:
        f.attrs["dataset_type"] = "rl_garden_residual_offline"
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.arange(15, dtype=np.float32).reshape(5, 3))
        group.create_dataset("actions", data=actions)
        group.create_dataset("base_actions", data=base_actions)
        group.create_dataset("next_base_actions", data=next_base_actions)
        group.create_dataset("rewards", data=np.arange(4, dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, False, False, True]))
        group.create_dataset("truncated", data=np.array([False, False, False, False]))

    buffer = ResidualTensorReplayBuffer(
        observation_space=spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=8,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_residual_h5_to_replay_buffer(buffer, path, bootstrap_at_done="never")

    assert loaded == 4
    assert len(buffer) == 4
    torch.testing.assert_close(
        buffer.actions[:2].reshape(-1, 2), torch.as_tensor(actions)
    )
    torch.testing.assert_close(
        buffer.base_actions[:2].reshape(-1, 2), torch.as_tensor(base_actions)
    )
    torch.testing.assert_close(
        buffer.next_base_actions[:2].reshape(-1, 2),
        torch.as_tensor(next_base_actions),
    )
    torch.testing.assert_close(
        buffer.dones[:2].reshape(-1), torch.tensor([0, 0, 0, 1.0])
    )
