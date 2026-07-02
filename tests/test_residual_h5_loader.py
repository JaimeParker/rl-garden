"""Tests for loading residual-offline H5 trajectories into replay buffers."""

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    ResidualDictReplayBuffer,
    ResidualTensorReplayBuffer,
    load_residual_h5_to_replay_buffer,
)


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


def test_load_residual_dict_h5_to_residual_dict_replay_buffer(tmp_path):
    path = tmp_path / "residual_rgb_demo.h5"
    with h5py.File(path, "w") as f:
        f.attrs["dataset_type"] = "rl_garden_residual_offline"
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.ones((5, 3), dtype=np.float32))
        obs.create_dataset(
            "rgb_base_camera", data=np.ones((5, 8, 8, 3), dtype=np.uint8)
        )
        group.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("base_actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset(
            "next_base_actions", data=np.ones((4, 2), dtype=np.float32)
        )
        group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, False, False, True]))
        group.create_dataset("truncated", data=np.array([False, False, False, False]))

    buffer = ResidualDictReplayBuffer(
        observation_space=spaces.Dict(
            {
                "state": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                "rgb_base_camera": spaces.Box(
                    low=0, high=255, shape=(8, 8, 3), dtype=np.uint8
                ),
            }
        ),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=1,
        buffer_size=4,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_residual_h5_to_replay_buffer(buffer, path, bootstrap_at_done="never")
    sample = buffer.sample(2)

    assert loaded == 4
    assert sample.obs["state"].shape == (2, 3)
    assert sample.obs["rgb_base_camera"].shape == (2, 8, 8, 3)
    assert sample.obs["rgb_base_camera"].dtype == torch.uint8
    assert sample.base_actions.shape == (2, 2)
