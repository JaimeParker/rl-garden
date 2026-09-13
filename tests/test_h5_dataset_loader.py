"""Tests for loading trajectory H5 datasets into replay buffers."""

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    MCReplayBuffer,
    load_h5_dataset_to_replay_buffer,
)
from rl_garden.buffers.h5_dataset import infer_specs_from_h5
from rl_garden.observations.schema import ObservationContractError, validate_observation_space


def test_load_state_h5_to_dict_replay_buffer(tmp_path):
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

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)}),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 4
    assert len(buffer) == 4
    sample = buffer.sample(4)
    assert sample.obs["state"].shape == (4, 4)
    assert sample.next_obs["state"].shape == (4, 4)
    assert sample.actions.shape == (4, 2)
    assert sample.mc_returns.shape == (4,)
    assert torch.all(sample.rewards == 1.0)


def test_load_dict_h5_to_dict_replay_buffer(tmp_path):
    path = tmp_path / "demo_dict.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.ones((5, 3), dtype=np.float32))
        obs.create_dataset("rgb_front", data=np.ones((5, 8, 8, 3), dtype=np.uint8))
        group.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        group.create_dataset("dones", data=np.array([False, False, False, True]))

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict(
            {
                "state": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                "rgb_front": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 4
    sample = buffer.sample(4)
    assert sample.obs["state"].shape == (4, 3)
    assert sample.obs["rgb_front"].shape == (4, 8, 8, 3)
    assert sample.obs["rgb_front"].dtype == torch.uint8


def test_load_dict_h5_with_extra_state_group_to_dict_replay_buffer(tmp_path):
    """A state_<name> group key (Section A's extra_state family) is accepted
    as-is by the generic H5 loader, alongside "state"."""
    path = tmp_path / "demo_extra_state.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.ones((5, 3), dtype=np.float32))
        obs.create_dataset("state_object_pose", data=np.ones((5, 2), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        group.create_dataset("dones", data=np.array([False, False, False, True]))

    obs_space = spaces.Dict(
        {
            "state": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            "state_object_pose": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        }
    )
    inferred_obs_space, _ = infer_specs_from_h5(path)
    assert set(inferred_obs_space.spaces.keys()) == {"state", "state_object_pose"}
    validate_observation_space(inferred_obs_space)

    buffer = MCReplayBuffer(
        observation_space=obs_space,
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 4
    sample = buffer.sample(4)
    assert sample.obs["state"].shape == (4, 3)
    assert sample.obs["state_object_pose"].shape == (4, 2)


def test_load_state_only_h5_to_dict_replay_buffer_wraps_as_state_key(tmp_path):
    """A flat (non-Dict) H5 obs dataset loaded into a Dict-observation buffer
    (the registry-driven flow: infer_specs_from_h5 -> Dict({"state": ...}) ->
    a Dict buffer) must be wrapped as {"state": obs}, not passed through as a
    bare tensor."""
    path = tmp_path / "demo_state.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.ones((3, 4), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((2, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(2, dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, True]))
        group.create_dataset("truncated", data=np.array([False, False]))

    obs_space, action_space = infer_specs_from_h5(path)
    assert isinstance(obs_space, spaces.Dict)
    assert set(obs_space.spaces) == {"state"}

    buffer = MCReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 2
    sample = buffer.sample(2)
    assert sample.obs["state"].shape == (2, 4)


def test_infer_specs_from_h5_state_only_dataset_is_dict_with_state_key(tmp_path):
    path = tmp_path / "demo_state.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.ones((3, 4), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((2, 2), dtype=np.float32))

    obs_space, _ = infer_specs_from_h5(path)
    assert isinstance(obs_space, spaces.Dict)
    assert set(obs_space.spaces) == {"state"}
    assert obs_space["state"].shape == (4,)
    assert obs_space["state"].dtype == np.float32
    validate_observation_space(obs_space)


def test_infer_specs_from_h5_rejects_non_contract_obs_group_keys(tmp_path):
    path = tmp_path / "demo_bad_keys.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.ones((3, 4), dtype=np.float32))
        obs.create_dataset("extra", data=np.ones((3, 2), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ObservationContractError, match="extra"):
        infer_specs_from_h5(path)


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

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)}),
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=2,
        buffer_size=8,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 6
    expected = torch.tensor(
        [
            [2.71, 1.9],
            [1.0, 5.42],
            [3.8, 2.0],
        ]
    )
    # Loader-provided MC values are stored as _external_mc (trusted as-is,
    # not derived from _build_mc_table()'s own recursion) -- see mc_buffer.py.
    assert torch.allclose(buffer._external_mc[:3], expected)
    assert buffer._externally_valid[:3].all()


def test_loader_infers_success_when_sparse_reward_mc_enabled(tmp_path):
    path = tmp_path / "sparse_reward_mc.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.ones((3, 4), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((2, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.array([0.0, 1.0], dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, True]))
        group.create_dataset("truncated", data=np.array([False, False]))

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)}),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        success_threshold=0.5,
    )

    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 2


def test_loader_uses_explicit_episode_end_separately_from_terminal(tmp_path):
    path = tmp_path / "binary.h5"
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=np.zeros((3, 2), dtype=np.float32))
        traj.create_dataset("next_obs", data=np.ones((3, 2), dtype=np.float32))
        traj.create_dataset("actions", data=np.zeros((3, 1), dtype=np.float32))
        traj.create_dataset("rewards", data=np.array([-1, 0, 0], dtype=np.float32))
        traj.create_dataset("terminated", data=np.array([0, 1, 1], dtype=np.bool_))
        traj.create_dataset("episode_end", data=np.array([0, 0, 1], dtype=np.bool_))

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)}),
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=4,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    loaded = load_h5_dataset_to_replay_buffer(buffer, path)

    assert loaded == 3
    assert buffer._episode_end[:3, 0].tolist() == [False, False, True]


def test_sparse_binary_mc_stops_at_every_success_boundary(tmp_path):
    path = tmp_path / "binary_success_boundaries.h5"
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=np.zeros((3, 2), dtype=np.float32))
        traj.create_dataset("next_obs", data=np.ones((3, 2), dtype=np.float32))
        traj.create_dataset("actions", data=np.zeros((3, 1), dtype=np.float32))
        traj.create_dataset("rewards", data=np.array([-1, 0, 0], dtype=np.float32))
        traj.create_dataset("terminated", data=np.array([0, 1, 1], dtype=np.bool_))
        traj.create_dataset("success", data=np.array([0, 1, 1], dtype=np.bool_))
        traj.create_dataset("episode_end", data=np.array([0, 1, 1], dtype=np.bool_))

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)}),
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=4,
        gamma=0.99,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        sparse_negative_reward=-5.0,
        success_threshold=0.5,
    )
    loaded = load_h5_dataset_to_replay_buffer(
        buffer,
        path,
        reward_scale=10.0,
        reward_bias=5.0,
    )

    assert loaded == 3
    assert torch.allclose(
        buffer._build_mc_table()[:3, 0],
        torch.tensor([-0.05, 5.0, 5.0]),
    )
