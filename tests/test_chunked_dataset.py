from __future__ import annotations

import h5py
import numpy as np
import torch

from rl_garden.buffers.chunked_dataset import load_h5_dataset_as_chunks


def _write_traj(group, *, num_steps: int) -> None:
    # obs[i] = i (as a 1-D feature), length num_steps + 1 so next_obs derives
    # by slicing; actions[i] = i * 10.
    group.create_dataset(
        "obs", data=np.arange(num_steps + 1, dtype=np.float32).reshape(-1, 1)
    )
    group.create_dataset(
        "actions", data=(np.arange(num_steps, dtype=np.float32) * 10).reshape(-1, 1)
    )
    group.create_dataset("rewards", data=np.zeros(num_steps, dtype=np.float32))
    dones = np.zeros(num_steps, dtype=np.float32)
    dones[-1] = 1.0
    group.create_dataset("dones", data=dones)


def test_windows_never_cross_trajectory_boundary_and_pad_history_by_repeat(tmp_path):
    path = tmp_path / "chunks.h5"
    with h5py.File(path, "w") as f:
        _write_traj(f.create_group("traj_0"), num_steps=5)
        # Too short for horizon_steps=2 -- must contribute zero windows.
        _write_traj(f.create_group("traj_1"), num_steps=1)

    obs_history, action_chunks = load_h5_dataset_as_chunks(
        path, horizon_steps=2, cond_steps=3
    )

    assert action_chunks.shape == (4, 2, 1)
    assert obs_history.shape == (4, 3, 1)

    expected_action_chunks = torch.tensor(
        [[[0.0], [10.0]], [[10.0], [20.0]], [[20.0], [30.0]], [[30.0], [40.0]]]
    )
    assert torch.equal(action_chunks, expected_action_chunks)

    expected_obs_history = torch.tensor(
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0]],
            [[0.0], [1.0], [2.0]],
            [[1.0], [2.0], [3.0]],
        ]
    )
    assert torch.equal(obs_history, expected_obs_history)


def test_dict_observations_supported(tmp_path):
    path = tmp_path / "chunks_dict.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("traj_0")
        obs = g.create_group("obs")
        obs.create_dataset("state", data=np.arange(6, dtype=np.float32).reshape(-1, 1))
        g.create_dataset("actions", data=np.zeros((5, 2), dtype=np.float32))
        g.create_dataset("rewards", data=np.zeros(5, dtype=np.float32))
        dones = np.zeros(5, dtype=np.float32)
        dones[-1] = 1.0
        g.create_dataset("dones", data=dones)

    obs_history, action_chunks = load_h5_dataset_as_chunks(
        path, horizon_steps=2, cond_steps=2
    )
    assert isinstance(obs_history, dict)
    assert obs_history["state"].shape == (4, 2, 1)
    assert action_chunks.shape == (4, 2, 2)


def test_no_trajectory_long_enough_raises(tmp_path):
    path = tmp_path / "chunks_short.h5"
    with h5py.File(path, "w") as f:
        _write_traj(f.create_group("traj_0"), num_steps=1)

    try:
        load_h5_dataset_as_chunks(path, horizon_steps=4, cond_steps=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
