from __future__ import annotations

import h5py
import numpy as np
import torch

from rl_garden.buffers.windowed_trajectory_dataset import WindowedTrajectoryDataset


def _write_h5(path, num_traj: int = 4, length: int = 10) -> None:
    with h5py.File(path, "w") as f:
        for i in range(num_traj):
            g = f.create_group(f"traj_{i}")
            g.create_dataset(
                "obs", data=np.random.randn(length + 1, 4).astype(np.float32)
            )
            g.create_dataset(
                "actions", data=np.random.uniform(-1, 1, (length, 2)).astype(np.float32)
            )
            g.create_dataset("rewards", data=np.zeros(length, dtype=np.float32))
            terminated = np.zeros(length, dtype=bool)
            terminated[-1] = True
            g.create_dataset("terminated", data=terminated)
            g.create_dataset("truncated", data=np.zeros(length, dtype=bool))


def test_windows_have_exact_chunk_size(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    ds = WindowedTrajectoryDataset(path, chunk_size=4)
    sample = ds.sample(32)
    assert sample.obs_window.shape == (32, 4, 4)
    assert sample.action_window.shape == (32, 4, 2)


def test_obs_and_action_windows_are_aligned(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path, num_traj=1, length=10)
    ds = WindowedTrajectoryDataset(path, chunk_size=4)
    # Every valid start index is directly recoverable: obs_window[:, 0]
    # should equal ds._obs[start] for the corresponding start, and the
    # action_window should be the actions immediately following it. Since
    # start indices aren't returned directly, check internal consistency by
    # re-deriving the start from the raw obs array (obs are unique here).
    sample = ds.sample(16)
    for b in range(sample.obs_window.shape[0]):
        matches = (ds._obs == sample.obs_window[b, 0]).all(dim=-1).nonzero(as_tuple=True)[0]
        assert matches.numel() >= 1
        start = int(matches[0])
        assert torch.equal(sample.obs_window[b], ds._obs[start : start + 4])
        assert torch.equal(sample.action_window[b], ds._actions[start : start + 4])


def test_windows_never_cross_trajectory_boundary(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path, num_traj=3, length=5)
    ds = WindowedTrajectoryDataset(path, chunk_size=4)
    # Each trajectory has length=5, chunk_size=4 -> exactly 2 valid starts
    # per trajectory (0 and 1); with 3 trajectories that's 6 valid starts.
    assert ds.valid_starts.shape[0] == 6


def test_chunk_size_larger_than_all_trajectories_raises(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path, num_traj=2, length=3)
    try:
        WindowedTrajectoryDataset(path, chunk_size=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
