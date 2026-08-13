from __future__ import annotations

import hashlib

import h5py
import numpy as np

from tools.conversion.convert_calql_adroit_binary import convert_adroit_binary_dataset


def _save_object_array(path, trajectories):
    values = np.empty(len(trajectories), dtype=object)
    values[:] = trajectories
    np.save(path, values, allow_pickle=True)


def test_converter_matches_calql_filtering_and_truncation(tmp_path):
    expert = tmp_path / "pen2_sparse.npy"
    bc = tmp_path / "pen_bc_sparse4.npy"
    output = tmp_path / "pen-binary-v0.h5"
    expert_obs = [
        {"state_observation": np.array([i, i + 1], dtype=np.float32)}
        for i in range(4)
    ]
    _save_object_array(
        expert,
        [
            {
                "observations": expert_obs,
                "next_observations": expert_obs,
                "actions": np.full((4, 1), 2.0, dtype=np.float32),
                "rewards": np.array([-1, 0, 0, -1], dtype=np.float32),
                "terminals": np.zeros(4, dtype=np.float32),
            },
            {
                "observations": expert_obs,
                "next_observations": expert_obs,
                "actions": np.zeros((4, 1), dtype=np.float32),
                "rewards": -np.ones(4, dtype=np.float32),
                "terminals": np.zeros(4, dtype=np.float32),
            },
        ],
    )
    _save_object_array(
        bc,
        [
            {
                "observations": np.arange(6, dtype=np.float32).reshape(3, 2),
                "next_observations": np.arange(6, dtype=np.float32).reshape(3, 2),
                "actions": np.full((3, 1), -2.0, dtype=np.float32),
                "rewards": np.array([[-1], [0], [-1]], dtype=np.float32),
                "terminals": np.zeros((3, 1), dtype=np.float32),
            }
        ],
    )

    count = convert_adroit_binary_dataset(
        task="pen-binary-v0",
        expert_path=expert,
        bc_path=bc,
        output_path=output,
    )

    assert count == 2
    with h5py.File(output, "r") as f:
        assert f.attrs["schema"] == "rl_garden.trajectory_h5.v1"
        assert f.attrs["task"] == "pen-binary-v0"
        assert f.attrs["expert_sha256"] == hashlib.sha256(expert.read_bytes()).hexdigest()
        assert sorted(f.keys()) == ["traj_0", "traj_1"]
        np.testing.assert_allclose(f["traj_0/actions"][:], 0.99999)
        assert f["traj_0/rewards"][:].tolist() == [-1.0, 0.0, 0.0]
        assert f["traj_0/terminated"][:].tolist() == [False, True, True]
        assert f["traj_0/episode_end"][:].tolist() == [False, False, True]
        assert f["traj_0"].attrs["source_kind"] == "expert"
        assert f["traj_0"].attrs["source_index"] == 0
        assert f["traj_1"].attrs["source_kind"] == "bc"
