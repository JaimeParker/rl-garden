from __future__ import annotations

import h5py
import numpy as np


def _write_h5_dataset(path, *, num_traj: int, steps_per_traj: int, obs_dim: int, action_dim: int) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for traj_idx in range(num_traj):
            g = f.create_group(f"traj_{traj_idx}")
            g.create_dataset(
                "obs", data=rng.standard_normal((steps_per_traj + 1, obs_dim)).astype(np.float32)
            )
            g.create_dataset(
                "actions",
                data=(rng.random((steps_per_traj, action_dim)).astype(np.float32) * 2 - 1),
            )
            g.create_dataset("rewards", data=np.zeros(steps_per_traj, dtype=np.float32))
            dones = np.zeros(steps_per_traj, dtype=np.float32)
            dones[-1] = 1.0
            g.create_dataset("dones", data=dones)


def test_diffusion_bc_registered_and_dry_run_materializes_config(tmp_path):
    from rl_garden.training.offline import registry

    dataset_path = tmp_path / "bc.h5"
    _write_h5_dataset(dataset_path, num_traj=4, steps_per_traj=10, obs_dim=3, action_dim=2)

    registry.run_cli(
        [
            "diffusion_bc",
            "--dataset_path",
            str(dataset_path),
            "--horizon_steps",
            "2",
            "--cond_steps",
            "1",
            "--denoising_steps",
            "5",
            "--device",
            "cpu",
            "--log_dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )


def test_diffusion_bc_cli_trains_and_saves_checkpoint(tmp_path):
    from rl_garden.training.offline import registry

    dataset_path = tmp_path / "bc.h5"
    _write_h5_dataset(dataset_path, num_traj=4, steps_per_traj=10, obs_dim=3, action_dim=2)
    checkpoint_dir = tmp_path / "ckpt"

    registry.run_cli(
        [
            "diffusion_bc",
            "--dataset_path",
            str(dataset_path),
            "--horizon_steps",
            "2",
            "--cond_steps",
            "1",
            "--denoising_steps",
            "5",
            "--device",
            "cpu",
            "--num_offline_steps",
            "5",
            "--batch_size",
            "8",
            "--log_type",
            "none",
            "--log_dir",
            str(tmp_path / "runs"),
            "--checkpoint_dir",
            str(checkpoint_dir),
        ]
    )

    assert (checkpoint_dir / "diffusion_bc_offline_pretrained.pt").exists()
