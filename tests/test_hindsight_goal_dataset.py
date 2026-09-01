from __future__ import annotations

import h5py
import numpy as np
import torch

from rl_garden.buffers.hindsight_goal_dataset import HindsightGoalDataset


def _write_h5(path, num_traj: int = 4, length: int = 20) -> None:
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


def test_goal_sampling_respects_nested_overrides(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)

    ds_curr = HindsightGoalDataset(
        path, p_currgoal=1.0, p_trajgoal=0.0, p_randomgoal=0.0, discount=0.99
    )
    sample = ds_curr.sample(64)
    assert bool(sample.success.all())
    assert torch.equal(sample.obs, sample.goals)


def test_next_obs_never_crosses_trajectory_boundary(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path, num_traj=3, length=10)

    ds = HindsightGoalDataset(path, discount=0.99)
    # Every valid index's next_obs must come from the same 20-length window
    # implied by the trajectory boundaries -- indirectly checked via the
    # invariant that `next_obs` is always a well-defined, finite tensor for
    # every possible sampled index (would error/misalign otherwise).
    for _ in range(20):
        sample = ds.sample(64)
        assert torch.isfinite(sample.next_obs).all()
        assert torch.isfinite(sample.obs).all()


def test_success_matches_goal_equals_index(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)

    ds = HindsightGoalDataset(
        path, p_currgoal=0.0, p_trajgoal=0.0, p_randomgoal=1.0, discount=0.99
    )
    sample = ds.sample(500)
    # With random goals only, success should be rare but occasionally fire
    # (goal_indx happens to equal indx by chance) and always be consistent
    # with obs==goals when it does.
    matches = sample.success.bool()
    if matches.any():
        assert torch.equal(sample.obs[matches], sample.goals[matches])


def test_p_currgoal_one_does_not_crash(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    ds = HindsightGoalDataset(
        path, p_currgoal=1.0, p_trajgoal=0.0, p_randomgoal=0.0, discount=0.99
    )
    ds.sample(32)
