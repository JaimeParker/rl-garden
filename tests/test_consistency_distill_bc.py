from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import ConsistencyDistillBC, DiffusionBC, OfflineEnvSpec
from rl_garden.networks import DiffusionUNet1D


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


def _env_spec(obs_dim: int, action_dim: int) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
    )


def _train_teacher_and_save(tmp_path, *, obs_dim, action_dim, horizon_steps=2, cond_steps=1, mlp_dims=(32, 32, 32)):
    path = tmp_path / "bc_dataset.h5"
    _write_h5_dataset(path, num_traj=8, steps_per_traj=20, obs_dim=obs_dim, action_dim=action_dim)

    teacher = DiffusionBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        denoising_steps=10,
        mlp_dims=list(mlp_dims),
        batch_size=16,
        actor_lr=1e-3,
        device="cpu",
    )
    teacher.train(10)
    ckpt_path = teacher.save(tmp_path / "teacher.pt")
    return str(path), str(ckpt_path)


def test_teacher_frozen_student_trains_target_ema_moves(tmp_path):
    obs_dim, action_dim = 4, 2
    dataset_path, ckpt_path = _train_teacher_and_save(tmp_path, obs_dim=obs_dim, action_dim=action_dim)

    agent = ConsistencyDistillBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=dataset_path,
        bc_checkpoint=ckpt_path,
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        cm_lr=1e-3,
        cm_ema_decay=0.5,
        device="cpu",
    )

    for p in agent.policy.parameters():
        assert not p.requires_grad

    teacher_before = {k: v.clone() for k, v in agent.policy.net.state_dict().items()}
    student_before = {k: v.clone() for k, v in agent.cm_student.state_dict().items()}
    target_before = {k: v.clone() for k, v in agent.cm_target.state_dict().items()}

    metrics = agent.train(5)
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    for k, v in agent.policy.net.state_dict().items():
        assert torch.allclose(v, teacher_before[k]), f"teacher param {k} changed"

    student_changed = any(
        not torch.allclose(v, student_before[k]) for k, v in agent.cm_student.state_dict().items()
    )
    assert student_changed

    target_changed = any(
        not torch.allclose(v, target_before[k]) for k, v in agent.cm_target.state_dict().items()
    )
    assert target_changed
    for k, v in agent.cm_target.state_dict().items():
        assert not torch.allclose(v, agent.cm_student.state_dict()[k]), (
            "target should not exactly equal student after a partial EMA step"
        )


def test_checkpoint_roundtrips(tmp_path):
    obs_dim, action_dim = 4, 2
    dataset_path, ckpt_path = _train_teacher_and_save(tmp_path, obs_dim=obs_dim, action_dim=action_dim)

    agent = ConsistencyDistillBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=dataset_path,
        bc_checkpoint=ckpt_path,
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        device="cpu",
    )
    agent.train(5)
    saved_path = agent.save(tmp_path / "cm.pt")
    student_state_before = {k: v.clone() for k, v in agent.cm_student.state_dict().items()}
    target_state_before = {k: v.clone() for k, v in agent.cm_target.state_dict().items()}

    agent2 = ConsistencyDistillBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=dataset_path,
        bc_checkpoint=ckpt_path,
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        device="cpu",
    )
    agent2.load(saved_path)
    for k, v in agent2.cm_student.state_dict().items():
        assert torch.allclose(v, student_state_before[k])
    for k, v in agent2.cm_target.state_dict().items():
        assert torch.allclose(v, target_state_before[k])


def test_net_cls_mismatch_against_teacher_checkpoint_fails_loudly(tmp_path):
    obs_dim, action_dim = 4, 2
    dataset_path, ckpt_path = _train_teacher_and_save(
        tmp_path, obs_dim=obs_dim, action_dim=action_dim, horizon_steps=4
    )

    with pytest.raises(ValueError, match="net_cls"):
        ConsistencyDistillBC(
            env=_env_spec(obs_dim, action_dim),
            dataset_path=dataset_path,
            bc_checkpoint=ckpt_path,
            horizon_steps=4,
            cond_steps=1,
            denoising_steps=10,
            net_cls=DiffusionUNet1D,
            net_kwargs=dict(down_dims=(8, 16), kernel_size=3, n_groups=4),
            batch_size=16,
            device="cpu",
        )


def test_denoising_steps_mismatch_against_teacher_checkpoint_raises(tmp_path):
    obs_dim, action_dim = 4, 2
    dataset_path, ckpt_path = _train_teacher_and_save(tmp_path, obs_dim=obs_dim, action_dim=action_dim)

    with pytest.raises(ValueError, match="denoising_steps"):
        ConsistencyDistillBC(
            env=_env_spec(obs_dim, action_dim),
            dataset_path=dataset_path,
            bc_checkpoint=ckpt_path,
            horizon_steps=2,
            cond_steps=1,
            denoising_steps=25,  # teacher was trained with denoising_steps=10
            mlp_dims=[32, 32, 32],
            batch_size=16,
            device="cpu",
        )


def test_activation_fn_mismatch_against_teacher_checkpoint_raises(tmp_path):
    obs_dim, action_dim = 4, 2
    dataset_path, ckpt_path = _train_teacher_and_save(tmp_path, obs_dim=obs_dim, action_dim=action_dim)

    with pytest.raises(ValueError, match="activation_fn"):
        ConsistencyDistillBC(
            env=_env_spec(obs_dim, action_dim),
            dataset_path=dataset_path,
            bc_checkpoint=ckpt_path,
            horizon_steps=2,
            cond_steps=1,
            denoising_steps=10,
            mlp_dims=[32, 32, 32],
            activation_fn="mish",  # teacher's default is relu
            batch_size=16,
            device="cpu",
        )


def test_checkpoint_missing_net_cls_metadata_does_not_force_mismatch(tmp_path):
    """Simulates a `diffusion_bc` checkpoint saved before `net_cls` existed
    in `_checkpoint_metadata` -- must not be treated as a forced mismatch."""
    obs_dim, action_dim = 4, 2
    dataset_path, ckpt_path = _train_teacher_and_save(tmp_path, obs_dim=obs_dim, action_dim=action_dim)

    raw = torch.load(ckpt_path, map_location="cpu")
    del raw["metadata"]["hyperparameters"]["net_cls"]
    del raw["metadata"]["hyperparameters"]["net_kwargs"]
    torch.save(raw, ckpt_path)

    ConsistencyDistillBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=dataset_path,
        bc_checkpoint=ckpt_path,
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        device="cpu",
    )
