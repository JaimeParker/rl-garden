from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import DiffusionBC, OfflineEnvSpec


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


def test_diffusion_bc_loss_decreases_and_checkpoint_roundtrips(tmp_path):
    obs_dim, action_dim = 4, 2
    path = tmp_path / "bc_dataset.h5"
    _write_h5_dataset(path, num_traj=8, steps_per_traj=20, obs_dim=obs_dim, action_dim=action_dim)

    agent = DiffusionBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        actor_lr=1e-3,
        device="cpu",
    )

    early = agent.train(20)["loss"]
    late = agent.train(80)["loss"]
    assert late < early

    ckpt_path = agent.save(tmp_path / "diffusion_bc.pt")
    ema_state_before = {
        k: v.clone() for k, v in agent.ema_policy.net.state_dict().items()
    }

    agent2 = DiffusionBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        device="cpu",
    )
    agent2.load(ckpt_path)
    for k, v in agent2.ema_policy.net.state_dict().items():
        assert torch.allclose(v, ema_state_before[k])


def test_ema_start_step_defaults_to_dataset_scaled_warmup():
    obs_dim, action_dim = 4, 2
    path = ""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/bc_dataset.h5"
        _write_h5_dataset(path, num_traj=8, steps_per_traj=20, obs_dim=obs_dim, action_dim=action_dim)

        agent = DiffusionBC(
            env=_env_spec(obs_dim, action_dim),
            dataset_path=path,
            horizon_steps=2,
            cond_steps=1,
            denoising_steps=10,
            mlp_dims=[32, 32, 32],
            batch_size=16,
            device="cpu",
        )
        steps_per_epoch = agent._dataset_size // agent.batch_size
        assert agent.ema_start_step == 20 * steps_per_epoch
        assert agent.ema_start_step > 0

        agent_override = DiffusionBC(
            env=_env_spec(obs_dim, action_dim),
            dataset_path=path,
            horizon_steps=2,
            cond_steps=1,
            denoising_steps=10,
            mlp_dims=[32, 32, 32],
            batch_size=16,
            ema_start_step=0,
            device="cpu",
        )
        assert agent_override.ema_start_step == 0


def test_predict_returns_action_chunk_within_bounds():
    obs_dim, action_dim = 3, 2
    torch.manual_seed(0)
    from rl_garden.policies.diffusion_policy import DiffusionPolicy

    policy = DiffusionPolicy(
        observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        horizon_steps=3,
        cond_steps=2,
        denoising_steps=5,
        mlp_dims=[16, 16, 16],
    )
    obs = torch.randn(4, obs_dim)
    with torch.no_grad():
        action_chunk = policy.predict(obs, deterministic=True)
    assert action_chunk.shape == (4, 3, action_dim)
    assert (action_chunk >= -1.0 - 1e-5).all() and (action_chunk <= 1.0 + 1e-5).all()
