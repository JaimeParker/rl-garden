"""Tests for VisionDiffusionBC/VisionDiffusionPolicy: vision-conditioned
diffusion BC pretraining, standalone sibling of DiffusionBC/DiffusionPolicy
(neither existing class is modified)."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec, VisionDiffusionBC
from rl_garden.encoders.combined import default_image_encoder_factory
from rl_garden.policies.vision_diffusion_policy import VisionDiffusionPolicy

_IMAGE_SIZE = 16
_test_image_encoder_factory = default_image_encoder_factory(
    features_dim=16, plain_conv_pooling="gap"
)


def _write_vision_h5_dataset(
    path, *, num_traj: int, steps_per_traj: int, state_dim: int, action_dim: int
) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for traj_idx in range(num_traj):
            g = f.create_group(f"traj_{traj_idx}")
            obs = g.create_group("obs")
            obs.create_dataset(
                "rgb",
                data=rng.integers(
                    0, 256, (steps_per_traj + 1, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8
                ),
            )
            obs.create_dataset(
                "state",
                data=rng.standard_normal((steps_per_traj + 1, state_dim)).astype(np.float32),
            )
            g.create_dataset(
                "actions",
                data=(rng.random((steps_per_traj, action_dim)).astype(np.float32) * 2 - 1),
            )
            g.create_dataset("rewards", data=np.zeros(steps_per_traj, dtype=np.float32))
            dones = np.zeros(steps_per_traj, dtype=np.float32)
            dones[-1] = 1.0
            g.create_dataset("dones", data=dones)


def _env_spec(state_dim: int, action_dim: int) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
    )


def _make_agent(path, state_dim=4, action_dim=2, **kwargs) -> VisionDiffusionBC:
    defaults = dict(
        env=_env_spec(state_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        actor_lr=1e-3,
        device="cpu",
        image_encoder_factory=_test_image_encoder_factory,
        image_keys=("rgb",),
        state_key="state",
    )
    defaults.update(kwargs)
    return VisionDiffusionBC(**defaults)


def test_policy_is_vision_diffusion_policy(tmp_path):
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=6, steps_per_traj=10, state_dim=4, action_dim=2)
    agent = _make_agent(path)
    assert isinstance(agent.policy, VisionDiffusionPolicy)


def test_rejects_box_observation_space():
    env = OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )
    try:
        VisionDiffusionBC(env=env, dataset_path="unused.h5", device="cpu")
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_loss_decreases_and_checkpoint_roundtrips(tmp_path):
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=8, steps_per_traj=16, state_dim=4, action_dim=2)
    agent = _make_agent(path)

    early = agent.train(10)["loss"]
    late = agent.train(40)["loss"]
    assert late < early

    ckpt_path = agent.save(tmp_path / "vision_diffusion_bc.pt")
    ema_state_before = {
        k: v.clone() for k, v in agent.ema_policy.net.state_dict().items()
    }

    agent2 = _make_agent(path)
    agent2.load(ckpt_path)
    for k, v in agent2.ema_policy.net.state_dict().items():
        assert torch.allclose(v, ema_state_before[k])


def test_predict_returns_action_chunk_within_bounds(tmp_path):
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=4, steps_per_traj=10, state_dim=3, action_dim=2)
    agent = _make_agent(path, state_dim=3, action_dim=2)

    obs = {
        "rgb": torch.randint(0, 256, (4, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=torch.uint8),
        "state": torch.randn(4, 3),
    }
    with torch.no_grad():
        action_chunk = agent.policy.predict(obs, deterministic=True)
    assert action_chunk.shape == (4, 2, 2)
    assert (action_chunk >= -1.0 - 1e-5).all() and (action_chunk <= 1.0 + 1e-5).all()
