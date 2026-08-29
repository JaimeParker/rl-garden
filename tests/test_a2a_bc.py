"""Tests for A2ABC/A2APolicy: A2A flow-matching BC pretraining, standalone
sibling of VisionDiffusionBC/VisionDiffusionPolicy (neither existing class is
modified)."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import A2ABC, OfflineEnvSpec
from rl_garden.encoders.combined import default_image_encoder_factory
from rl_garden.networks import ActionChunkDecoder, CNNSequenceEncoder
from rl_garden.policies.a2a_policy import A2APolicy

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


def _make_agent(path, state_dim=4, action_dim=2, **kwargs) -> A2ABC:
    defaults = dict(
        env=_env_spec(state_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=4,
        cond_steps=2,
        latent_dim=16,
        cnn_num_layers=1,
        cnn_hidden_channels=8,
        decoder_net_arch=[32, 32],
        flow_hidden_dims=[32, 32],
        batch_size=16,
        actor_lr=1e-3,
        device="cpu",
        image_encoder_factory=_test_image_encoder_factory,
        image_keys=("rgb",),
        state_key="state",
    )
    defaults.update(kwargs)
    return A2ABC(**defaults)


def test_policy_is_a2a_policy(tmp_path):
    path = tmp_path / "a2a_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=6, steps_per_traj=10, state_dim=4, action_dim=2)
    agent = _make_agent(path)
    assert isinstance(agent.policy, A2APolicy)


def test_rejects_box_observation_space():
    env = OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )
    with pytest.raises(TypeError):
        A2ABC(env=env, dataset_path="unused.h5", device="cpu")


def test_loss_decreases_and_checkpoint_roundtrips(tmp_path):
    path = tmp_path / "a2a_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=8, steps_per_traj=16, state_dim=4, action_dim=2)
    agent = _make_agent(path)

    early = agent.train(10)["loss"]
    late = agent.train(40)["loss"]
    assert late < early

    ckpt_path = agent.save(tmp_path / "a2a_bc.pt")
    policy_state_before = {k: v.clone() for k, v in agent.policy.state_dict().items()}

    agent2 = _make_agent(path)
    agent2.load(ckpt_path)
    for k, v in agent2.policy.state_dict().items():
        assert torch.allclose(v, policy_state_before[k])


def test_predict_returns_action_chunk_within_bounds(tmp_path):
    path = tmp_path / "a2a_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=4, steps_per_traj=10, state_dim=3, action_dim=2)
    agent = _make_agent(path, state_dim=3, action_dim=2)

    # Single-frame obs, broadcast to cond_steps.
    obs_single = {
        "rgb": torch.randint(0, 256, (4, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=torch.uint8),
        "state": torch.randn(4, 3),
    }
    with torch.no_grad():
        action_chunk = agent.policy.predict(obs_single, deterministic=True)
    assert action_chunk.shape == (4, 4, 2)
    assert (action_chunk >= -1.0 - 1e-5).all() and (action_chunk <= 1.0 + 1e-5).all()

    # Explicit history obs.
    obs_history = {
        "rgb": torch.randint(
            0, 256, (4, agent.cond_steps, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=torch.uint8
        ),
        "state": torch.randn(4, agent.cond_steps, 3),
    }
    with torch.no_grad():
        action_chunk = agent.policy.predict(obs_history, deterministic=True)
    assert action_chunk.shape == (4, 4, 2)
    assert (action_chunk >= -1.0 - 1e-5).all() and (action_chunk <= 1.0 + 1e-5).all()


def test_cnn_sequence_encoder_decoder_roundtrip_shapes():
    encoder = CNNSequenceEncoder(input_dim=5, seq_len=8, latent_dim=16, num_layers=2)
    decoder = ActionChunkDecoder(latent_dim=16, horizon=4, output_dim=3, net_arch=[32, 32])

    x = torch.randn(6, 8, 5)
    z = encoder(x)
    assert z.shape == (6, 16)

    z2 = torch.randn(6, 16)
    y = decoder(z2)
    assert y.shape == (6, 4, 3)

    with pytest.raises(ValueError):
        CNNSequenceEncoder(input_dim=5, seq_len=0, latent_dim=16, num_layers=2)
    with pytest.raises(ValueError):
        CNNSequenceEncoder(input_dim=5, seq_len=8, latent_dim=16, kernel_size=2)


def test_flow_operates_in_latent_space(tmp_path):
    path = tmp_path / "a2a_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=4, steps_per_traj=10, state_dim=4, action_dim=2)
    agent = _make_agent(path, state_dim=4, action_dim=2)
    assert agent.policy.flow_net.action_dim == agent.latent_dim
    assert agent.policy.flow_net.action_dim != agent.policy.action_dim
