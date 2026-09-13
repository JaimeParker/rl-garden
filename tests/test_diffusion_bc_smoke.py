from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import DiffusionBC, OfflineEnvSpec
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks import DiffusionUNet1D

_IMAGE_SIZE = 16
# Small + fast: "gap" pooling (unlike the default "flatten") tolerates tiny
# images without PlainConv's flatten-layer size mismatch.
_TEST_ENCODER_CONFIG = EncoderConfig(features_dim=16, plain_conv_pooling="gap")


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


def _write_vision_h5_dataset(
    path, *, num_traj: int, steps_per_traj: int, state_dim: int, action_dim: int
) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for traj_idx in range(num_traj):
            g = f.create_group(f"traj_{traj_idx}")
            obs = g.create_group("obs")
            obs.create_dataset(
                "rgb_cam",
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


def _env_spec(obs_dim: int, action_dim: int) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
    )


def _dict_env_spec(state_dim: int, action_dim: int) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
    )


def _make_dict_agent(path, state_dim=4, action_dim=2, **kwargs) -> DiffusionBC:
    defaults = dict(
        env=_dict_env_spec(state_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        mlp_dims=[32, 32, 32],
        batch_size=16,
        actor_lr=1e-3,
        device="cpu",
        encoder_config=_TEST_ENCODER_CONFIG,
    )
    defaults.update(kwargs)
    return DiffusionBC(**defaults)


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

    # State-only purity pin: the Box path's actor_extractor is a
    # parameterless FlattenExtractor, so a Box-trained checkpoint's
    # state_dict stays byte-identical to before the observation redesign
    # (required for DPPOPolicy.load_actor_weights, which only loads
    # ``net.*`` keys).
    assert not any("actor_extractor" in k for k in agent.policy.state_dict())

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
    from rl_garden.encoders.flatten import FlattenExtractor
    from rl_garden.policies.diffusion_policy import DiffusionPolicy

    state_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
    obs_space = spaces.Dict({"state": state_space})
    policy = DiffusionPolicy(
        observation_space=obs_space,
        action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        actor_extractor=FlattenExtractor(obs_space),
        horizon_steps=3,
        cond_steps=2,
        denoising_steps=5,
        mlp_dims=[16, 16, 16],
    )
    obs = {"state": torch.randn(4, obs_dim)}
    with torch.no_grad():
        action_chunk = policy.predict(obs, deterministic=True)
    assert action_chunk.shape == (4, 3, action_dim)
    assert (action_chunk >= -1.0 - 1e-5).all() and (action_chunk <= 1.0 + 1e-5).all()


def test_unet_backbone_trains_and_checkpoint_roundtrips(tmp_path):
    obs_dim, action_dim = 4, 2
    path = tmp_path / "bc_dataset.h5"
    _write_h5_dataset(path, num_traj=8, steps_per_traj=20, obs_dim=obs_dim, action_dim=action_dim)

    net_kwargs = dict(down_dims=(8, 16), kernel_size=3, n_groups=4)
    agent = DiffusionBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        net_cls=DiffusionUNet1D,
        net_kwargs=net_kwargs,
        batch_size=16,
        actor_lr=1e-3,
        device="cpu",
    )
    assert isinstance(agent.policy.net, DiffusionUNet1D)

    metrics = agent.train(5)
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    ckpt_path = agent.save(tmp_path / "diffusion_bc_unet.pt")
    ema_state_before = {
        k: v.clone() for k, v in agent.ema_policy.net.state_dict().items()
    }

    # Reloading always needs net_cls/net_kwargs passed explicitly again --
    # the saved "net_cls"/"net_kwargs" checkpoint metadata is informational
    # only (matches every other algorithm's checkpoint-metadata convention
    # in this repo, e.g. MeanFlowBC), not consulted by `load()` to rebuild
    # the network shape.
    agent2 = DiffusionBC(
        env=_env_spec(obs_dim, action_dim),
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=10,
        net_cls=DiffusionUNet1D,
        net_kwargs=net_kwargs,
        batch_size=16,
        device="cpu",
    )
    agent2.load(ckpt_path)
    for k, v in agent2.ema_policy.net.state_dict().items():
        assert torch.allclose(v, ema_state_before[k])


def test_kernel_init_is_forwarded_to_non_default_backbone():
    """Regression test: DiffusionPolicy's net_cls dispatch (`build_diffusion_net`)
    must forward kernel_init to non-DiffusionMLP backbones too, not only the
    default DiffusionMLP path."""
    from rl_garden.encoders.flatten import FlattenExtractor
    from rl_garden.policies.diffusion_policy import DiffusionPolicy

    obs_dim, action_dim = 4, 2
    obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
    net_kwargs = dict(down_dims=(8, 16), kernel_size=3, n_groups=4)

    def build(kernel_init):
        torch.manual_seed(0)
        return DiffusionPolicy(
            observation_space=obs_space,
            action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
            actor_extractor=FlattenExtractor(obs_space),
            horizon_steps=2,
            cond_steps=1,
            denoising_steps=10,
            net_cls=DiffusionUNet1D,
            net_kwargs=net_kwargs,
            kernel_init=kernel_init,
        )

    default_init = build(None)
    xavier_init = build("xavier_uniform")

    differs = any(
        not torch.allclose(v, xavier_init.net.state_dict()[k])
        for k, v in default_init.net.state_dict().items()
    )
    assert differs, "kernel_init should change DiffusionUNet1D's initial parameters"


def test_dict_obs_actor_extractor_attribute_present(tmp_path):
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=6, steps_per_traj=10, state_dim=4, action_dim=2)
    agent = _make_dict_agent(path)
    assert hasattr(agent.policy, "actor_extractor")


def test_dict_obs_loss_decreases_and_checkpoint_roundtrips(tmp_path):
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=8, steps_per_traj=16, state_dim=4, action_dim=2)
    agent = _make_dict_agent(path)

    early = agent.train(10)["loss"]
    late = agent.train(40)["loss"]
    assert late < early

    ckpt_path = agent.save(tmp_path / "diffusion_bc_dict.pt")
    ema_state_before = {
        k: v.clone() for k, v in agent.ema_policy.net.state_dict().items()
    }

    agent2 = _make_dict_agent(path)
    agent2.load(ckpt_path)
    for k, v in agent2.ema_policy.net.state_dict().items():
        assert torch.allclose(v, ema_state_before[k])


def test_dict_obs_predict_returns_action_chunk_within_bounds(tmp_path):
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=4, steps_per_traj=10, state_dim=3, action_dim=2)
    agent = _make_dict_agent(path, state_dim=3, action_dim=2)

    obs = {
        "rgb_cam": torch.randint(0, 256, (4, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=torch.uint8),
        "state": torch.randn(4, 3),
    }
    with torch.no_grad():
        action_chunk = agent.policy.predict(obs, deterministic=True)
    assert action_chunk.shape == (4, 2, 2)
    assert (action_chunk >= -1.0 - 1e-5).all() and (action_chunk <= 1.0 + 1e-5).all()


def test_dict_obs_unet_backbone_trains_and_checkpoint_roundtrips(tmp_path):
    """Newly reachable combination through the shared build_diffusion_net
    dispatch (net_cls was hardcoded to DiffusionMLP on the old
    VisionDiffusionPolicy)."""
    path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(path, num_traj=8, steps_per_traj=16, state_dim=4, action_dim=2)
    net_kwargs = dict(down_dims=(8, 16), kernel_size=3, n_groups=4)
    agent = _make_dict_agent(path, net_cls=DiffusionUNet1D, net_kwargs=net_kwargs)
    assert isinstance(agent.policy.net, DiffusionUNet1D)

    metrics = agent.train(5)
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    ckpt_path = agent.save(tmp_path / "diffusion_bc_dict_unet.pt")
    ema_state_before = {
        k: v.clone() for k, v in agent.ema_policy.net.state_dict().items()
    }

    agent2 = _make_dict_agent(path, net_cls=DiffusionUNet1D, net_kwargs=net_kwargs)
    agent2.load(ckpt_path)
    for k, v in agent2.ema_policy.net.state_dict().items():
        assert torch.allclose(v, ema_state_before[k])


def test_dict_dataset_rejected_for_box_obs_agent(tmp_path):
    """_load_dataset must reject a Dict-shaped H5 file for a Box observation
    space (and vice versa) -- both mismatch directions."""
    dict_path = tmp_path / "vision_bc_dataset.h5"
    _write_vision_h5_dataset(dict_path, num_traj=2, steps_per_traj=4, state_dim=4, action_dim=2)
    with pytest.raises(TypeError):
        DiffusionBC(
            env=_env_spec(4, 2),
            dataset_path=str(dict_path),
            horizon_steps=2,
            cond_steps=1,
            denoising_steps=5,
            mlp_dims=[16, 16, 16],
            batch_size=8,
            device="cpu",
        )


def test_box_dataset_rejected_for_dict_obs_agent(tmp_path):
    box_path = tmp_path / "bc_dataset.h5"
    _write_h5_dataset(box_path, num_traj=2, steps_per_traj=4, obs_dim=4, action_dim=2)
    with pytest.raises(TypeError):
        _make_dict_agent(box_path)
