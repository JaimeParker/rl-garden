"""ACT training from transition pkls: a tiny run whose checkpoint ACTBaseActionProvider loads."""
from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

# The vendored ACT model code needs torchvision and IPython.
pytest.importorskip("torchvision")
pytest.importorskip("IPython")

from rl_garden.models.act.provider import ACTBaseActionProvider  # noqa: E402
from rl_garden.models.act.train_trajectories import (  # noqa: E402
    ACTWeightEma,
    Args,
    prepare_act_images,
    train_act_from_trajectories,
)

STATE_DIM = 3
ACTION_DIM = 2
CAMERAS = ("rgb_wrist", "rgb_front")
IMAGE_HW = (20, 24)
IMAGE_SIZE = 32


def _obs(generator: torch.Generator) -> dict[str, torch.Tensor]:
    obs = {"state": torch.randn(1, STATE_DIM, generator=generator)}
    for camera in CAMERAS:
        obs[camera] = torch.randint(0, 256, (1, *IMAGE_HW, 3), dtype=torch.uint8, generator=generator)
    return obs


def _write_demo_pkl(path: Path, *, num_episodes: int = 2, episode_length: int = 5) -> None:
    generator = torch.Generator().manual_seed(0)
    transitions = []
    for _ in range(num_episodes):
        obs = _obs(generator)
        for step in range(episode_length):
            next_obs = _obs(generator)
            transitions.append(
                {
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": torch.rand(1, ACTION_DIM, generator=generator) * 2 - 1,
                    "reward": torch.zeros(1),
                    "done": torch.tensor([step == episode_length - 1]),
                }
            )
            obs = next_obs
    with open(path, "wb") as stream:
        pickle.dump(transitions, stream)


def _tiny_args(tmp_path: Path, **overrides) -> Args:
    args = Args(
        demo_paths=(str(tmp_path / "*.pkl"),),
        exp_name="tiny",
        log_dir=str(tmp_path / "runs"),
        log_type="none",
        cuda=False,
        total_iters=2,
        batch_size=4,
        log_freq=1,
        backbone_weights="none",
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=32,
        hidden_dim=16,
        nheads=2,
        num_queries=4,
        image_size=IMAGE_SIZE,
    )
    return dataclasses.replace(args, **overrides)


def _spaces(*, cameras: tuple[str, ...]) -> tuple[spaces.Dict, spaces.Box]:
    entries = {"state": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32)}
    for camera in cameras:
        entries[camera] = spaces.Box(0, 255, shape=(*IMAGE_HW, 3), dtype=np.uint8)
    return spaces.Dict(entries), spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)


def _load_provider(path: Path, *, cameras: tuple[str, ...]) -> ACTBaseActionProvider:
    observation_space, action_space = _spaces(cameras=cameras)
    return ACTBaseActionProvider.from_checkpoint(
        observation_space=observation_space, action_space=action_space, ckpt_path=path
    )


def test_visual_checkpoint_loads_and_acts(tmp_path) -> None:
    _write_demo_pkl(tmp_path / "demos.pkl")

    path = train_act_from_trajectories(_tiny_args(tmp_path))

    assert path == tmp_path / "runs" / "tiny" / "checkpoints" / "final.pt"
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["camera_keys"] == ["rgb_front", "rgb_wrist"]
    provider = _load_provider(path, cameras=CAMERAS)
    assert provider.spec.visual
    assert (provider.spec.state_dim, provider.spec.action_dim) == (STATE_DIM, ACTION_DIM)
    assert (provider.config.image_size, provider.config.nheads) == (IMAGE_SIZE, 2)

    obs = _obs(torch.Generator().manual_seed(1))
    action = provider.select_action(obs)
    assert action.shape == (1, ACTION_DIM) and torch.isfinite(action).all()

    # Training feeds the backbone and the state exactly as the provider does at inference.
    prepared = provider._prepare_obs(obs)
    stacked = torch.stack([obs[key].permute(0, 3, 1, 2) for key in sorted(CAMERAS)], dim=1)
    torch.testing.assert_close(prepared["rgb"], prepare_act_images(stacked, IMAGE_SIZE))
    stats = checkpoint["norm_stats"]
    torch.testing.assert_close(prepared["state"], (obs["state"] - stats["state_mean"]) / stats["state_std"])


def test_state_only_checkpoint_loads_and_acts(tmp_path) -> None:
    _write_demo_pkl(tmp_path / "demos.pkl")

    path = train_act_from_trajectories(_tiny_args(tmp_path, use_cameras=False))

    provider = _load_provider(path, cameras=())
    assert not provider.spec.visual and provider.config.nheads == 2
    action = provider.select_action({"state": torch.zeros(1, STATE_DIM)})
    assert action.shape == (1, ACTION_DIM) and torch.isfinite(action).all()


def test_stored_config_is_checked_against_weights_and_optional(tmp_path) -> None:
    _write_demo_pkl(tmp_path / "demos.pkl")
    checkpoint = torch.load(
        train_act_from_trajectories(_tiny_args(tmp_path, use_cameras=False)), map_location="cpu"
    )

    mismatched = {**checkpoint, "act_config": {**checkpoint["act_config"], "hidden_dim": 32}}
    torch.save(mismatched, tmp_path / "mismatched.pt")
    with pytest.raises(ValueError, match="disagrees"):
        _load_provider(tmp_path / "mismatched.pt", cameras=())

    # Without a stored config the architecture is inferred, as for earlier ACT checkpoints.
    legacy = {key: value for key, value in checkpoint.items() if key != "act_config"}
    torch.save(legacy, tmp_path / "legacy.pt")
    assert _load_provider(tmp_path / "legacy.pt", cameras=()).config.nheads == 4


def test_ema_copies_on_first_update_then_blends() -> None:
    model = nn.Linear(2, 1)
    ema = ACTWeightEma(model)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight, model.weight)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    decay = 2 / 11
    torch.testing.assert_close(ema.model.weight, torch.full_like(model.weight, decay * 1.0 + (1 - decay) * 3.0))
