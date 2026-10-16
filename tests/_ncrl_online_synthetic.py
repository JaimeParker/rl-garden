"""Synthetic artifacts for the NCRL stage-2 tests: a converted two-embodiment
dataset (64x64 frames, walker action dim 6, quadruped 12 -> padded 12), a
tiny ``RSSMPretrain`` checkpoint trained a few steps on it, the retrieval DB
built from both, and a fake pixel vector env with ``rgb_default`` 64x64
observations and fixed-length episodes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from _ncrl_synthetic import write_benchmark  # noqa: E402
from build_ncrl_retrieval_db import build_retrieval_db  # noqa: E402
from convert_ncrl_dataset import convert_dataset  # noqa: E402

from rl_garden.algorithms.rssm_pretrain import RSSMPretrain, sequence_dataset_env_spec  # noqa: E402
from rl_garden.world_models.rssm import RSSMSize  # noqa: E402

EMBODIMENTS = {"walker": (6, (14, 16, 12, 15)), "quadruped": (12, (15, 13))}
IMAGE_SIZE = 64
TINY_SIZE = RSSMSize(deter=32, hidden=16, discrete=4, units=32, cnn_depth=4)
TINY_STOCH = 4
ENV_ACTION_DIM = 6
PADDED_ACTION_DIM = 12


def make_pretrain_artifacts(root: Path, backbone: str = "dreamer_v3") -> dict[str, Path]:
    """Writes ``{dataset_dir, checkpoint, retrieval_db}`` under ``root``, the
    checkpoint pretrained with RSSM ``backbone``."""
    write_benchmark(root / "src", EMBODIMENTS, image_size=IMAGE_SIZE)
    dataset_dir = root / "dataset"
    convert_dataset(root / "src", dataset_dir)
    pretrain = RSSMPretrain(
        sequence_dataset_env_spec(dataset_dir),
        dataset_dir=dataset_dir,
        backbone=backbone,
        rssm_size=TINY_SIZE,
        stoch=TINY_STOCH,
        batch_size=4,
        batch_length=6,
        compute_dtype="float32",
        device="cpu",
        buffer_device="cpu",
        log_freq=0,
        std_log=False,
    )
    pretrain.train(2)
    checkpoint = pretrain.save(root / "rssm_pretrain.pt")
    db_path = root / "retrieval_db.pt"
    torch.save(build_retrieval_db(dataset_dir, checkpoint, device="cpu"), db_path)
    return {"dataset_dir": dataset_dir, "checkpoint": checkpoint, "retrieval_db": db_path}


class PixelVecEnv:
    """Fake vectorized env: ``{"rgb_default": uint8 (N, 64, 64, 3)}``
    observations, Box(-1, 1, (action_dim,)) actions, every episode truncated
    after ``episode_steps`` steps (gymnasium same-step autoreset)."""

    def __init__(
        self,
        num_envs: int = 1,
        action_dim: int = ENV_ACTION_DIM,
        episode_steps: int = 10,
        extra_state: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        obs_spaces = {"rgb_default": spaces.Box(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)}
        if extra_state:
            obs_spaces["state"] = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.single_observation_space = spaces.Dict(obs_spaces)
        self.single_action_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (num_envs, action_dim), dtype=np.float32)
        self.extra_state = extra_state
        self._t = 0
        self.actions_seen: list[torch.Tensor] = []

    def _obs(self):
        obs = {"rgb_default": torch.randint(0, 256, (self.num_envs, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8)}
        if self.extra_state:
            obs["state"] = torch.zeros(self.num_envs, 3)
        return obs

    def reset(self, seed: int | None = None):
        del seed
        self._t = 0
        return self._obs(), {}

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, self.single_action_space.shape[0]), actions.shape
        assert torch.all(actions.abs() <= 1.0 + 1e-5)
        self.actions_seen.append(actions.clone())
        self._t += 1
        rewards = torch.rand(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        infos: dict = {}
        if self._t % self.episode_steps == 0:
            truncations[:] = True
            infos["final_observation"] = self._obs()
            infos["final_info"] = {"episode": {"return": torch.ones(self.num_envs)}}
            infos["_final_info"] = torch.ones(self.num_envs, dtype=torch.bool)
        return self._obs(), rewards, terminations, truncations, infos

    def close(self) -> None:
        return None


def tiny_ncrl_kwargs(artifacts: dict[str, Path], **overrides) -> dict:
    kwargs = dict(
        pretrained_world_model=str(artifacts["checkpoint"]),
        retrieval_db=str(artifacts["retrieval_db"]),
        num_retrieve_episodes=3,
        bc_train_steps=20,
        bc_batch_size=8,
        episode_frames=10,
        action_repeat=1,
        warmup_frames=20,  # -> warmup_steps = max(20, 5 * 10) = 50
        learning_starts=60,
        batch_size=4,
        batch_length=6,
        train_ratio=24.0,  # utd = 1
        imag_horizon=3,
        ac_units=32,
        buffer_size=400,
        buffer_device="cpu",
        device="cpu",
        compute_dtype="float32",
        eval_freq=0,
        log_freq=0,
        std_log=False,
    )
    kwargs.update(overrides)
    return kwargs
