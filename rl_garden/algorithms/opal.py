"""OPAL (Ajay et al. 2021, "OPAL: Offline Primitive Discovery for
Accelerating Offline Reinforcement Learning"): pretrains a skill-space VAE
(``OPALVAE``, ``rl_garden/networks/opal_vae.py``) over fixed-length action
chunks, producing a learned skill embedding and a frozen low-level decoder
that a downstream high-level policy can act through.

Standalone offline algorithm (``OfflineRLAlgorithm``), self-managed dataset
via ``WindowedTrajectoryDataset`` (``rl_garden/buffers/windowed_trajectory_dataset.py``)
-- no replay buffer, mirrors ``HILP``/``A2ABC``'s reasoning for the same
shape. State-based (Box observations) only.

Ported from ``SUPE/supe/pretraining/opal.py`` (read in full). SUPE's online
meta-policy (a plain SAC/RLPD actor-critic over this VAE's skill space, plus
a skill-macro-action env wrapper) is implemented separately as
``rl_garden/algorithms/supe.py``, which consumes an ``OPAL`` checkpoint
produced by this class.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.windowed_trajectory_dataset import WindowedTrajectoryDataset
from rl_garden.common.logger import Logger
from rl_garden.networks.opal_vae import OPALVAE


class OPAL(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("OPAL",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        dataset_path: str,
        *,
        skill_dim: int = 8,
        chunk_size: int = 4,
        hidden_size: int = 256,
        vae_hidden_dims: Sequence[int] = (256, 256),
        kl_coef: float = 0.1,
        lr: float = 3e-4,
        batch_size: int = 256,
        num_traj: Optional[int] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            buffer_size=1,
            buffer_device="cpu",
            batch_size=batch_size,
            offline_sampling="with_replace",
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=0,
            eval_env=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=False,
            save_final_checkpoint=save_final_checkpoint,
        )
        if not isinstance(self.env.single_observation_space, spaces.Box):
            raise TypeError("OPAL supports Box observation spaces only.")

        self.dataset_path = dataset_path
        self.skill_dim = skill_dim
        self.chunk_size = chunk_size
        self.hidden_size = hidden_size
        self.vae_hidden_dims = tuple(vae_hidden_dims)
        self.kl_coef = kl_coef
        self.lr = lr
        self.num_traj = num_traj

        self._dataset = WindowedTrajectoryDataset(
            dataset_path, chunk_size=chunk_size, device=self.device, num_traj=num_traj,
        )
        self._setup_model()

    def _setup_model(self) -> None:
        obs_dim = int(np.prod(self.env.single_observation_space.shape))
        action_space = self.env.single_action_space

        self.policy = OPALVAE(
            obs_dim, action_space, self.skill_dim, self.chunk_size,
            hidden_size=self.hidden_size,
            prior_hidden_dims=self.vae_hidden_dims,
            decoder_hidden_dims=self.vae_hidden_dims,
        ).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            batch = self._dataset.sample(self.batch_size)
            losses = self.policy.loss(batch.obs_window, batch.action_window, kl_coef=self.kl_coef)

            self.vae_optimizer.zero_grad(set_to_none=True)
            losses["vae_loss"].backward()
            self.vae_optimizer.step()

            for key, value in losses.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value.detach().item())

        del compute_info
        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("vae_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "skill_dim": self.skill_dim,
            "chunk_size": self.chunk_size,
            "hidden_size": self.hidden_size,
            "vae_hidden_dims": self.vae_hidden_dims,
            "kl_coef": self.kl_coef,
        }
