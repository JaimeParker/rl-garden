"""``TDMPC2Multitask``: offline multitask training, ported from the
``cfg.multitask`` branches of ``3rd_party/tdmpc2/tdmpc2/{tdmpc2.py,
trainer/offline_trainer.py}``.

Scope for this port (see the brainstorming session this was planned from):

- **Offline only, matching upstream exactly.** Upstream's ``train.py:52``
  picks ``OfflineTrainer`` (never ``OnlineTrainer``) whenever
  ``cfg.multitask`` -- multitask training never touches a live env; data
  comes entirely from a pre-converted dataset (see
  ``multitask/dataset.py``). This class therefore inherits
  ``OfflineRLAlgorithm``, not ``BaseAlgorithm``/``OffPolicyAlgorithm``, and
  implements ``train(gradient_steps)`` for the generic
  ``run_offline_pretraining`` loop instead of a custom ``learn()``.
- **No CEM planner / no eval.** The planner (``act()``/``_plan()`` upstream)
  is only invoked for live rollout, which multitask training never does in
  this port -- evaluation (upstream's ``OfflineTrainer.eval()``, which needs
  a DMControl/MetaWorld-backed ``MultitaskWrapper``-equivalent env) is a
  deliberately deferred future extension, not attempted here.
- **No termination head / no episodic tasks.** Same reasoning as the
  single-task port: without a live rollout there is no notion of a "true"
  terminal transition distinct from a truncated one in this buffer's data,
  and upstream's own task sets default to non-episodic anyway.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms.tdmpc2 import math_utils
from rl_garden.algorithms.tdmpc2.agent import _compute_discount
from rl_garden.algorithms.tdmpc2.multitask.buffer import MmapMultitaskEpisodeBuffer
from rl_garden.algorithms.tdmpc2.multitask.policy import MultitaskTDMPC2Policy
from rl_garden.algorithms.tdmpc2.multitask.world_model import MultitaskWorldModel
from rl_garden.buffers.mmap_storage import MmapMode
from rl_garden.common.logger import Logger


class TDMPC2Multitask(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("TDMPC2Multitask",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        tasks: Sequence[str],
        obs_dims: Sequence[int],
        action_dims: Sequence[int],
        episode_lengths: Sequence[int],
        mmap_dir: str | Path,
        mmap_mode: MmapMode = "create",
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        horizon: int = 3,
        task_dim: int = 96,
        latent_dim: int = 512,
        enc_dim: int = 256,
        num_enc_layers: int = 2,
        mlp_dim: int = 512,
        simnorm_dim: int = 8,
        num_q: int = 5,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        entropy_coef: float = 1e-4,
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        grad_clip_norm: float = 20.0,
        tau: float = 0.01,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        discount_denom: float = 5.0,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        if not (len(tasks) == len(obs_dims) == len(action_dims) == len(episode_lengths)):
            raise ValueError(
                "tasks/obs_dims/action_dims/episode_lengths must all have the "
                f"same length, got {len(tasks)}/{len(obs_dims)}/{len(action_dims)}/"
                f"{len(episode_lengths)}."
            )
        super().__init__(
            env=env,
            buffer_size=buffer_size,
            buffer_device="cpu",  # MmapMultitaskEpisodeBuffer is always CPU-backed.
            batch_size=batch_size,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=0,  # v1 has no multitask eval path; see module docstring.
            eval_env=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self.tasks = list(tasks)
        self.obs_dims = list(obs_dims)
        self.action_dims = list(action_dims)
        self.episode_lengths = list(episode_lengths)
        self.mmap_dir = mmap_dir
        self.mmap_mode = mmap_mode
        self.horizon = horizon
        self.task_dim = task_dim
        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.num_enc_layers = num_enc_layers
        self.mlp_dim = mlp_dim
        self.simnorm_dim = simnorm_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.dropout = dropout
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.enc_lr_scale = enc_lr_scale
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.rho = rho
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.discount_denom = discount_denom
        self.discount_min = discount_min
        self.discount_max = discount_max

        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        world_model = MultitaskWorldModel(
            num_tasks=len(self.tasks),
            obs_dims=self.obs_dims,
            action_dims=self.action_dims,
            task_dim=self.task_dim,
            latent_dim=self.latent_dim,
            enc_dim=self.enc_dim,
            num_enc_layers=self.num_enc_layers,
            mlp_dim=self.mlp_dim,
            simnorm_dim=self.simnorm_dim,
            num_q=self.num_q,
            num_bins=self.num_bins,
            vmin=self.vmin,
            vmax=self.vmax,
            dropout=self.dropout,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            tau=self.tau,
        ).to(self.device)

        self.discount = torch.tensor(
            [
                _compute_discount(ep_len, self.discount_denom, self.discount_min, self.discount_max)
                for ep_len in self.episode_lengths
            ],
            device=self.device,
        )

        self.policy = MultitaskTDMPC2Policy(world_model).to(self.device)

        enc_params = list(world_model._encoder.parameters())
        other_params = (
            list(world_model._dynamics.parameters())
            + list(world_model._reward.parameters())
            + list(world_model._Q.parameters())
        )
        self.world_optimizer = torch.optim.Adam(
            [
                {"params": enc_params, "lr": self.lr * self.enc_lr_scale},
                {"params": other_params, "lr": self.lr},
            ]
        )
        self.pi_optimizer = torch.optim.Adam(world_model._pi.parameters(), lr=self.lr, eps=1e-5)

        self.replay_buffer = MmapMultitaskEpisodeBuffer(
            obs_dim=world_model.obs_dim,
            action_dim=world_model.action_dim,
            buffer_size=self.buffer_size,
            horizon=self.horizon,
            mmap_dir=self.mmap_dir,
            mmap_mode=self.mmap_mode,
            storage_device="cpu",
            sample_device=self.device,
        )

    # ------------------------------------------------------------------
    # Gradient step
    # ------------------------------------------------------------------

    def _td_target(
        self,
        next_z: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        world_model = self.policy.world_model
        action, _ = world_model.pi(next_z, task)
        return reward + discount * world_model.Q(next_z, action, task, return_type="min", target=True)

    def _update_pi(self, zs: torch.Tensor, task: torch.Tensor) -> dict[str, float]:
        world_model = self.policy.world_model
        action, info = world_model.pi(zs, task)
        qs = world_model.Q(zs, action, task, return_type="avg", detach=True)
        world_model.scale.update(qs[0])
        qs = world_model.scale(qs)

        rho_pows = self.rho ** torch.arange(zs.shape[0], device=zs.device)
        pi_loss = (
            -(self.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho_pows
        ).mean()

        self.pi_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(world_model._pi.parameters(), self.grad_clip_norm)
        self.pi_optimizer.step()

        return {
            "pi_loss": float(pi_loss.detach()),
            "pi_grad_norm": float(pi_grad_norm),
            "pi_scale": float(world_model.scale.value.item()),
        }

    def _update_step(self) -> dict[str, float]:
        world_model = self.policy.world_model
        sample = self.replay_buffer.sample(self.batch_size)
        obs, action, reward, task = sample.obs, sample.action, sample.reward, sample.task
        horizon, batch_size = action.shape[0], action.shape[1]
        reward = reward.unsqueeze(-1)
        discount = self.discount[task].unsqueeze(-1)

        with torch.no_grad():
            next_obs = obs[1:]
            next_z = world_model.encode(next_obs, task)
            td_targets = self._td_target(next_z, reward, discount, task)

        world_model.train()
        zs = torch.empty(horizon + 1, batch_size, self.latent_dim, device=self.device)
        z = world_model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = torch.zeros((), device=self.device)
        for t in range(horizon):
            z = world_model.next(z, action[t], task)
            consistency_loss = consistency_loss + F.mse_loss(z, next_z[t]) * self.rho**t
            zs[t + 1] = z

        _zs = zs[:-1]
        qs = world_model.Q(_zs, action, task, return_type="all")
        reward_preds = world_model.reward(_zs, action, task)

        reward_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        for t in range(horizon):
            reward_loss = reward_loss + math_utils.soft_ce(
                reward_preds[t], reward[t], self.num_bins, self.vmin, self.vmax, world_model.bin_size
            ).mean() * self.rho**t
            for qi in range(self.num_q):
                value_loss = value_loss + math_utils.soft_ce(
                    qs[qi, t], td_targets[t], self.num_bins, self.vmin, self.vmax, world_model.bin_size
                ).mean() * self.rho**t

        consistency_loss = consistency_loss / horizon
        reward_loss = reward_loss / horizon
        value_loss = value_loss / (horizon * self.num_q)

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        )

        self.world_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(), self.grad_clip_norm)
        self.world_optimizer.step()

        pi_info = self._update_pi(zs.detach(), task)
        world_model.soft_update_target_Q()
        world_model.eval()

        self._global_update += 1
        info = {
            "consistency_loss": float(consistency_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm),
        }
        info.update(pi_info)
        return info

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info  # every step's info dict is cheap; always computed.
        info: dict[str, float] = {}
        for _ in range(gradient_steps):
            info = self._update_step()
        return info

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("world_optimizer", "pi_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "tasks": self.tasks,
            "obs_dims": self.obs_dims,
            "action_dims": self.action_dims,
            "episode_lengths": self.episode_lengths,
            "horizon": self.horizon,
            "task_dim": self.task_dim,
            "latent_dim": self.latent_dim,
            "num_q": self.num_q,
            "num_bins": self.num_bins,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "discount_denom": self.discount_denom,
            "discount_min": self.discount_min,
            "discount_max": self.discount_max,
        }
