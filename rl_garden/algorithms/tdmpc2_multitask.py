"""``TDMPC2Multitask``: offline multitask training, ported from the
``cfg.multitask`` branches of ``3rd_party/tdmpc2/tdmpc2/{tdmpc2.py,
trainer/offline_trainer.py}``.

Scope for this port (see the brainstorming session this was planned from):

- **Offline only, matching upstream exactly.** Upstream's ``train.py:52``
  picks ``OfflineTrainer`` (never ``OnlineTrainer``) whenever
  ``cfg.multitask`` -- multitask training never touches a live env; data
  comes entirely from a pre-converted dataset (see
  ``rl_garden.buffers.tdmpc2_multitask_dataset``). This class therefore sits on
  ``OfflineModelBasedAlgorithm`` (``OfflineRLAlgorithm`` + the shared
  three-hook gradient step) with ``TDMPC2Core`` for the losses/optimizers
  it shares with the single-task ``TDMPC2``; the generic
  ``run_offline_pretraining`` loop drives its inherited ``train()``.
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

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.offline_model_based import OfflineModelBasedAlgorithm
from rl_garden.algorithms.tdmpc2 import TDMPC2Core, _compute_discount
from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer
from rl_garden.buffers.mmap_storage import MmapMode
from rl_garden.common.logger import Logger
from rl_garden.policies.tdmpc2_multitask_policy import MultitaskTDMPC2Policy
from rl_garden.world_models.base import State
from rl_garden.world_models.batch import SequenceBatch
from rl_garden.world_models.multitask_latent_consistency import MultitaskWorldModel


class TDMPC2Multitask(TDMPC2Core, OfflineModelBasedAlgorithm):
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
        self.task_dim = task_dim
        self.enc_dim = enc_dim
        self.num_enc_layers = num_enc_layers
        self.tau = tau
        self._init_tdmpc2_params(
            horizon=horizon,
            latent_dim=latent_dim,
            mlp_dim=mlp_dim,
            simnorm_dim=simnorm_dim,
            num_q=num_q,
            num_bins=num_bins,
            vmin=vmin,
            vmax=vmax,
            dropout=dropout,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            entropy_coef=entropy_coef,
            lr=lr,
            enc_lr_scale=enc_lr_scale,
            grad_clip_norm=grad_clip_norm,
            rho=rho,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            discount_denom=discount_denom,
            discount_min=discount_min,
            discount_max=discount_max,
        )

        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # NOTE: no `.to(self.device)` here -- deferred to the single
        # `self.policy = MultitaskTDMPC2Policy(...).to(self.device)` call
        # below, mirroring TDMPC2._setup_model's ordering (see that
        # method's comment).
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
            num_bins=self.num_bins,
            vmin=self.vmin,
            vmax=self.vmax,
            rho=self.rho,
        )

        self.discount = torch.tensor(
            [
                _compute_discount(ep_len, self.discount_denom, self.discount_min, self.discount_max)
                for ep_len in self.episode_lengths
            ],
            device=self.device,
        )

        self.policy = MultitaskTDMPC2Policy(
            world_model,
            mlp_dim=self.mlp_dim,
            num_q=self.num_q,
            dropout=self.dropout,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            tau=self.tau,
        ).to(self.device)
        world_model = self.policy.world_model  # now on self.device
        self._build_tdmpc2_optimizers()

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
    # Gradient step: OfflineModelBasedAlgorithm's three hooks.
    # ------------------------------------------------------------------

    def _td_target(
        self,
        next_z: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        action, _ = self.policy.pi(next_z, task)
        return reward + discount * self.policy.Q(next_z, action, task, return_type="min", target=True)

    def _update_model(self, batch) -> tuple[dict[str, float], State]:
        """Consistency + reward + **value** losses in one joint
        ``world_optimizer`` backward, as ``TDMPC2._update_model`` (no
        termination head here); per-task discount in the TD target."""
        world_model = self.policy.world_model
        obs, action, reward, task = batch.obs, batch.action, batch.reward, batch.task
        reward = reward.unsqueeze(-1)
        discount = self.discount[task].unsqueeze(-1)

        with torch.no_grad():
            next_z = world_model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, discount, task)

        world_model.train()
        model_batch = SequenceBatch(obs=obs, action=action, reward=reward, task=task, next_z=next_z)
        losses, posterior = world_model.model_loss(model_batch)

        # posterior["z"] is live (graph-attached) -- the
        # critic's value loss reads it directly so its gradient reaches the
        # world model's dynamics/encoder too, matching upstream's single
        # shared graph, with no second rollout call needed.
        qs = self.policy.Q(posterior["z"][:-1], action, task, return_type="all")
        value_loss = self._value_loss(qs, td_targets)

        total_loss = (
            self.consistency_coef * losses["consistency_loss"]
            + self.reward_coef * losses["reward_loss"]
            + self.value_coef * value_loss
        )
        grad_norm = self._step_world_optimizer(total_loss)

        metrics = {
            "consistency_loss": float(losses["consistency_loss"].detach()),
            "reward_loss": float(losses["reward_loss"].detach()),
            "value_loss": float(value_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm),
        }
        return metrics, posterior

    def _update_actor_critic(self, batch, posterior: State) -> dict[str, float]:
        # .detach(): the world model was already stepped in _update_model --
        # see TDMPC2._update_actor_critic.
        return self._update_pi(posterior["z"].detach(), (batch.task,))

    def _update_targets(self) -> None:
        super()._update_targets()
        self.policy.world_model.eval()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "tasks": self.tasks,
            "obs_dims": self.obs_dims,
            "action_dims": self.action_dims,
            "episode_lengths": self.episode_lengths,
            "task_dim": self.task_dim,
            "tau": self.tau,
            "enc_dim": self.enc_dim,
            "num_enc_layers": self.num_enc_layers,
        }
