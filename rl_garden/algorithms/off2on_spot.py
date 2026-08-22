"""SPOT offline-to-online, built on the shared shell.

Unlike ``Off2OnIQL``/``Off2OnAWAC`` (no algorithm-specific change at the
offline->online switch), ``Off2OnSPOT`` inherits
``_SPOTRolloutTrainingShell``'s override of ``_apply_online_regularizer_override``
(optimizer reset + discount swap, reproducing ``spot.py``'s "Resetting
optimizers" block) and ``_rollout_action`` (TD3-style exploration noise) --
both are shell-level behavior, so this class adds only a construction-time
preset, matching ``Off2OnIQL``'s shape.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch

from rl_garden.algorithms.spot import _SPOTRolloutTrainingShell
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.networks import KernelInit
from rl_garden.networks.actor_critic import BackboneType


class Off2OnSPOT(_SPOTRolloutTrainingShell):
    """SPOT offline pretraining + online fine-tuning."""

    _compatible_checkpoint_algorithms = ("Off2OnSPOT", "SPOT")

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "always",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: ScheduleType = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        vae_lr: float = 1e-3,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
        vae_iterations: int = 100_000,
        beta: float = 0.5,
        lambd: float = 1.0,
        num_samples: int = 1,
        iwae: bool = False,
        lambd_cool: bool = False,
        lambd_end: float = 0.2,
        expl_noise: float = 0.1,
        online_discount: float = 0.995,
        max_online_updates: int = 1_000_000,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
        initial_training_phase: Optional[InitialTrainingPhase] = None,
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            offline_sampling=offline_sampling,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            vae_lr=vae_lr,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
            vae_iterations=vae_iterations,
            beta=beta,
            lambd=lambd,
            num_samples=num_samples,
            iwae=iwae,
            lambd_cool=lambd_cool,
            lambd_end=lambd_end,
            expl_noise=expl_noise,
            online_discount=online_discount,
            max_online_updates=max_online_updates,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
            initial_training_phase=initial_training_phase,
        )
