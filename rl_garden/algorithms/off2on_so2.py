"""SO2 offline-to-online, built on the shared shell.

``Off2OnSO2`` adds no behavior on top of ``_SO2RolloutTrainingShell`` beyond
a construction-time preset: offline data retained and mixed throughout
online fine-tuning by default (matching upstream's ``concat_online_ratio``
mixed-batch scheme -- see ``training/off2on/so2.py`` for the CLI preset).

Only Box-observation, non-nstep SAC kwargs are exposed here (no
image/vision/mmap kwargs) -- matches this port's Box-only scope (see
``so2.py``'s module docstring).
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch

from rl_garden.algorithms.so2 import _SO2RolloutTrainingShell
from rl_garden.common.alpha_tuning import AlphaTuning
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType
from rl_garden.common.training_phase import InitialTrainingPhase


class Off2OnSO2(_SO2RolloutTrainingShell):
    """SO2 offline pretraining + online fine-tuning."""

    _compatible_checkpoint_algorithms = ("Off2OnSO2", "SO2")

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 1024,
        gamma: float = 0.99,
        tau: float = 0.005,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "always",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        alpha_lr: Optional[float] = None,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        alpha_tuning: AlphaTuning = "legacy_exp",
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = None,
        backup_entropy: bool = True,
        critic_impl: Literal["vmap", "legacy"] = "vmap",
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_log_std_min: float = -5.0,
        actor_log_std_mode: Literal["clamp", "tanh"] = "clamp",
        critic_backbone_type: Optional[Literal["mlp", "mlp_resnet"]] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        target_smoothing_noise_std: float = 0.3,
        target_smoothing_noise_clip_min: float = -0.6,
        target_smoothing_noise_clip_max: float = 0.6,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        num_eval_episodes: Optional[int] = None,
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
            tau=tau,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            offline_sampling=offline_sampling,
            policy_lr=policy_lr,
            q_lr=q_lr,
            alpha_lr=alpha_lr,
            policy_frequency=policy_frequency,
            target_network_frequency=target_network_frequency,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            alpha_tuning=alpha_tuning,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            backup_entropy=backup_entropy,
            critic_impl=critic_impl,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_log_std_min=actor_log_std_min,
            actor_log_std_mode=actor_log_std_mode,
            critic_backbone_type=critic_backbone_type,
            policy_kwargs=policy_kwargs,
            target_smoothing_noise_std=target_smoothing_noise_std,
            target_smoothing_noise_clip_min=target_smoothing_noise_clip_min,
            target_smoothing_noise_clip_max=target_smoothing_noise_clip_max,
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
        self.num_eval_episodes = num_eval_episodes

    def _evaluate(self) -> dict[str, float]:
        if self.num_eval_episodes is None:
            return super()._evaluate()
        from rl_garden.algorithms.offline import run_exact_episode_eval

        return run_exact_episode_eval(
            self,
            num_eval_episodes=self.num_eval_episodes,
            num_eval_steps=self.num_eval_steps,
        )
