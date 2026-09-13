"""IQL offline-to-online, built on the shared shell.

Confirmed against the reference WSRL/IQL JAX implementation
(``3rd_party/wsrl/wsrl/agents/iql.py`` + ``finetune.py``): IQL needs no
algorithm-specific override at the offline->online switch (unlike Cal-QL's
CQL-alpha swap) and uses a plain (non-MC) replay buffer, so ``Off2OnIQL``
adds no behavior on top of ``_IQLRolloutTrainingShell`` beyond a
construction-time preset matching ``Off2OnCalQL``: no warmup, offline data
retained and mixed throughout online fine-tuning by default.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch

from rl_garden.algorithms._observation import EncoderSharing
from rl_garden.algorithms.iql import _IQLRolloutTrainingShell
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups


class Off2OnIQL(_IQLRolloutTrainingShell):
    """IQL offline pretraining + online fine-tuning, no warmup."""

    _compatible_checkpoint_algorithms = ("Off2OnIQL", "IQL")

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
        tau: float = 0.005,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "truncated",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        actor_lr: float = 3e-4,
        critic_value_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        actor_lr_schedule: Optional[
            Literal["constant", "linear_warmup", "warmup_cosine"]
        ] = None,
        actor_lr_warmup_steps: Optional[int] = None,
        actor_lr_decay_steps: Optional[int] = None,
        actor_lr_min_ratio: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
        adv_clip_max: float = 100.0,
        actor_distribution: Literal["squashed", "unsquashed"] = "squashed",
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
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
            actor_lr=actor_lr,
            critic_value_lr=critic_value_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            actor_lr_schedule=actor_lr_schedule,
            actor_lr_warmup_steps=actor_lr_warmup_steps,
            actor_lr_decay_steps=actor_lr_decay_steps,
            actor_lr_min_ratio=actor_lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            expectile=expectile,
            temperature=temperature,
            adv_clip_max=adv_clip_max,
            actor_distribution=actor_distribution,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            value_hidden_dims=value_hidden_dims,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            encoder_sharing=encoder_sharing,
            policy_kwargs=policy_kwargs,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            value_use_layer_norm=value_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            value_use_group_norm=value_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            value_dropout_rate=value_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
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
