"""Original Cal-QL paper's own offline-to-online design, built on the shared shell.

``Off2OnCalQL`` adds no behavior on top of ``_CalQLRolloutTrainingShell``
beyond a construction-time preset: no warmup by default (unlike ``WSRL``),
offline data retained and mixed throughout online fine-tuning by default, and
the CQL/Cal-QL regularizer retained online by default. See Nakamoto et al.
2023. ``initial_training_phase`` is still accepted (for interface parity with
the shared off2on runner) but left unused by the ``calql`` CLI preset.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch

from rl_garden.algorithms.calql import _CalQLRolloutTrainingShell
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.encoders.config import EncoderConfig
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.cql import Off2OnCalQLParams
from rl_garden.observations import ObsGroups
from rl_garden.policies.sac_policy import SACActorFactory


class Off2OnCalQL(_CalQLRolloutTrainingShell):
    """Cal-QL offline pretraining + online fine-tuning, no warmup."""

    params_class = Off2OnCalQLParams

    _compatible_checkpoint_algorithms = ("Off2OnCalQL", "WSRL", "CalQL", "CQL")

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        buffer_size: int = 1000000,
        buffer_device: str = 'cuda',
        learning_starts: int = 4000,
        batch_size: int = 256,
        gamma: float = 0.99,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = 'truncated',
        online_episodes_per_iteration: Optional[int] = None,
        stats_window_size: Optional[int] = None,
        use_compile: bool = False,
        compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'default',
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        actor_feature_dim: Optional[int] = None,
        critic_spatial_emb_dim: int = 1024,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        offline_sampling: Literal['with_replace', 'without_replace'] = 'with_replace',
        policy_kwargs: Optional[dict[str, Any]] = None,
        actor_factory: Optional[SACActorFactory] = None,
        seed: int = 1,
        device: str | torch.device = 'auto',
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        num_eval_episodes: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
        initial_training_phase: Optional[InitialTrainingPhase] = None,
        params: Off2OnCalQLParams | None = None,
        **overrides: Any,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config
        self.encoder_sharing = encoder_sharing

        super().__init__(
            params=params,
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
            online_episodes_per_iteration=online_episodes_per_iteration,
            stats_window_size=stats_window_size,
            use_compile=use_compile,
            compile_mode=compile_mode,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_feature_dim=actor_feature_dim,
            critic_spatial_emb_dim=critic_spatial_emb_dim,
            offline_sampling=offline_sampling,
            policy_kwargs=policy_kwargs,
            actor_factory=actor_factory,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            num_eval_episodes=num_eval_episodes,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
            initial_training_phase=initial_training_phase,
        )
