"""WSRL warm-start flow built on Cal-QL.

WSRL's distinguishing behavior on top of ``_CalQLRolloutTrainingShell`` (which
owns the generic off2on Cal-QL machinery: mixed/empty/append replay switching,
online CQL-alpha override, checkpoint/probe/logging plumbing) is a warmup
phase: freeze updates for a fixed number of steps, then discard the offline
replay data entirely and continue as SAC (no CQL conservatism retained by
default). See ``Off2OnCalQL`` for the original Cal-QL paper's own off2on
preset (no warmup, offline data retained and mixed throughout).
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
from rl_garden.hyperparameters.cql import WSRLParams
from rl_garden.observations import ObsGroups
from rl_garden.policies.sac_policy import SACActorFactory


class WSRL(_CalQLRolloutTrainingShell):
    """WSRL warm-start: Cal-QL offline pretraining + warmup + online SAC."""

    params_class = WSRLParams

    _compatible_checkpoint_algorithms = ("WSRL", "CalQL", "CQL")

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
        seed: int = 1,
        device: str | torch.device = 'auto',
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
        initial_training_phase: Optional[InitialTrainingPhase] = None,
        actor_factory: Optional[SACActorFactory] = None,
        params: WSRLParams | None = None,
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
            use_compile=use_compile,
            compile_mode=compile_mode,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_feature_dim=actor_feature_dim,
            critic_spatial_emb_dim=critic_spatial_emb_dim,
            offline_sampling=offline_sampling,
            policy_kwargs=policy_kwargs,
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
            actor_factory=actor_factory,
        )

    @property
    def training_stage(self) -> str:
        if not self.use_cql_loss:
            objective = "SAC"
        else:
            objective = "Cal-QL" if self.use_calql else "CQL"
        return f"{super().training_stage} ({objective})"

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_training_state_dict(state)
        if not state:
            # Legacy WSRL checkpoints start online warmup only when switched.
            self._initial_phase_start_step = None
