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

from rl_garden.algorithms.iql import _IQLRolloutTrainingShell
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.encoders.config import EncoderConfig
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.iql import IQLParams
from rl_garden.observations import ObsGroups


class Off2OnIQL(_IQLRolloutTrainingShell):
    """IQL offline pretraining + online fine-tuning, no warmup."""

    _compatible_checkpoint_algorithms = ("Off2OnIQL", "IQL")

    params_class = IQLParams

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
        bootstrap_at_done: str = "truncated",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
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
        params: Optional[IQLParams] = None,
        **overrides,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
        super().__init__(
            env=env,
            eval_env=eval_env,
            params=params,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            offline_sampling=offline_sampling,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            encoder_sharing=encoder_sharing,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            value_hidden_dims=value_hidden_dims,
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
