"""Standard FQL offline-to-online training shell.

This deliberately uses FQL's unchanged critic and flow-matching actor during
online fine-tuning.  The offline replay is retained in one append-only buffer,
matching the reference FQL continuation protocol.
"""
from __future__ import annotations

import time
from typing import Any, Literal, Optional

import torch

from rl_garden.algorithms.fql import FQLCore
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.fql import FQLReferenceParams
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups


class Off2OnFQL(Off2OnReplayMixin, FQLCore, OffPolicyAlgorithm):
    """FQL pretraining followed by uninterrupted online FQL updates."""

    _compatible_checkpoint_algorithms = ("Off2OnFQL", "FQL")

    params_class = FQLReferenceParams

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        params: Optional[FQLReferenceParams] = None,
        buffer_size: int = 1000000,
        buffer_device: str = 'cuda',
        learning_starts: int = 0,
        batch_size: int = 256,
        gamma: float = 0.99,
        training_freq: int = 1,
        utd: float = 1.0,
        bootstrap_at_done: str = 'truncated',
        actor_use_group_norm: bool = False,
        encoder_sharing: Optional[EncoderSharing] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        offline_sampling: Literal['with_replace', 'without_replace'] = 'with_replace',
        seed: int = 1,
        device: str | torch.device = 'auto',
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1000,
        eval_freq: int = 0,
        num_eval_steps: int = 50,
        num_eval_episodes: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
        initial_training_phase: Optional[InitialTrainingPhase] = None,
        **overrides,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
        super().__init__(
            env=env, eval_env=eval_env, buffer_size=buffer_size, buffer_device=buffer_device,
            learning_starts=learning_starts, batch_size=batch_size, gamma=gamma, tau=params.tau,
            training_freq=training_freq, utd=utd, bootstrap_at_done=bootstrap_at_done,
            seed=seed, device=device, logger=logger, std_log=std_log, log_freq=log_freq,
            eval_freq=eval_freq, num_eval_steps=num_eval_steps, checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq, save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint, initial_training_phase=initial_training_phase,
        )
        self._init_fql_params(
            params,
            actor_use_group_norm=actor_use_group_norm,
            alpha=params.alpha,
            critic_dropout_rate=params.critic_dropout_rate,
            critic_encoder_config=critic_encoder_config,
            critic_use_group_norm=params.critic_use_group_norm,
            encoder_config=encoder_config,
            encoder_sharing=encoder_sharing,
            num_groups=params.num_groups,
            obs_groups=obs_groups,
            reference_compatible=params.reference_compatible,
        )
        self.num_eval_episodes = num_eval_episodes
        self._init_off2on_params(offline_sampling=offline_sampling)
        self._setup_model()

    def _evaluate(self) -> dict[str, float]:
        if self.num_eval_episodes is None:
            return super()._evaluate()
        from rl_garden.algorithms.offline import run_exact_episode_eval

        metrics = run_exact_episode_eval(
            self,
            num_eval_episodes=self.num_eval_episodes,
            num_eval_steps=self.num_eval_steps,
        )
        completed = int(metrics.get("episodes_completed", 0))
        if self.reference_compatible and completed != self.num_eval_episodes:
            raise RuntimeError(
                "Reference-compatible FQL evaluation did not complete the requested "
                f"{self.num_eval_episodes} episodes before its {self.num_eval_steps}-step cap "
                f"(completed {completed})."
            )
        return metrics

    def learn(self, total_timesteps: int) -> "Off2OnFQL":
        """Continue online training and emit FQL's final fixed-cadence eval.

        ``OffPolicyAlgorithm`` evaluates a cadence boundary at the beginning
        of the following rollout iteration.  An off2on run ends immediately
        when its online budget lands on that boundary, so the final point
        would otherwise be absent.  This intentionally covers only FQL's
        fixed ``eval_freq`` recipe and does not add a shared scheduling API.
        """
        initial_step = self._global_step
        super().learn(total_timesteps)
        if (
            self.eval_env is None
            or self.eval_freq <= 0
            or initial_step >= total_timesteps
            or self._global_step != total_timesteps
            or total_timesteps % self.eval_freq != 0
        ):
            return self

        started = time.perf_counter()
        metrics = self._evaluate()
        if self.logger is not None:
            self._log_eval_metrics(metrics, self._global_step)
            self.logger.add_scalar(
                "time/eval_time", time.perf_counter() - started, self._global_step
            )
        if self.std_log:
            print(
                "[eval] "
                f"step={self._global_step}/{total_timesteps} "
                f"return={self._fmt_metric(self._first_metric(metrics, ('return',)))} "
                "success_at_end="
                f"{self._fmt_metric(self._first_metric(metrics, ('success_at_end', 'success_once')))}",
                flush=True,
            )
        return self
