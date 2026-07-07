"""On-policy training loop for ManiSkill GPU-parallel environments."""

from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.buffers.rollout_buffer import RolloutBuffer
from rl_garden.common.logger import Logger


class OnPolicyAlgorithm(BaseAlgorithm):
    """Base class for PPO-style rollout/update algorithms."""

    rollout_buffer: RolloutBuffer

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        num_steps: int = 50,
        gamma: float = 0.8,
        gae_lambda: float = 0.9,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env, eval_env=eval_env, seed=seed, device=device, logger=logger
        )
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.std_log = std_log
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.num_eval_steps = num_eval_steps
        self.finite_horizon_gae = finite_horizon_gae
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.save_final_checkpoint = save_final_checkpoint
        self._last_checkpoint_step = -1
        self.num_envs = env.num_envs
        self.batch_size = self.num_steps * self.num_envs

    @abstractmethod
    def train(self) -> dict[str, float]: ...

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("policy_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "num_steps": self.num_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "finite_horizon_gae": self.finite_horizon_gae,
        }

    def _checkpoint_path(self, name: str) -> Path:
        assert self.checkpoint_dir is not None
        return Path(self.checkpoint_dir) / name

    def _save_checkpoint(self, name: str) -> None:
        if self.checkpoint_dir is not None:
            self.save(self._checkpoint_path(name), include_replay_buffer=False)

    def _maybe_save_periodic_checkpoint(self, previous_step: int) -> None:
        if self.checkpoint_dir is None or self.checkpoint_freq <= 0:
            return
        if (
            self._global_step // self.checkpoint_freq
            <= previous_step // self.checkpoint_freq
        ):
            return
        if self._global_step == self._last_checkpoint_step:
            return
        self._save_checkpoint(f"checkpoint_{self._global_step}.pt")
        self._last_checkpoint_step = self._global_step

    @staticmethod
    def _first_metric(metrics: dict[str, float], keys: tuple[str, ...]) -> float:
        for key in keys:
            if key in metrics:
                return float(metrics[key])
        return float("nan")

    @staticmethod
    def _fmt_metric(value: float) -> str:
        return "nan" if value != value else f"{value:.4f}"

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in metrics.items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _log_rollout_metric(self, key: str, value: float, step: int) -> None:
        if self.logger is not None:
            self.logger.add_scalar(f"train/{key}", value, step)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in metrics.items():
            self.logger.add_scalar(f"losses/{key}", value, step)

    def _rollout_policy(
        self, obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.policy(
                self._obs_to_policy_device(obs),
                deterministic=False,
                stop_gradient_actor=self._actor_stop_gradient(),
            )

    def _actor_stop_gradient(self) -> bool:
        return False

    def _eval_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.predict(
                self._obs_to_policy_device(obs), deterministic=True
            )

    def _env_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.policy.clamp_action(raw_action).detach()

    def _compute_final_values(self, infos, done_mask: torch.Tensor, hidden) -> torch.Tensor:
        """Bootstrap value for envs that just finished this step.

        ``hidden`` is unused by this stateless default; recurrent subclasses
        override this method to bootstrap from the POST-step, UNMASKED hidden
        state (``final_observation`` continues the same episode, it is not a
        fresh start).
        """
        final_values = torch.zeros(self.num_envs, device=self.device)
        if "final_observation" not in infos or not done_mask.any():
            return final_values
        final_obs = infos["final_observation"]
        if isinstance(final_obs, dict):
            final_obs = {k: v[done_mask] for k, v in final_obs.items()}
        else:
            final_obs = final_obs[done_mask]
        with torch.no_grad():
            values = self.policy.predict_values(
                self._obs_to_policy_device(final_obs)
            ).view(-1)
        final_values[done_mask] = values
        return final_values

    # ------------------------------------------------------------------
    # Recurrent-state hooks. Trivial, hidden-agnostic defaults so subclasses
    # that never carry a recurrent hidden state are completely unaffected;
    # recurrent subclasses (e.g. RecurrentPPO) override these directly rather
    # than the base class branching on any "is this recurrent" flag.
    # ------------------------------------------------------------------

    def _initial_hidden_state(self, batch_size: int):
        """Opaque recurrent state carried across rollout steps, or None for
        stateless policies."""
        return None

    def _snapshot_window_initial_hidden(self, hidden, next_done: torch.Tensor):
        """Hidden-state snapshot to use as the BPTT initial state for the
        upcoming rollout window. Default: pass through unchanged (irrelevant
        when hidden is None)."""
        return hidden

    def _rollout_step(self, obs, hidden, episode_starts: torch.Tensor):
        """Single rollout step. Returns (actions, values, log_probs, entropy,
        new_hidden). Default: stateless _rollout_policy(obs), hidden passed
        through unchanged."""
        actions, values, log_probs, entropy = self._rollout_policy(obs)
        return actions, values, log_probs, entropy, hidden

    def _predict_last_values(self, obs, hidden) -> torch.Tensor:
        """Bootstrap value for GAE at the end of a rollout window. Default:
        stateless policy.predict_values(obs)."""
        return self.policy.predict_values(obs).view(-1)

    def learn(self, total_timesteps: int) -> "OnPolicyAlgorithm":
        obs, _ = self.env.reset(seed=self.seed)
        next_done = torch.zeros(self.num_envs, device=self.device)
        cumulative = defaultdict(float)
        hidden = self._initial_hidden_state(self.num_envs)

        while self._global_step < total_timesteps:
            previous_step = self._global_step
            if self.eval_freq > 0 and self._global_update % self.eval_freq == 0:
                stime = time.perf_counter()
                eval_metrics = self._evaluate()
                if self.logger is not None:
                    self._log_eval_metrics(eval_metrics, self._global_step)
                    self.logger.add_scalar(
                        "time/eval_time", time.perf_counter() - stime, self._global_step
                    )
                if self.std_log:
                    eval_return = self._first_metric(eval_metrics, ("return",))
                    eval_success = self._first_metric(
                        eval_metrics, ("success_at_end", "success_once")
                    )
                    print(
                        "[eval] "
                        f"step={self._global_step}/{total_timesteps} "
                        f"return={self._fmt_metric(eval_return)} "
                        f"success_at_end={self._fmt_metric(eval_success)}",
                        flush=True,
                    )

            self.rollout_buffer.reset()
            # Fold the episode-boundary reset from the END of the previous
            # rollout window into the snapshot BEFORE this window starts. This
            # is what lets RolloutBuffer.get_sequences() reconstruct BPTT
            # windows without needing to store dones[-1] anywhere -- see
            # rollout_buffer.py::get_sequences() docstring. No-op (hidden is
            # None) for stateless policies.
            window_initial_hidden = self._snapshot_window_initial_hidden(hidden, next_done)
            rollout_t = time.perf_counter()
            rollout_reward_sum = 0.0
            rollout_reward_count = 0
            rollout_episode_metrics: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.num_steps):
                self._global_step += self.num_envs
                actions, values, log_probs, _, hidden = self._rollout_step(
                    obs, hidden, next_done
                )
                next_obs, rewards, terminations, truncations, infos = self.env.step(
                    self._env_action(actions)
                )
                next_done = torch.logical_or(terminations, truncations).to(self.device)
                final_values = self._compute_final_values(infos, next_done.bool(), hidden)
                self.rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    next_done,
                    values,
                    log_probs,
                    final_values=final_values,
                )
                rollout_reward_sum += float(rewards.float().sum().item())
                rollout_reward_count += int(rewards.numel())

                if "final_info" in infos:
                    fi = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for k, v in fi["episode"].items():
                        done_values = v[done_mask]
                        if done_values.numel() == 0:
                            continue
                        mean_value = float(done_values.float().mean().item())
                        self._log_rollout_metric(k, mean_value, self._global_step)
                        rollout_episode_metrics[k].append(mean_value)
                obs = next_obs
            rollout_time = time.perf_counter() - rollout_t
            cumulative["rollout_time"] += rollout_time

            with torch.no_grad():
                last_values = self._predict_last_values(
                    self._obs_to_policy_device(obs), hidden
                )
            self.rollout_buffer.compute_returns_and_advantage(
                last_values,
                next_done,
                finite_horizon_gae=self.finite_horizon_gae,
            )
            self._rollout_initial_hidden = window_initial_hidden

            update_t = time.perf_counter()
            losses = self.train()
            update_time = time.perf_counter() - update_t
            cumulative["update_time"] += update_time

            should_log = (
                self.log_freq > 0
                and (self._global_step - self.batch_size) // self.log_freq
                < self._global_step // self.log_freq
            )
            if should_log:
                rollout_fps = (
                    self.batch_size / rollout_time if rollout_time > 0 else float("nan")
                )
                if self.logger is not None:
                    self._log_update_metrics(losses, self._global_step)
                    self.logger.add_scalar(
                        "time/update_time", update_time, self._global_step
                    )
                    self.logger.add_scalar(
                        "time/rollout_time", rollout_time, self._global_step
                    )
                    self.logger.add_scalar(
                        "time/rollout_fps", rollout_fps, self._global_step
                    )
                    for k, v in cumulative.items():
                        self.logger.add_scalar(f"time/total_{k}", v, self._global_step)
                    self.logger.add_scalar(
                        "time/total_rollout+update_time",
                        cumulative["rollout_time"] + cumulative["update_time"],
                        self._global_step,
                    )
                if self.std_log:
                    reward_mean = (
                        rollout_reward_sum / rollout_reward_count
                        if rollout_reward_count > 0
                        else float("nan")
                    )
                    episode_means = {
                        k: float(sum(v) / len(v))
                        for k, v in rollout_episode_metrics.items()
                        if len(v) > 0
                    }
                    rollout_return = self._first_metric(episode_means, ("return",))
                    rollout_success = self._first_metric(
                        episode_means, ("success_at_end", "success_once")
                    )
                    progress = 100.0 * self._global_step / total_timesteps
                    print(
                        "[train] "
                        f"step={self._global_step}/{total_timesteps} ({progress:.2f}%) "
                        f"reward={self._fmt_metric(reward_mean)} "
                        f"return={self._fmt_metric(rollout_return)} "
                        f"success_at_end={self._fmt_metric(rollout_success)} "
                        f"fps={self._fmt_metric(rollout_fps)}",
                        flush=True,
                    )
            self._maybe_save_periodic_checkpoint(previous_step)

        if self.checkpoint_dir is not None and self.save_final_checkpoint:
            self._save_checkpoint("final.pt")
        return self
