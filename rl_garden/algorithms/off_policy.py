"""Off-policy training loop targeting ManiSkill's GPU-parallel envs.

Structurally based on ``examples/baselines/sac/sac.py``'s ``while`` loop
(L388-L552), rewritten as an abstract base that SAC subclasses fill in by
implementing ``train(gradient_steps)``. Unlike SB3's ``OffPolicyAlgorithm``,
this loop never touches numpy in the hot path. The primary path is ManiSkill
GPU training where rollouts, buffer, and updates stay on CUDA tensors; CPU
observations are only supported as a compatibility fallback for CPU-backed envs.
"""
from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import (
    InitialTrainingPhase,
    STANDARD_UPDATE_MASK,
    TrainingUpdateMask,
)


class OffPolicyAlgorithm(BaseAlgorithm):
    replay_buffer: BaseReplayBuffer
    num_envs: int

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 512,
        gamma: float = 0.8,
        tau: float = 0.01,
        training_freq: int = 64,
        utd: float = 0.5,
        bootstrap_at_done: str = "always",
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
        super().__init__(env=env, eval_env=eval_env, seed=seed, device=device, logger=logger)
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.training_freq = training_freq
        self.utd = utd
        assert bootstrap_at_done in ("always", "never", "truncated"), bootstrap_at_done
        self.bootstrap_at_done = bootstrap_at_done
        self.std_log = std_log
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.num_eval_steps = num_eval_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.save_replay_buffer = save_replay_buffer
        self.save_final_checkpoint = save_final_checkpoint
        self._last_checkpoint_step = -1
        self.initial_training_phase = initial_training_phase
        self._initial_phase_start_step: Optional[int] = None

        self.num_envs = env.num_envs
        self.steps_per_env = max(1, training_freq // self.num_envs)
        self.grad_steps_per_iteration = max(1, int(training_freq * utd))

    # --- subclass hooks ---

    @abstractmethod
    def _setup_model(self) -> None:
        """Build policy, optimizers, replay buffer, etc."""

    @abstractmethod
    def train(
        self, gradient_steps: int, compute_info: bool = False
    ) -> dict[str, float]: ...

    # --- logging hooks ---

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in metrics.items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _log_rollout_metric(self, key: str, value: float, step: int) -> None:
        if self.logger is None:
            return
        self.logger.add_scalar(f"train/{key}", value, step)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        self.logger.log_metrics(metrics, step)

    def _explore_action(self, obs) -> torch.Tensor:
        """Random uniform action in [-1, 1] across all envs. Used pre-learning."""
        shape = self.env.action_space.shape
        return 2 * torch.rand(shape, dtype=torch.float32, device=self.device) - 1

    def _policy_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.predict(
                self._obs_to_policy_device(obs), deterministic=False
            ).detach()

    def _eval_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.predict(
                self._obs_to_policy_device(obs), deterministic=True
            )

    def _eval_action_and_critic_action(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        action = self._eval_action(obs)
        return action, action

    def _eval_q_mc_enabled(self) -> bool:
        return False

    def _eval_q_values(self, obs, actions) -> torch.Tensor:
        raise NotImplementedError

    def _on_env_reset(self, obs) -> None:
        del obs

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        phase = self._active_initial_training_phase()
        if phase is not None:
            actions = self._policy_action(obs)
            if phase.random_action_prob > 0.0:
                random_actions = self._explore_action(obs)
                mask_shape = (actions.shape[0],) + (1,) * (actions.ndim - 1)
                random_mask = (
                    torch.rand(mask_shape, device=actions.device)
                    < phase.random_action_prob
                )
                actions = torch.where(random_mask, random_actions, actions)
            return actions, actions, None
        if not learning_has_started:
            actions = self._explore_action(obs)
        else:
            actions = self._policy_action(obs)
        return actions, actions, None

    def _on_training_start(self, total_timesteps: int) -> None:
        super()._on_training_start(total_timesteps)
        if self._should_start_initial_training_phase_on_learn():
            self._start_initial_training_phase()

    def _should_start_initial_training_phase_on_learn(self) -> bool:
        return True

    def _start_initial_training_phase(
        self, start_step: Optional[int] = None
    ) -> None:
        if self.initial_training_phase is None:
            return
        if self._initial_phase_start_step is None:
            self._initial_phase_start_step = (
                self._global_step if start_step is None else int(start_step)
            )

    def _active_initial_training_phase(self) -> Optional[InitialTrainingPhase]:
        phase = self.initial_training_phase
        start = self._initial_phase_start_step
        if phase is None or phase.duration_steps == 0 or start is None:
            return None
        if self._global_step - start >= phase.duration_steps:
            return None
        return phase

    def _training_update_mask(self) -> TrainingUpdateMask:
        phase = self._active_initial_training_phase()
        return phase.update_mask if phase is not None else STANDARD_UPDATE_MASK

    def _replay_buffer_add_kwargs(
        self,
        action_context: Optional[dict[str, Any]],
        obs,
        next_obs,
        real_next_obs,
        infos,
        need_final_obs: torch.Tensor,
    ) -> dict[str, Any]:
        del action_context, obs, next_obs, real_next_obs, infos, need_final_obs
        return {}

    def _replay_buffer_step_kwargs(
        self,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
    ) -> dict[str, Any]:
        del terminations, truncations
        return {}

    def _post_rollout_step(
        self,
        action_context: Optional[dict[str, Any]],
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos,
    ) -> None:
        del action_context, terminations, truncations, infos

    # --- bootstrap bookkeeping ---

    def _compute_done_masks(
        self, terminations: torch.Tensor, truncations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bootstrap_at_done == "never":
            need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
            stop_bootstrap = truncations | terminations
        elif self.bootstrap_at_done == "always":
            need_final_obs = truncations | terminations
            stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
        else:  # "truncated"
            need_final_obs = truncations & (~terminations)
            stop_bootstrap = terminations
        return need_final_obs, stop_bootstrap

    @staticmethod
    def _clone_obs(obs):
        if isinstance(obs, dict):
            return {k: v.clone() for k, v in obs.items()}
        return obs.clone()

    def _obs_to_policy_device(self, obs):
        """Move CPU-backed env observations to the policy device for inference.

        GPU ManiSkill envs already return tensors on ``self.device`` and this is
        a no-op. This fallback exists for CPU simulator backends such as
        ``physx_cpu``; model training and replay sampling remain CUDA-first.
        """
        if isinstance(obs, dict):
            return {
                k: v if v.device == self.device else v.to(self.device)
                for k, v in obs.items()
            }
        if obs.device == self.device:
            return obs
        return obs.to(self.device)

    @staticmethod
    def _write_final_obs(real_next_obs, infos, need_final_obs):
        if "final_observation" not in infos:
            return
        final = infos["final_observation"]
        if isinstance(real_next_obs, dict):
            for k in real_next_obs.keys():
                real_next_obs[k][need_final_obs] = final[k][need_final_obs].clone()
        else:
            real_next_obs[need_final_obs] = final[need_final_obs]

    @staticmethod
    def _first_metric(metrics: dict[str, float], keys: tuple[str, ...]) -> float:
        for key in keys:
            if key in metrics:
                return float(metrics[key])
        return float("nan")

    @staticmethod
    def _fmt_metric(value: float) -> str:
        return "nan" if value != value else f"{value:.4f}"

    # --- checkpointing ---

    def _checkpoint_metadata(self) -> dict[str, Any]:
        metadata = {
            **super()._checkpoint_metadata(),
            "buffer_size": self.buffer_size,
            "buffer_device": self.buffer_device,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "training_freq": self.training_freq,
            "utd": self.utd,
            "bootstrap_at_done": self.bootstrap_at_done,
        }
        if self.initial_training_phase is not None:
            metadata["initial_training_phase"] = (
                self.initial_training_phase.to_dict()
            )
        return metadata

    def _training_state_dict(self) -> dict[str, Any]:
        state = super()._training_state_dict()
        state["initial_phase_start_step"] = self._initial_phase_start_step
        return state

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_training_state_dict(state)
        if "initial_phase_start_step" in state:
            start = state["initial_phase_start_step"]
            self._initial_phase_start_step = None if start is None else int(start)
        elif self.initial_training_phase is not None:
            # Pre-phase-state checkpoints used absolute global-step thresholds.
            self._initial_phase_start_step = 0

    def _checkpoint_path(self, name: str) -> Path:
        assert self.checkpoint_dir is not None
        return Path(self.checkpoint_dir) / name

    def _save_checkpoint(self, name: str) -> None:
        if self.checkpoint_dir is None:
            return
        self.save(
            self._checkpoint_path(name),
            include_replay_buffer=self.save_replay_buffer,
        )

    def _maybe_save_periodic_checkpoint(self, previous_step: int) -> None:
        if self.checkpoint_dir is None or self.checkpoint_freq <= 0:
            return
        if self._global_step // self.checkpoint_freq <= previous_step // self.checkpoint_freq:
            return
        if self._global_step == self._last_checkpoint_step:
            return
        self._save_checkpoint(f"checkpoint_{self._global_step}.pt")
        self._last_checkpoint_step = self._global_step

    # --- evaluation ---

    def _finish_eval_q_mc_episode(
        self,
        q_values: list[torch.Tensor],
        rewards: list[torch.Tensor],
        completed_q_values: list[torch.Tensor],
        completed_mc_returns: list[torch.Tensor],
        completed_errors: list[torch.Tensor],
    ) -> None:
        q_ep = torch.stack(q_values).reshape(-1)
        rewards_ep = torch.stack(rewards).reshape(-1).to(q_ep.device)
        returns = torch.empty_like(rewards_ep)
        running = torch.zeros((), dtype=rewards_ep.dtype, device=rewards_ep.device)
        for idx in range(rewards_ep.shape[0] - 1, -1, -1):
            running = rewards_ep[idx] + self.gamma * running
            returns[idx] = running
        errors = q_ep - returns
        completed_q_values.append(q_ep)
        completed_mc_returns.append(returns)
        completed_errors.append(errors)

    @staticmethod
    def _reduce_eval_q_mc_metrics(
        completed_q_values: list[torch.Tensor],
        completed_mc_returns: list[torch.Tensor],
        completed_errors: list[torch.Tensor],
    ) -> dict[str, float]:
        if not completed_errors:
            return {}
        q_values = torch.cat(completed_q_values)
        mc_returns = torch.cat(completed_mc_returns)
        errors = torch.cat(completed_errors)
        mse = errors.square().mean()
        return {
            "q_mc/mean_error": float(errors.mean().item()),
            "q_mc/abs_error": float(errors.abs().mean().item()),
            "q_mc/rmse": float(torch.sqrt(mse).item()),
            "q_mc/q_mean": float(q_values.mean().item()),
            "q_mc/mc_return_mean": float(mc_returns.mean().item()),
            "q_mc/num_steps": float(errors.numel()),
        }

    def _evaluate(self) -> dict[str, float]:
        if self.eval_env is None:
            return {}
        self.policy.eval()
        obs, _ = self.eval_env.reset()
        metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        q_mc_enabled = self._eval_q_mc_enabled()
        num_eval_envs = self.eval_env.num_envs
        pending_q_values: list[list[torch.Tensor]] = [[] for _ in range(num_eval_envs)]
        pending_rewards: list[list[torch.Tensor]] = [[] for _ in range(num_eval_envs)]
        completed_q_values: list[torch.Tensor] = []
        completed_mc_returns: list[torch.Tensor] = []
        completed_errors: list[torch.Tensor] = []
        for _ in range(self.num_eval_steps):
            with torch.no_grad():
                env_action, critic_action = self._eval_action_and_critic_action(obs)
                q_values: torch.Tensor | None = None
                if q_mc_enabled:
                    q_values = self._eval_q_values(
                        self._obs_to_policy_device(obs), critic_action
                    ).reshape(-1).detach()
                obs, rewards, terminations, truncations, infos = self.eval_env.step(
                    env_action
                )
                if q_mc_enabled:
                    assert q_values is not None
                    step_rewards = rewards.reshape(-1).to(q_values.device).detach()
                    dones = (terminations | truncations).reshape(-1)
                    for env_idx in range(num_eval_envs):
                        pending_q_values[env_idx].append(q_values[env_idx])
                        pending_rewards[env_idx].append(step_rewards[env_idx])
                    for env_idx in torch.where(dones)[0].detach().cpu().tolist():
                        self._finish_eval_q_mc_episode(
                            pending_q_values[env_idx],
                            pending_rewards[env_idx],
                            completed_q_values,
                            completed_mc_returns,
                            completed_errors,
                        )
                        pending_q_values[env_idx] = []
                        pending_rewards[env_idx] = []
                if "final_info" in infos:
                    for k, v in infos["final_info"]["episode"].items():
                        metrics[k].append(v)
        self.policy.train()
        out: dict[str, float] = {}
        for k, vs in metrics.items():
            out[k] = float(torch.stack(vs).float().mean().item())
        if q_mc_enabled:
            out.update(
                self._reduce_eval_q_mc_metrics(
                    completed_q_values, completed_mc_returns, completed_errors
                )
            )
        return out

    # --- main loop ---

    def learn(self, total_timesteps: int) -> "OffPolicyAlgorithm":
        self._on_training_start(total_timesteps)
        obs, _ = self.env.reset(seed=self.seed)
        self._on_env_reset(obs)
        learning_has_started = False
        cumulative = defaultdict(float)
        global_steps_per_iteration = self.num_envs * self.steps_per_env

        while self._global_step < total_timesteps:
            previous_step = self._global_step
            # Eval at iteration boundary.
            if (
                self.eval_freq > 0
                and (self._global_step - self.training_freq) // self.eval_freq
                < self._global_step // self.eval_freq
            ):
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

            # Rollout.
            rollout_t = time.perf_counter()
            rollout_reward_sum = 0.0
            rollout_reward_count = 0
            rollout_episode_metrics: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.steps_per_env):
                self._global_step += self.num_envs
                actions, env_actions, action_context = self._rollout_action(
                    obs, learning_has_started
                )

                next_obs, rewards, terminations, truncations, infos = self.env.step(
                    env_actions
                )
                rollout_reward_sum += float(rewards.float().sum().item())
                rollout_reward_count += int(rewards.numel())
                real_next_obs = self._clone_obs(next_obs)
                need_final_obs, stop_bootstrap = self._compute_done_masks(
                    terminations, truncations
                )
                self._write_final_obs(real_next_obs, infos, need_final_obs)

                if "final_info" in infos and self.logger is not None:
                    fi = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for k, v in fi["episode"].items():
                        done_values = v[done_mask]
                        if done_values.numel() == 0:
                            continue
                        mean_value = float(done_values.float().mean().item())
                        self._log_rollout_metric(k, mean_value, self._global_step)
                        rollout_episode_metrics[k].append(mean_value)
                elif "final_info" in infos:
                    fi = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for k, v in fi["episode"].items():
                        done_values = v[done_mask]
                        if done_values.numel() == 0:
                            continue
                        rollout_episode_metrics[k].append(
                            float(done_values.float().mean().item())
                        )

                replay_kwargs = self._replay_buffer_add_kwargs(
                    action_context,
                    obs,
                    next_obs,
                    real_next_obs,
                    infos,
                    need_final_obs,
                )
                replay_kwargs.update(
                    self._replay_buffer_step_kwargs(terminations, truncations)
                )
                self.replay_buffer.add(
                    obs, real_next_obs, actions, rewards, stop_bootstrap, **replay_kwargs
                )
                self._post_rollout_step(action_context, terminations, truncations, infos)
                obs = next_obs
            rollout_time = time.perf_counter() - rollout_t
            cumulative["rollout_time"] += rollout_time
            rollout_reward_mean = (
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
            should_log = (
                self.log_freq > 0
                and (self._global_step - self.training_freq) // self.log_freq
                < self._global_step // self.log_freq
            )
            rollout_fps = (
                global_steps_per_iteration / rollout_time
                if rollout_time > 0
                else float("nan")
            )

            if self._global_step < self.learning_starts:
                self._maybe_save_periodic_checkpoint(previous_step)
                if self.std_log and should_log:
                    progress = 100.0 * self._global_step / total_timesteps
                    print(
                        "[train] "
                        f"step={self._global_step}/{total_timesteps} ({progress:.2f}%) "
                        "phase=warmup "
                        f"reward={self._fmt_metric(rollout_reward_mean)} "
                        f"return={self._fmt_metric(rollout_return)} "
                        f"success_at_end={self._fmt_metric(rollout_success)} "
                        f"fps={self._fmt_metric(rollout_fps)}",
                        flush=True,
                    )
                continue
            learning_has_started = True

            # Update.
            update_t = time.perf_counter()
            losses = self.train(self.grad_steps_per_iteration, compute_info=should_log)
            update_time = time.perf_counter() - update_t
            cumulative["update_time"] += update_time

            if should_log:
                if self.logger is not None:
                    self._log_update_metrics(losses, self._global_step)
                    self.logger.add_scalar("time/update_time", update_time, self._global_step)
                    self.logger.add_scalar("time/rollout_time", rollout_time, self._global_step)
                    self.logger.add_scalar("time/rollout_fps", rollout_fps, self._global_step)
                    for k, v in cumulative.items():
                        self.logger.add_scalar(f"time/total_{k}", v, self._global_step)
                    self.logger.add_scalar(
                        "time/total_rollout+update_time",
                        cumulative["rollout_time"] + cumulative["update_time"],
                        self._global_step,
                    )
                if self.std_log:
                    progress = 100.0 * self._global_step / total_timesteps
                    print(
                        "[train] "
                        f"step={self._global_step}/{total_timesteps} ({progress:.2f}%) "
                        f"reward={self._fmt_metric(rollout_reward_mean)} "
                        f"return={self._fmt_metric(rollout_return)} "
                        f"success_at_end={self._fmt_metric(rollout_success)} "
                        f"fps={self._fmt_metric(rollout_fps)}",
                        flush=True,
                    )

            self._maybe_save_periodic_checkpoint(previous_step)

        if self.checkpoint_dir is not None and self.save_final_checkpoint:
            self._save_checkpoint("final.pt")

        return self
