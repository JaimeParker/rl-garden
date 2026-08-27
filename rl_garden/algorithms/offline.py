"""Offline-only algorithm utilities.

This module is intentionally separate from ``OffPolicyAlgorithm``. Pure offline
algorithms such as IQL/BC should not inherit ManiSkill rollout logic, while
warm-start algorithms such as WSRL may still reuse the offline runner here for
pretraining entrypoints.
"""

from __future__ import annotations

import time
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from gymnasium import spaces
from tqdm import trange

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.buffers.h5_dataset import (
    infer_box_specs_from_h5 as _infer_box_specs_from_h5,
    infer_specs_from_h5 as _infer_specs_from_h5,
)
from rl_garden.common.eval_metrics import EVAL_METRIC_ALIASES, EVAL_METRIC_DROP
from rl_garden.common.logger import Logger


@dataclass
class OfflinePretrainResult:
    """Result returned by :func:`run_offline_pretraining`."""

    final_step: int
    final_update: int
    last_metrics: dict[str, float]
    final_checkpoint: Optional[Path] = None


class OfflineEnvSpec:
    """Minimal env-like object for offline-only algorithms.

    It exposes the space and ``num_envs`` attributes that algorithms and
    checkpoint metadata need, but deliberately has no ``reset`` or ``step``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        num_envs: int = 1,
    ) -> None:
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs


class OfflineRLAlgorithm(BaseAlgorithm):
    """Base class for pure offline RL algorithms.

    Subclasses own their policy/networks/replay buffer and implement
    ``train(gradient_steps)``. ``learn(total_timesteps)`` means offline update
    steps, not environment interaction steps.
    """

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        offline_sampling: str = "with_replace",
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 0,
        num_eval_steps: Optional[int] = 50,
        num_eval_episodes: int = 100,
        eval_env: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env, eval_env=eval_env, seed=seed, device=device, logger=logger
        )
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.batch_size = batch_size
        self.gamma = gamma
        self.offline_sampling = offline_sampling
        self.std_log = std_log
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.num_eval_steps = 50 if num_eval_steps is None else int(num_eval_steps)
        self.num_eval_episodes = int(num_eval_episodes)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.save_replay_buffer = save_replay_buffer
        self.save_final_checkpoint = save_final_checkpoint
        self.num_envs = env.num_envs

    @abstractmethod
    def train(
        self, gradient_steps: int, compute_info: bool = False
    ) -> dict[str, float]: ...

    def learn(self, total_timesteps: int) -> "OfflineRLAlgorithm":
        self.learn_offline(total_timesteps)
        return self

    def learn_offline(
        self,
        num_steps: int,
        *,
        gradient_steps: Optional[int] = None,
        save_filename: str = "offline_pretrained.pt",
    ) -> OfflinePretrainResult:
        return run_offline_pretraining(
            self,
            num_steps=num_steps,
            gradient_steps=gradient_steps,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_freq=self.checkpoint_freq,
            save_filename=save_filename,
            save_replay_buffer=self.save_replay_buffer,
            save_final_checkpoint=self.save_final_checkpoint,
            log_freq=self.log_freq,
            std_log=self.std_log,
            eval_freq=self.eval_freq,
        )

    # --- eval ---

    def _evaluate(self) -> dict[str, float]:
        return run_exact_episode_eval(
            self,
            num_eval_episodes=self.num_eval_episodes,
            num_eval_steps=self.num_eval_steps,
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "buffer_size": self.buffer_size,
            "buffer_device": self.buffer_device,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "offline_sampling": self.offline_sampling,
        }


def infer_specs_from_h5(
    path: str | Path,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> tuple[spaces.Box | spaces.Dict, spaces.Box]:
    """Compatibility wrapper for H5 observation/action space inference."""
    return _infer_specs_from_h5(
        path,
        action_low=action_low,
        action_high=action_high,
    )


def infer_box_specs_from_h5(
    path: str | Path,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> tuple[spaces.Box, spaces.Box]:
    """Compatibility wrapper for flat Box H5 space inference."""
    return _infer_box_specs_from_h5(
        path,
        action_low=action_low,
        action_high=action_high,
    )


def _default_gradient_steps(agent: Any) -> int:
    return 1


def _log_update_metrics(agent: Any, metrics: dict[str, float], step: int) -> None:
    if hasattr(agent, "_log_update_metrics"):
        agent._log_update_metrics(metrics, step)
        return
    logger = getattr(agent, "logger", None)
    if logger is None:
        return
    logger.log_metrics(metrics, step)


def _to_cpu_1d(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)
    return torch.as_tensor(value).detach().cpu().reshape(-1)


def _eval_done_mask(
    infos: Mapping[str, Any], terminations: Any, truncations: Any
) -> torch.Tensor:
    if "_final_info" in infos:
        return _to_cpu_1d(infos["_final_info"]).bool()
    return (_to_cpu_1d(terminations).bool() | _to_cpu_1d(truncations).bool())


def _append_episode_metrics(
    metrics: dict[str, list[torch.Tensor]],
    episode: Mapping[str, Any],
    done_mask: torch.Tensor,
    remaining: int,
) -> int:
    appended = 0
    for raw_key, value in episode.items():
        if raw_key.startswith("_") or raw_key in EVAL_METRIC_DROP:
            continue
        if isinstance(value, Mapping):
            continue
        key = EVAL_METRIC_ALIASES.get(raw_key, raw_key)
        values = _to_cpu_1d(value)
        if values.numel() == done_mask.numel():
            values = values[done_mask]
        take = min(values.numel(), remaining)
        if take <= 0:
            continue
        metrics[key].append(values[:take])
        appended = max(appended, int(take))
    return appended


def run_exact_episode_eval(
    agent: Any, *, num_eval_episodes: int, num_eval_steps: int
) -> dict[str, float]:
    """Evaluate for exactly ``num_eval_episodes`` completed episodes (vector-env
    overshoot on the final step is aggregated, not double-counted), with
    ``num_eval_steps`` as a hard cap.

    Shared by ``OfflineRLAlgorithm._evaluate`` and any off-policy rollout
    shell that opts into exact-episode-count evaluation (e.g. Cal-QL off2on
    via its own ``num_eval_episodes``, kept step-capped by default for other
    off-policy algorithms). Requires ``agent.eval_env``, ``agent.policy``,
    and the standard ``BaseAlgorithm`` eval hooks (``_eval_start_hook``,
    ``_eval_action_and_critic_action``, ``_eval_step_hook``,
    ``_eval_finalize_hook``).
    """
    if agent.eval_env is None:
        return {}
    agent.policy.eval()
    obs, _ = agent.eval_env.reset()
    agent._eval_start_hook()

    metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
    running_returns = torch.zeros(agent.eval_env.num_envs, dtype=torch.float32)
    completed = 0
    steps = 0
    target_episodes = max(int(num_eval_episodes), 1)
    max_steps = max(int(num_eval_steps), 1)

    try:
        while completed < target_episodes and steps < max_steps:
            with torch.no_grad():
                env_action, critic_action = agent._eval_action_and_critic_action(obs)
                obs_before = obs
                obs, rewards, terminations, truncations, infos = agent.eval_env.step(
                    env_action
                )
                agent._eval_step_hook(
                    obs_before,
                    critic_action,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                )

            rewards_cpu = _to_cpu_1d(rewards).float()
            running_returns[: rewards_cpu.numel()] += rewards_cpu
            done_mask = _eval_done_mask(infos, terminations, truncations)
            remaining = target_episodes - completed

            appended = 0
            if isinstance(infos, Mapping) and "final_info" in infos:
                final_info = infos["final_info"]
                if isinstance(final_info, Mapping) and "episode" in final_info:
                    episode = final_info["episode"]
                    if isinstance(episode, Mapping):
                        appended = _append_episode_metrics(
                            metrics, episode, done_mask, remaining
                        )

            if appended == 0 and done_mask.any():
                done_returns = running_returns[done_mask[: running_returns.numel()]]
                take = min(done_returns.numel(), remaining)
                if take > 0:
                    metrics["return"].append(done_returns[:take])
                    appended = int(take)

            if done_mask.any():
                running_returns[done_mask[: running_returns.numel()]] = 0.0
            completed += appended
            steps += 1
    finally:
        agent.policy.train()

    if completed == 0:
        warnings.warn(
            f"Evaluation completed 0 episodes in {steps} steps (num_eval_steps "
            "cap). Reported eval metrics are empty. Raise --num_eval_steps or "
            "set --eval_episode_horizon to the task's episode length.",
            RuntimeWarning,
            stacklevel=2,
        )

    out: dict[str, float] = {}
    for key, values in metrics.items():
        out[key] = float(torch.cat(values).float().mean().item())
    out["episodes_completed"] = float(completed)
    out["eval_steps"] = float(steps)
    out.update(agent._eval_finalize_hook())
    return out


def _log_eval_stdout(agent: Any, metrics: dict[str, float], step: int) -> None:
    """Print a one-line eval summary to stdout in the style of
    ``OffPolicyAlgorithm.learn``."""
    # Use _first_metric if the agent provides it, otherwise fall back to dict.get.
    first = getattr(agent, "_first_metric", None)
    if first is not None:
        eval_return = first(metrics, ("return",))
        eval_success = first(metrics, ("success_at_end", "success_once"))
    else:
        eval_return = metrics.get("return", float("nan"))
        eval_success = metrics.get(
            "success_at_end", metrics.get("success_once", float("nan"))
        )
    fmt = getattr(agent, "_fmt_metric", lambda v: "nan" if v != v else f"{v:.4f}")
    print(
        f"[offline_eval] step={step} "
        f"return={fmt(eval_return)} "
        f"success_at_end={fmt(eval_success)}",
        flush=True,
    )


def run_offline_pretraining(
    agent: Any,
    *,
    num_steps: int,
    gradient_steps: Optional[int] = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_freq: int = 0,
    save_filename: str = "offline_pretrained.pt",
    save_replay_buffer: bool = False,
    save_final_checkpoint: bool = True,
    log_freq: int = 1_000,
    std_log: bool = True,
    eval_freq: int = 0,
    desc: str = "offline",
) -> OfflinePretrainResult:
    """Run an offline gradient loop for any agent exposing ``train()``.

    ``agent._global_step`` is advanced in offline update-step units. The agent's
    own ``train()`` method remains responsible for ``_global_update``.

    When *eval_freq* > 0 and the agent provides ``_evaluate`` /
    ``_log_eval_metrics``, the loop evaluates the current policy at regular
    intervals.  The agent must have a real ``eval_env`` set beforehand.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if gradient_steps is None:
        gradient_steps = _default_gradient_steps(agent)
    if gradient_steps <= 0:
        raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    last_metrics: dict[str, float] = {}
    start_step = int(getattr(agent, "_global_step", 0))
    final_target = start_step + num_steps

    _has_eval = (
        eval_freq > 0
        and hasattr(agent, "_evaluate")
        and hasattr(agent, "_log_eval_metrics")
    )

    interval_update_time = 0.0
    interval_update_steps = 0
    for step in trange(start_step, final_target, desc=desc):
        global_step = step + 1
        should_log = log_freq > 0 and (
            global_step % log_freq == 0 or global_step == final_target
        )
        update_t = time.perf_counter()
        last_metrics = agent.train(gradient_steps, compute_info=should_log)
        interval_update_time += time.perf_counter() - update_t
        interval_update_steps += gradient_steps
        agent._global_step = global_step

        if (
            _has_eval
            and global_step % eval_freq == 0
            and getattr(agent, "eval_env", None) is not None
        ):
            t0 = time.perf_counter()
            eval_metrics = agent._evaluate()
            agent._log_eval_metrics(eval_metrics, global_step)
            if std_log:
                _log_eval_stdout(agent, eval_metrics, global_step)
            logger = getattr(agent, "logger", None)
            if logger is not None:
                logger.add_scalar(
                    "time/eval_time", time.perf_counter() - t0, global_step
                )

        if log_freq > 0 and global_step % log_freq == 0:
            _log_update_metrics(agent, last_metrics, global_step)
            logger = getattr(agent, "logger", None)
            if logger is not None:
                offline_update_fps = (
                    interval_update_steps / interval_update_time
                    if interval_update_time > 0
                    else float("nan")
                )
                logger.add_scalar(
                    "time/offline_update_time", interval_update_time, global_step
                )
                logger.add_scalar(
                    "time/offline_update_fps", offline_update_fps, global_step
                )
            interval_update_time = 0.0
            interval_update_steps = 0
            if std_log:
                completed = global_step - start_step
                progress = 100.0 * completed / num_steps
                loss_summary = " ".join(
                    f"{key}={value:.4f}"
                    for key, value in last_metrics.items()
                    if isinstance(value, (int, float))
                )
                print(
                    f"[offline] step={completed}/{num_steps} "
                    f"global_step={global_step} ({progress:.2f}%) {loss_summary}",
                    flush=True,
                )

        if (
            checkpoint_root is not None
            and checkpoint_freq > 0
            and global_step % checkpoint_freq == 0
        ):
            ckpt = agent.save(
                checkpoint_root / f"checkpoint_{global_step}.pt",
                include_replay_buffer=save_replay_buffer,
            )
            if std_log:
                print(f"[offline] intermediate_checkpoint={ckpt}", flush=True)

    final_checkpoint: Optional[Path] = None
    if checkpoint_root is not None and save_final_checkpoint:
        final_checkpoint = agent.save(
            checkpoint_root / save_filename,
            include_replay_buffer=save_replay_buffer,
        )
        logger = getattr(agent, "logger", None)
        if logger is not None:
            logger.add_summary("offline/final_checkpoint", str(final_checkpoint))
        if std_log:
            print(f"[pretrain] final_checkpoint={final_checkpoint}", flush=True)
    elif std_log and checkpoint_root is None:
        print(
            "[pretrain] no checkpoint_dir resolved; pass --checkpoint_dir "
            "or --save_final_checkpoint=True to keep the pretrained weights.",
            flush=True,
        )

    return OfflinePretrainResult(
        final_step=int(getattr(agent, "_global_step", final_target)),
        final_update=int(getattr(agent, "_global_update", 0)),
        last_metrics=last_metrics,
        final_checkpoint=final_checkpoint,
    )
