"""Offline-only algorithm utilities.

This module is intentionally separate from ``OffPolicyAlgorithm``. Pure offline
algorithms such as IQL/BC should not inherit ManiSkill rollout logic, while
warm-start algorithms such as WSRL may still reuse the offline runner here for
pretraining entrypoints.
"""
from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces
from tqdm import trange

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
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
        num_eval_steps: int = 50,
        eval_env: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, seed=seed, device=device, logger=logger)
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.batch_size = batch_size
        self.gamma = gamma
        self.offline_sampling = offline_sampling
        self.std_log = std_log
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.num_eval_steps = num_eval_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.save_replay_buffer = save_replay_buffer
        self.save_final_checkpoint = save_final_checkpoint
        self.num_envs = env.num_envs

    @abstractmethod
    def train(self, gradient_steps: int) -> dict[str, float]: ...

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
        if self.eval_env is None:
            return {}
        self.policy.eval()
        obs, _ = self.eval_env.reset()
        metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(self.num_eval_steps):
            with torch.no_grad():
                obs, _, _, _, infos = self.eval_env.step(self._eval_action(obs))
                if "final_info" in infos:
                    for k, v in infos["final_info"]["episode"].items():
                        metrics[k].append(v)
        self.policy.train()
        out: dict[str, float] = {}
        for k, vs in metrics.items():
            out[k] = float(torch.stack(vs).float().mean().item())
        return out

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in metrics.items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "buffer_size": self.buffer_size,
            "buffer_device": self.buffer_device,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "offline_sampling": self.offline_sampling,
        }

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        self.logger.log_metrics(metrics, step)


def infer_box_specs_from_h5(
    path: str | Path,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> tuple[spaces.Box, spaces.Box]:
    """Infer flat Box observation/action spaces from a trajectory H5 file."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional env.
        raise ImportError(
            "Loading H5 datasets requires h5py. Install with `pip install h5py`."
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith("traj_")])
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {path}.")
        traj = f[traj_keys[0]]

        if "obs" in traj:
            obs_node = traj["obs"]
        elif "observations" in traj:
            obs_node = traj["observations"]
        else:
            raise ValueError(f"No obs/observations field in {traj_keys[0]}.")
        if isinstance(obs_node, h5py.Group):
            raise NotImplementedError(
                "Dict observations detected. infer_box_specs_from_h5 supports flat "
                "Box observations only. Use a vision-specific spec builder for RGBD."
            )
        obs_shape = tuple(obs_node.shape[1:])

        if "actions" in traj:
            action_node = traj["actions"]
        elif "action" in traj:
            action_node = traj["action"]
        else:
            raise ValueError(f"No actions/action field in {traj_keys[0]}.")
        action_shape = tuple(action_node.shape[1:])

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
    action_space = spaces.Box(
        low=action_low,
        high=action_high,
        shape=action_shape,
        dtype=np.float32,
    )
    return obs_space, action_space


def _default_gradient_steps(agent: Any) -> int:
    utd = float(getattr(agent, "utd", 1.0))
    return int(utd) if utd.is_integer() and utd > 1 else 1


def _log_update_metrics(agent: Any, metrics: dict[str, float], step: int) -> None:
    if hasattr(agent, "_log_update_metrics"):
        agent._log_update_metrics(metrics, step)
        return
    logger = getattr(agent, "logger", None)
    if logger is None:
        return
    logger.log_metrics(metrics, step)


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
        eval_success = metrics.get("success_at_end", metrics.get("success_once", float("nan")))
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

    for step in trange(start_step, final_target, desc=desc):
        last_metrics = agent.train(gradient_steps)
        global_step = step + 1
        agent._global_step = global_step

        if _has_eval and global_step % eval_freq == 0 and getattr(agent, "eval_env", None) is not None:
            t0 = time.perf_counter()
            eval_metrics = agent._evaluate()
            agent._log_eval_metrics(eval_metrics, global_step)
            if std_log:
                _log_eval_stdout(agent, eval_metrics, global_step)
            logger = getattr(agent, "logger", None)
            if logger is not None:
                logger.add_scalar("time/eval_time", time.perf_counter() - t0, global_step)

        if log_freq > 0 and global_step % log_freq == 0:
            _log_update_metrics(agent, last_metrics, global_step)
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
