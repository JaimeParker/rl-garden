"""Threaded RoboTwin executor used internally by :class:`RoboTwinEnv`."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NoReturn, Optional

import numpy as np

from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter, StepResult
from rl_garden.envs.robotwin.config import RoboTwinEnvConfig


_EXECUTOR_TIMEOUT_EXIT_CODE = 124


def _terminate_process_after_executor_timeout(operation: str, timeout_seconds: float) -> NoReturn:
    """Hard-exit when a native RoboTwin call ignores cancellation.

    A timed-out MPLib/SAPIEN call can remain inside C++ while holding a SubEnv
    lock. Normal exception unwinding then deadlocks in ``close()`` while trying
    to acquire that same lock. A process boundary is the only reliable way to
    reclaim those native resources; the outer launcher may resume from the
    latest complete checkpoint/replay pair.
    """
    print(
        "[robotwin-executor-fatal] "
        f"RoboTwin {operation} exceeded executor_timeout_seconds={timeout_seconds:g}; "
        f"hard-exiting with code {_EXECUTOR_TIMEOUT_EXIT_CODE} because native cleanup may deadlock.",
        file=sys.stderr,
        flush=True,
    )
    os._exit(_EXECUTOR_TIMEOUT_EXIT_CODE)


class SubEnv:
    def __init__(
        self,
        env_id: int,
        cfg: RoboTwinEnvConfig,
        task_args: dict[str, Any],
        env_seed: Optional[int],
        global_lock: threading.Lock,
    ) -> None:
        self.adapter = RoboTwinTaskAdapter(env_id, cfg, task_args, env_seed)
        self.lock = threading.Lock()
        self.global_lock = global_lock

    def reset(self, env_seed: Optional[int] = None) -> dict[str, Any]:
        with self.global_lock:
            with self.lock:
                return self.adapter.reset(env_seed)

    def step(self, action: np.ndarray) -> StepResult:
        with self.lock:
            return self.adapter.step(action)

    def get_obs(self) -> dict[str, Any]:
        with self.lock:
            return self.adapter.get_obs()

    def close(self, clear_cache: bool = True) -> None:
        with self.lock:
            self.adapter.close(clear_cache=clear_cache)


class ThreadedRoboTwinExecutor:
    """Parallel step/reset executor.

    This class intentionally does not implement Gym. It owns RoboTwin task
    lifecycles and returns raw Python/numpy values to the public env wrapper.
    """

    def __init__(
        self,
        cfg: RoboTwinEnvConfig,
        task_args: dict[str, Any],
        env_seeds: list[int],
    ) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.global_lock = threading.Lock()
        self.timeout_seconds = float(cfg.executor_timeout_seconds)
        self.pool = ThreadPoolExecutor(max_workers=cfg.num_envs)
        self.envs = [
            SubEnv(i, cfg, task_args, env_seeds[i] if i < len(env_seeds) else None, self.global_lock)
            for i in range(cfg.num_envs)
        ]

    def reset(self, env_indices: Optional[list[int]] = None, env_seeds: Optional[list[int]] = None) -> list[dict[str, Any]]:
        indices = list(range(self.num_envs)) if env_indices is None else list(env_indices)
        futures = {}
        for offset, idx in enumerate(indices):
            seed = None if env_seeds is None else env_seeds[offset]
            futures[idx] = self.pool.submit(self.envs[idx].reset, seed)
        for idx in indices:
            self._wait(futures[idx], operation=f"reset env {idx}")
        return self.get_obs()

    def step(self, actions: np.ndarray) -> list[StepResult]:
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}.")
        futures = [self.pool.submit(self.envs[i].step, actions[i]) for i in range(self.num_envs)]
        return [
            self._wait(future, operation=f"step env {idx}")
            for idx, future in enumerate(futures)
        ]

    def _wait(self, future, *, operation: str):
        try:
            return future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            _terminate_process_after_executor_timeout(operation, self.timeout_seconds)

    def get_obs(self) -> list[dict[str, Any]]:
        return [env.get_obs() for env in self.envs]

    def close(self, clear_cache: bool = True) -> None:
        for env in self.envs:
            env.close(clear_cache=clear_cache)
        self.pool.shutdown(wait=True)
