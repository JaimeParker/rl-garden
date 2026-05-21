"""Process-based RoboTwin executor — one OS process per sub-environment.

Each worker process has its own mplib/SAPIEN instance, eliminating the TOPP
planning contention that collapses throughput in the thread executor when
num_envs > 1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Optional

import numpy as np

from rl_garden.envs.robotwin.adapter import (
    RoboTwinTaskAdapter,
    StepResult,
    ensure_robotwin_importable,
)
from rl_garden.envs.robotwin.config import RoboTwinEnvConfig


# ---------------------------------------------------------------------------
# Worker process entry point (top-level so spawn can pickle it)
# ---------------------------------------------------------------------------

def _worker_main(
    env_id: int,
    cfg: RoboTwinEnvConfig,
    task_args: dict[str, Any],
    env_seed: Optional[int],
    conn,  # multiprocessing.Connection (child side)
) -> None:
    # 1. Pin to a CPU core before heavy initialisation.
    if cfg.cpu_affinity and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {env_id % os.cpu_count()})
        except OSError:
            pass

    # 2. Make RoboTwin importable and stay in its working directory for the
    #    lifetime of this process.  robotwin_workdir() in adapter.reset() will
    #    chdir to the same path — effectively a no-op — which is correct.
    ensure_robotwin_importable(cfg.robotwin_root)
    if cfg.robotwin_root:
        os.chdir(os.path.abspath(cfg.robotwin_root))

    # 3. Create adapter and perform the initial reset.
    adapter = RoboTwinTaskAdapter(env_id, cfg, task_args, env_seed)
    try:
        obs = adapter.reset(env_seed)
        conn.send({"status": "ok", "obs": obs})
    except Exception as e:
        conn.send({"status": "error", "msg": repr(e)})
        conn.close()
        return

    # 4. Command loop.
    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            break

        try:
            name = cmd["cmd"]
            if name == "reset":
                obs = adapter.reset(cmd["seed"])
                conn.send({"status": "ok", "obs": obs})
            elif name == "step":
                result = adapter.step(cmd["action"])
                conn.send({"status": "ok", "result": result})
            elif name == "get_obs":
                obs = adapter.get_obs()
                conn.send({"status": "ok", "obs": obs})
            elif name == "close":
                adapter.close(clear_cache=cmd.get("clear_cache", True))
                conn.send({"status": "ok"})
                break
            else:
                conn.send({"status": "error", "msg": f"unknown command: {name!r}"})
        except Exception as e:
            conn.send({"status": "error", "msg": repr(e)})

    conn.close()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ProcessRoboTwinExecutor:
    """Parallel RoboTwin executor using one OS process per sub-environment.

    Drop-in replacement for :class:`ThreadedRoboTwinExecutor`.  Each worker
    process owns an independent mplib/SAPIEN instance so TOPP planning runs
    truly concurrently across CPU cores.
    """

    def __init__(
        self,
        cfg: RoboTwinEnvConfig,
        task_args: dict[str, Any],
        env_seeds: list[int],
    ) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        ctx = mp.get_context("spawn")

        self._procs: list[mp.Process] = []
        self._conns: list[mp.connection.Connection] = []

        for env_id in range(cfg.num_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            seed = env_seeds[env_id] if env_id < len(env_seeds) else None
            proc = ctx.Process(
                target=_worker_main,
                args=(env_id, cfg, task_args, seed, child_conn),
                daemon=True,
                name=f"robotwin-worker-{env_id}",
            )
            proc.start()
            child_conn.close()  # parent never uses the child-side end
            self._procs.append(proc)
            self._conns.append(parent_conn)

        # Wait for all workers to finish their initial reset.  SAPIEN init is
        # slow, so use a generous timeout.
        for i in range(cfg.num_envs):
            self._recv(i, timeout=300)

    # ------------------------------------------------------------------
    # Public interface (mirrors ThreadedRoboTwinExecutor)
    # ------------------------------------------------------------------

    def reset(
        self,
        env_indices: Optional[list[int]] = None,
        env_seeds: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        indices = list(range(self.num_envs)) if env_indices is None else list(env_indices)
        # Dispatch all resets concurrently (no global lock needed — each
        # process has an isolated SAPIEN scene).
        for offset, idx in enumerate(indices):
            seed = env_seeds[offset] if env_seeds is not None else None
            self._conns[idx].send({"cmd": "reset", "seed": seed})
        # Wait for resets to complete.
        for idx in indices:
            self._recv(idx, timeout=180)
        # Return a fresh obs snapshot from every env (matches thread executor).
        return self.get_obs()

    def step(self, actions: np.ndarray) -> list[StepResult]:
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}.")
        # Send all actions first so all workers start simultaneously.
        for i, conn in enumerate(self._conns):
            conn.send({"cmd": "step", "action": actions[i]})
        # Collect results in order.
        return [self._recv(i, timeout=180)["result"] for i in range(self.num_envs)]

    def get_obs(self) -> list[dict[str, Any]]:
        for conn in self._conns:
            conn.send({"cmd": "get_obs"})
        return [self._recv(i, timeout=30)["obs"] for i in range(self.num_envs)]

    def close(self, clear_cache: bool = True) -> None:
        for conn in self._conns:
            try:
                conn.send({"cmd": "close", "clear_cache": clear_cache})
            except (BrokenPipeError, OSError):
                pass
        for conn in self._conns:
            try:
                conn.recv()
            except (EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recv(self, idx: int, timeout: float) -> dict[str, Any]:
        conn = self._conns[idx]
        if not conn.poll(timeout):
            raise TimeoutError(
                f"Worker {idx} did not respond within {timeout}s. "
                "The worker process may have crashed."
            )
        resp = conn.recv()
        if resp.get("status") == "error":
            raise RuntimeError(f"Worker {idx} raised: {resp['msg']}")
        return resp
