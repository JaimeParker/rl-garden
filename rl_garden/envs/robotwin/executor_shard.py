"""Shard-based RoboTwin executor.

Each shard is one OS process that runs a local threaded RoboTwin executor. This
keeps PPO/SAC seeing one vector env while giving each shard its own SAPIEN/CUDA
context.  The measured high-throughput setting is four envs per shard with
parallel TOPP and serialized ctrl inside the shard.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import Any, Callable, Optional

import numpy as np

from rl_garden.envs.robotwin.config import RoboTwinEnvConfig
from rl_garden.envs.robotwin.executor import ThreadedRoboTwinExecutor


def _shard_worker_main(
    shard_id: int,
    cfg: RoboTwinEnvConfig,
    task_args: dict[str, Any],
    env_seeds: list[int],
    conn,  # multiprocessing.Connection (child side)
) -> None:
    del shard_id
    executor: Optional[ThreadedRoboTwinExecutor] = None
    try:
        executor = ThreadedRoboTwinExecutor(cfg, task_args=task_args, env_seeds=env_seeds)
        conn.send({"status": "ok"})
    except Exception as exc:
        conn.send({"status": "error", "msg": repr(exc)})
        conn.close()
        return

    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            break

        try:
            name = cmd["cmd"]
            if name == "reset":
                obs = executor.reset(
                    env_indices=cmd.get("env_indices"),
                    env_seeds=cmd.get("env_seeds"),
                )
                conn.send({"status": "ok", "obs": obs})
            elif name == "step":
                result = executor.step(cmd["actions"])
                conn.send({"status": "ok", "result": result})
            elif name == "get_obs":
                obs = executor.get_obs()
                conn.send({"status": "ok", "obs": obs})
            elif name == "close":
                executor.close(clear_cache=cmd.get("clear_cache", True))
                conn.send({"status": "ok"})
                break
            else:
                conn.send({"status": "error", "msg": f"unknown command: {name!r}"})
        except Exception as exc:
            conn.send({"status": "error", "msg": repr(exc)})

    conn.close()


class _ProcessShard:
    def __init__(
        self,
        *,
        ctx,
        shard_id: int,
        cfg: RoboTwinEnvConfig,
        task_args: dict[str, Any],
        env_seeds: list[int],
    ) -> None:
        self.shard_id = shard_id
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=_shard_worker_main,
            args=(shard_id, cfg, task_args, env_seeds, child_conn),
            daemon=False,
            name=f"robotwin-shard-{shard_id}",
        )
        self._proc.start()
        child_conn.close()
        self._recv(timeout=300)

    def reset(
        self,
        env_indices: Optional[list[int]],
        env_seeds: Optional[list[int]],
    ) -> list[dict[str, Any]]:
        self.start_reset(env_indices, env_seeds)
        return self.finish_reset()

    def start_reset(
        self,
        env_indices: Optional[list[int]],
        env_seeds: Optional[list[int]],
    ) -> None:
        self._conn.send({
            "cmd": "reset",
            "env_indices": env_indices,
            "env_seeds": env_seeds,
        })

    def finish_reset(self) -> list[dict[str, Any]]:
        return self._recv(timeout=180)["obs"]

    def step(self, actions: np.ndarray) -> list[Any]:
        self.start_step(actions)
        return self.finish_step()

    def start_step(self, actions: np.ndarray) -> None:
        self._conn.send({"cmd": "step", "actions": actions})

    def finish_step(self) -> list[Any]:
        return self._recv(timeout=180)["result"]

    def get_obs(self) -> list[dict[str, Any]]:
        self.start_get_obs()
        return self.finish_get_obs()

    def start_get_obs(self) -> None:
        self._conn.send({"cmd": "get_obs"})

    def finish_get_obs(self) -> list[dict[str, Any]]:
        return self._recv(timeout=30)["obs"]

    def close(self, clear_cache: bool = True) -> None:
        try:
            self._conn.send({"cmd": "close", "clear_cache": clear_cache})
        except (BrokenPipeError, OSError):
            pass
        try:
            self._conn.recv()
        except (EOFError, OSError):
            pass
        try:
            self._conn.close()
        except OSError:
            pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.terminate()

    def _recv(self, timeout: float) -> dict[str, Any]:
        if not self._conn.poll(timeout):
            raise TimeoutError(
                f"Shard {self.shard_id} did not respond within {timeout}s. "
                "The shard process may have crashed."
            )
        resp = self._conn.recv()
        if resp.get("status") == "error":
            raise RuntimeError(f"Shard {self.shard_id} raised: {resp['msg']}")
        return resp


class ShardedRoboTwinExecutor:
    """Vector executor that groups RoboTwin envs into process-level shards."""

    def __init__(
        self,
        cfg: RoboTwinEnvConfig,
        task_args: dict[str, Any],
        env_seeds: list[int],
        shard_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.shard_size = cfg.shard_size
        ctx = mp.get_context("spawn")

        self._shards: list[Any] = []
        self._slices: list[slice] = []
        factory = _ProcessShard if shard_factory is None else shard_factory

        for shard_id, start in enumerate(range(0, cfg.num_envs, cfg.shard_size)):
            stop = min(start + cfg.shard_size, cfg.num_envs)
            local_num_envs = stop - start
            shard_cfg = replace(
                cfg,
                num_envs=local_num_envs,
                executor_type="thread",
                parallel_topp=True if not cfg.parallel_topp else cfg.parallel_topp,
                ctrl_concurrency=1 if cfg.ctrl_concurrency == 0 else cfg.ctrl_concurrency,
            )
            shard = factory(
                ctx=ctx,
                shard_id=shard_id,
                cfg=shard_cfg,
                task_args=task_args,
                env_seeds=env_seeds[start:stop],
            )
            self._shards.append(shard)
            self._slices.append(slice(start, stop))

    def reset(
        self,
        env_indices: Optional[list[int]] = None,
        env_seeds: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        if env_indices is None:
            for shard_id, shard_slice in enumerate(self._slices):
                seeds = None if env_seeds is None else env_seeds[shard_slice.start:shard_slice.stop]
                self._shards[shard_id].start_reset(None, seeds)
            for shard in self._shards:
                shard.finish_reset()
        else:
            grouped: dict[int, tuple[list[int], list[int]]] = {}
            for offset, global_idx in enumerate(env_indices):
                shard_id, local_idx = self._locate(global_idx)
                local_indices, local_seeds = grouped.setdefault(shard_id, ([], []))
                local_indices.append(local_idx)
                if env_seeds is not None:
                    local_seeds.append(env_seeds[offset])
            for shard_id, (local_indices, local_seeds) in grouped.items():
                self._shards[shard_id].start_reset(
                    local_indices,
                    local_seeds if env_seeds is not None else None,
                )
            for shard_id in grouped:
                self._shards[shard_id].finish_reset()
        return self.get_obs()

    def step(self, actions: np.ndarray) -> list[Any]:
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}.")
        for shard_id, shard_slice in enumerate(self._slices):
            self._shards[shard_id].start_step(actions[shard_slice])
        results = []
        for shard in self._shards:
            results.extend(shard.finish_step())
        return results

    def get_obs(self) -> list[dict[str, Any]]:
        for shard in self._shards:
            shard.start_get_obs()
        obs = []
        for shard in self._shards:
            obs.extend(shard.finish_get_obs())
        return obs

    def close(self, clear_cache: bool = True) -> None:
        for shard in self._shards:
            shard.close(clear_cache=clear_cache)

    def _locate(self, global_idx: int) -> tuple[int, int]:
        if global_idx < 0 or global_idx >= self.num_envs:
            raise IndexError(f"env index {global_idx} out of range for {self.num_envs} envs")
        shard_id = global_idx // self.shard_size
        local_idx = global_idx - self._slices[shard_id].start
        return shard_id, local_idx
