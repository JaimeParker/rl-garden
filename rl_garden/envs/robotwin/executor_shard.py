"""Shard-based RoboTwin executor.

Each shard is one OS process that runs a local threaded RoboTwin executor. This
keeps PPO/SAC seeing one vector env while giving each shard its own SAPIEN/CUDA
context.  The measured high-throughput setting is four envs per shard with
parallel TOPP and serialized ctrl inside the shard.

Obs arrays (rgb, wrist cameras, state) transfer via shared memory to avoid
Pipe pickle overhead.  String metadata (instruction, _env_seed) travel through
the Pipe alongside small scalars (reward, terminated, truncated, info).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional

import numpy as np

from rl_garden.envs.robotwin.adapter import StepResult
from rl_garden.envs.robotwin.config import RoboTwinEnvConfig
from rl_garden.envs.robotwin.executor import ThreadedRoboTwinExecutor

_STATE_DIM = 14  # matches RoboTwinEnv hard-coded state observation shape


class _ShardShm:
    """Shared memory blocks for one shard's numpy observation arrays.

    Allocated in the parent process.  The worker attaches by name via specs().
    Only tensor-compatible keys are stored here (rgb, wrist cameras, state).
    String fields (instruction, _env_seed) travel through the Pipe instead.
    """

    def __init__(self, num_envs: int, image_size: tuple[int, int], include_wrist: bool) -> None:
        h, w = image_size
        self._num_envs = num_envs
        self._mems: dict[str, SharedMemory] = {}
        self._views: dict[str, np.ndarray] = {}

        def _alloc(key: str, shape: tuple, dtype) -> None:
            size = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
            shm = SharedMemory(create=True, size=size)
            self._mems[key] = shm
            self._views[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        _alloc("rgb", (num_envs, h, w, 3), np.uint8)
        if include_wrist:
            _alloc("rgb_left_wrist", (num_envs, h, w, 3), np.uint8)
            _alloc("rgb_right_wrist", (num_envs, h, w, 3), np.uint8)
        _alloc("state", (num_envs, _STATE_DIM), np.float32)

    def specs(self) -> dict[str, tuple]:
        """Serializable (shm_name, shape, dtype_str) per key — safe to pickle."""
        return {
            key: (shm.name, self._views[key].shape, self._views[key].dtype.str)
            for key, shm in self._mems.items()
        }

    def read_obs_list(self) -> list[dict[str, np.ndarray]]:
        """Copy current shm contents into fresh per-env obs dicts."""
        return [
            {key: self._views[key][i].copy() for key in self._views}
            for i in range(self._num_envs)
        ]

    def close(self) -> None:
        for shm in self._mems.values():
            shm.close()
            shm.unlink()
        self._mems.clear()
        self._views.clear()


def _attach_shm(specs: dict[str, tuple]) -> tuple[dict, dict]:
    """Worker-side: attach to parent-allocated shared memory, return (mems, views)."""
    mems: dict[str, SharedMemory] = {}
    views: dict[str, np.ndarray] = {}
    for key, (name, shape, dtype_str) in specs.items():
        shm = SharedMemory(name=name, create=False)
        mems[key] = shm
        views[key] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return mems, views


def _write_obs_to_shm(views: dict[str, np.ndarray], obs_list: list[dict[str, Any]]) -> None:
    """Write numpy obs fields from obs_list into shm views (worker-side only).

    Images are resized to the shm slot shape when the raw obs resolution differs
    from the target (e.g. RoboTwin returns 240×320 but shm is sized for 64×64).
    """
    for i, obs in enumerate(obs_list):
        for key, view in views.items():
            src = obs.get(key)
            if src is not None:
                arr = np.asarray(src)
                slot = view[i]
                if arr.shape != slot.shape:
                    h, w = slot.shape[:2]
                    from PIL import Image
                    arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((w, h)))
                slot[:] = arr.astype(view.dtype, copy=False)
            else:
                view[i] = 0


def _shard_worker_main(
    shard_id: int,
    cfg: RoboTwinEnvConfig,
    task_args: dict[str, Any],
    env_seeds: list[int],
    shm_specs: dict[str, tuple],
    conn,  # multiprocessing.Connection (child side)
) -> None:
    del shard_id
    shm_mems, shm_views = _attach_shm(shm_specs)
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
                obs_list = executor.reset(
                    env_indices=cmd.get("env_indices"),
                    env_seeds=cmd.get("env_seeds"),
                )
                _write_obs_to_shm(shm_views, obs_list)
                conn.send({
                    "status": "ok",
                    "meta_fields": [
                        {
                            "instruction": obs.get("instruction"),
                            "_env_seed": obs.get("_env_seed"),
                        }
                        for obs in obs_list
                    ],
                })
            elif name == "step":
                results = executor.step(cmd["actions"])
                _write_obs_to_shm(shm_views, [r.obs for r in results])
                conn.send({
                    "status": "ok",
                    "metas": [
                        {
                            "reward": r.reward,
                            "terminated": r.terminated,
                            "truncated": r.truncated,
                            "info": r.info,
                        }
                        for r in results
                    ],
                })
            elif name == "close":
                executor.close(clear_cache=cmd.get("clear_cache", True))
                conn.send({"status": "ok"})
                break
            else:
                conn.send({"status": "error", "msg": f"unknown command: {name!r}"})
        except Exception as exc:
            conn.send({"status": "error", "msg": repr(exc)})

    for shm in shm_mems.values():
        shm.close()
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
        shm: _ShardShm,
    ) -> None:
        self.shard_id = shard_id
        self._local_num_envs = cfg.num_envs
        self._shm = shm
        self._env_meta: list[dict[str, Any]] = [
            {"instruction": None, "_env_seed": None} for _ in range(cfg.num_envs)
        ]
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=_shard_worker_main,
            args=(shard_id, cfg, task_args, env_seeds, shm.specs(), child_conn),
            daemon=False,
            name=f"robotwin-shard-{shard_id}",
        )
        self._proc.start()
        child_conn.close()
        self._recv(timeout=300)

    def get_obs(self) -> list[dict[str, Any]]:
        """Read obs from shm and merge cached string metadata — no IPC."""
        obs_list = self._shm.read_obs_list()
        for i, obs in enumerate(obs_list):
            obs.update(self._env_meta[i])
        return obs_list

    def start_reset(self, env_indices: Optional[list[int]], env_seeds: Optional[list[int]]) -> None:
        self._conn.send({"cmd": "reset", "env_indices": env_indices, "env_seeds": env_seeds})

    def finish_reset(self) -> list[dict[str, Any]]:
        resp = self._recv(timeout=180)
        for i, meta in enumerate(resp.get("meta_fields", [])):
            if i < len(self._env_meta):
                self._env_meta[i] = meta
        obs_list = self._shm.read_obs_list()
        for i, obs in enumerate(obs_list):
            obs.update(self._env_meta[i])
        return obs_list

    def start_step(self, actions: np.ndarray) -> None:
        self._conn.send({"cmd": "step", "actions": actions})

    def finish_step(self) -> list[StepResult]:
        resp = self._recv(timeout=180)
        metas = resp["metas"]
        obs_list = self._shm.read_obs_list()
        return [
            StepResult(
                obs=obs_list[i],
                reward=metas[i]["reward"],
                terminated=metas[i]["terminated"],
                truncated=metas[i]["truncated"],
                info=metas[i]["info"],
            )
            for i in range(self._local_num_envs)
        ]

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
    """Vector executor that groups RoboTwin envs into process-level shards.

    Obs arrays transfer via shared memory; string metadata and step scalars
    travel through the Pipe.  get_obs() reads directly from shared memory
    without an IPC round-trip.
    """

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
        self._shm_blocks: list[_ShardShm] = []
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
            shm = _ShardShm(local_num_envs, cfg.image_size, cfg.include_wrist_cameras)
            self._shm_blocks.append(shm)
            shard = factory(
                ctx=ctx,
                shard_id=shard_id,
                cfg=shard_cfg,
                task_args=task_args,
                env_seeds=env_seeds[start:stop],
                shm=shm,
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
            obs = []
            for shard in self._shards:
                obs.extend(shard.finish_reset())
            return obs
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

    def step(self, actions: np.ndarray) -> list[StepResult]:
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}.")
        for shard_id, shard_slice in enumerate(self._slices):
            self._shards[shard_id].start_step(actions[shard_slice])
        results = []
        for shard in self._shards:
            results.extend(shard.finish_step())
        return results

    def get_obs(self) -> list[dict[str, Any]]:
        obs = []
        for shard in self._shards:
            obs.extend(shard.get_obs())
        return obs

    def close(self, clear_cache: bool = True) -> None:
        for shard, shm in zip(self._shards, self._shm_blocks):
            shard.close(clear_cache=clear_cache)
            shm.close()

    def _locate(self, global_idx: int) -> tuple[int, int]:
        if global_idx < 0 or global_idx >= self.num_envs:
            raise IndexError(f"env index {global_idx} out of range for {self.num_envs} envs")
        shard_id = global_idx // self.shard_size
        local_idx = global_idx - self._slices[shard_id].start
        return shard_id, local_idx
