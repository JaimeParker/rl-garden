"""TOPP worker pool — offloads mplib TOPP planning to independent CPU processes.

Each worker process owns its own ``mplib.Planner`` instances (left + right arm),
so TOPP calls from N environment threads run truly in parallel without GIL
contention.  The SAPIEN scene stays in the main process, avoiding GPU OOM.

Protocol (via multiprocessing.Pipe, duplex):
  main → worker: {"cmd": "init", "config": {...}}
  worker → main: {"status": "ok"}

  main → worker: {"side": "left"|"right", "path": np.ndarray, "dt": float}
  worker → main: {"status": "ok", "result": tuple} | {"status": "error", "msg": str}

  main closes conn to signal shutdown.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
from typing import Any


# ---------------------------------------------------------------------------
# Worker process entry point (top-level function — spawn-picklable)
# ---------------------------------------------------------------------------

def _topp_worker_main(env_id: int, conn, cpu_affinity: bool) -> None:  # noqa: ANN001
    """Run inside a spawned worker process.

    Waits for an "init" command with planner config, then enters a TOPP
    dispatch loop until the parent closes the connection.
    """
    import numpy as np  # imported here so spawn doesn't inherit main-process state

    if cpu_affinity and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {env_id % os.cpu_count()})
        except OSError:
            pass

    # --- Phase 1: receive planner configuration ---
    try:
        cmd = conn.recv()
    except EOFError:
        conn.close()
        return

    if cmd.get("cmd") != "init":
        conn.send({"status": "error", "msg": f"expected 'init', got {cmd!r}"})
        conn.close()
        return

    try:
        import mplib  # noqa: PLC0415

        def _make_planner(arm_cfg: dict[str, Any]):
            planner = mplib.Planner(
                urdf=arm_cfg["urdf_path"],
                srdf=arm_cfg["srdf_path"],
                move_group=arm_cfg["move_group"],
                user_link_names=arm_cfg["link_names"],
                user_joint_names=arm_cfg["joint_names"],
                use_convex=False,
            )
            # set_base_pose requires an mplib.Pose object (not a raw array).
            pose = mplib.Pose(
                p=np.array(arm_cfg["origin_pose_p"], dtype=np.float64),
                q=np.array(arm_cfg["origin_pose_q"], dtype=np.float64),
            )
            planner.set_base_pose(pose)
            return planner

        cfg = cmd["config"]
        left_planner = _make_planner(cfg["left"])
        right_planner = _make_planner(cfg["right"])
        conn.send({"status": "ok"})
    except Exception as e:
        conn.send({"status": "error", "msg": repr(e)})
        conn.close()
        return

    # --- Phase 2: TOPP dispatch loop ---
    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            break

        try:
            side = cmd.get("side")
            planner = left_planner if side == "left" else right_planner
            result = planner.TOPP(cmd["path"], cmd["dt"])
            conn.send({"status": "ok", "result": result})
        except Exception as e:
            conn.send({"status": "error", "msg": repr(e)})

    conn.close()


# ---------------------------------------------------------------------------
# Proxy planner — installed on task.robot in place of the real MplibPlanner
# ---------------------------------------------------------------------------

class RemoteToppPlanner:
    """Proxy that forwards ``TOPP()`` calls to a worker process via Pipe.

    ``Pipe.recv()`` is blocking I/O; Python releases the GIL while waiting,
    so N threads can each block on their own worker simultaneously, achieving
    true cross-env TOPP parallelism without process-level SAPIEN duplication.
    """

    def __init__(self, conn, side: str) -> None:
        self._conn = conn
        self._side = side

    def TOPP(self, path, dt, verbose: bool = True):  # noqa: N802
        self._conn.send({"side": self._side, "path": path, "dt": dt})
        resp = self._conn.recv()
        if resp["status"] == "error":
            # Re-raise so take_action()'s except branch can apply its fallback.
            raise RuntimeError(f"TOPP worker ({self._side}): {resp['msg']}")
        return resp["result"]


# ---------------------------------------------------------------------------
# Pool — manages N worker processes
# ---------------------------------------------------------------------------

class ToppWorkerPool:
    """Owns N TOPP worker processes, one per environment.

    Workers are spawned at construction time but wait for an "init" command
    before creating mplib planners (because planner config is only known after
    the first ``task.setup_demo()`` call).
    """

    def __init__(self, num_envs: int, cpu_affinity: bool = False) -> None:
        ctx = mp.get_context("spawn")
        self._conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        for env_id in range(num_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_topp_worker_main,
                args=(env_id, child_conn, cpu_affinity),
                daemon=True,
                name=f"topp-worker-{env_id}",
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

    def init_env(self, env_id: int, config: dict[str, Any], timeout: float = 120.0) -> None:
        """Send planner config to worker *env_id* and wait for its init ack."""
        conn = self._conns[env_id]
        conn.send({"cmd": "init", "config": config})
        if not conn.poll(timeout):
            raise TimeoutError(
                f"TOPP worker {env_id} did not respond to init within {timeout}s"
            )
        resp = conn.recv()
        if resp["status"] != "ok":
            raise RuntimeError(f"TOPP worker {env_id} init failed: {resp.get('msg')}")

    def make_planners(self, env_id: int) -> tuple[RemoteToppPlanner, RemoteToppPlanner]:
        """Return ``(left_proxy, right_proxy)`` for installation on ``task.robot``."""
        conn = self._conns[env_id]
        return RemoteToppPlanner(conn, "left"), RemoteToppPlanner(conn, "right")

    def suspend_all(self) -> None:
        """Freeze all alive TOPP worker processes before entering ctrl loops."""
        for proc in self._procs:
            if proc.is_alive() and proc.pid is not None:
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass

    def resume_all(self) -> None:
        """Resume all alive TOPP worker processes before TOPP planning."""
        for proc in self._procs:
            if proc.is_alive() and proc.pid is not None:
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass

    def close(self) -> None:
        self.resume_all()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()


class ToppCtrlCoordinator:
    """Coordinate TOPP worker lifecycle around env thread phases.

    All env threads synchronize before TOPP so workers are resumed together,
    then synchronize again before ctrl so workers are frozen while SAPIEN
    ``scene.step()`` runs in the main process.
    """

    def __init__(self, num_envs: int, pool: ToppWorkerPool) -> None:
        self._pre_topp = threading.Barrier(num_envs, action=pool.resume_all)
        self._pre_ctrl = threading.Barrier(num_envs, action=pool.suspend_all)

    def pre_topp(self) -> None:
        self._pre_topp.wait()

    def pre_ctrl(self) -> None:
        self._pre_ctrl.wait()
