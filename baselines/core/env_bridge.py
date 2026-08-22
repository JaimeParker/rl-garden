"""Subprocess bridge to a live env, stepped from a separate JAX process.

Lets an official baseline's JAX process (its own venv, possibly a different
Python/CUDA stack than rl-garden's own) evaluate against rl-garden's
canonical environments without ever importing JAX and rl-garden's own
torch/gym stack in the same process: the env itself runs in a THIRD
subprocess (``baselines.core.env_server``, its own venv), communicating
over the length-prefixed binary protocol in ``wire_protocol.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np

from baselines.core.wire_protocol import (
    CLOSE_REQUEST,
    HANDSHAKE,
    MAGIC,
    OP_CLOSE,
    OP_RESET,
    OP_STEP,
    RESET_REQUEST,
    STEP_REQUEST,
    STEP_RESULT,
    read_exact,
    read_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class GymnasiumEnvBridge:
    """Spawns and speaks to ``python -m baselines.core.env_server``."""

    def __init__(
        self,
        python_executable,
        dataset_id,
        datasets_path,
        observation_keys,
        initial_seed,
    ):
        # Always spawned as `-m baselines.core.env_server` with the repo
        # root on PYTHONPATH, regardless of which venv python_executable
        # points at -- see .agents/runbooks/baseline-install.md.
        env = os.environ.copy()
        env["MINARI_DATASETS_PATH"] = datasets_path
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            python_executable,
            "-m",
            "baselines.core.env_server",
            "--dataset-id",
            dataset_id,
            "--observation-keys",
            ",".join(observation_keys),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open environment bridge pipes")
        self.reader = self.process.stdout
        self.writer = self.process.stdin
        magic, self.observation_dim, self.action_dim, self.horizon = HANDSHAKE.unpack(
            read_exact(self.reader, HANDSHAKE.size)
        )
        if magic != MAGIC:
            raise RuntimeError("invalid environment bridge handshake: {!r}".format(magic))
        self.initial_seed = int(initial_seed)
        self._reset_count = 0
        self._closed = False

    def reset(self):
        seed = self.initial_seed if self._reset_count == 0 else -1
        self._reset_count += 1
        self.writer.write(RESET_REQUEST.pack(OP_RESET, seed))
        self.writer.flush()
        read_status(self.reader)
        return np.frombuffer(
            read_exact(self.reader, self.observation_dim * 4), dtype="<f4"
        ).copy()

    def step(self, action):
        action = np.asarray(action, dtype="<f4").reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(
                "expected action dimension {}, got {}".format(
                    self.action_dim, action.size
                )
            )
        self.writer.write(STEP_REQUEST.pack(OP_STEP))
        self.writer.write(action.tobytes())
        self.writer.flush()
        read_status(self.reader)
        observation = np.frombuffer(
            read_exact(self.reader, self.observation_dim * 4), dtype="<f4"
        ).copy()
        reward, terminated, truncated = STEP_RESULT.unpack(
            read_exact(self.reader, STEP_RESULT.size)
        )
        return observation, float(reward), bool(terminated), bool(truncated)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None:
                self.writer.write(CLOSE_REQUEST.pack(OP_CLOSE))
                self.writer.flush()
                read_status(self.reader)
        finally:
            self.writer.close()
            self.reader.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
