#!/usr/bin/env python3
"""Serve a recovered Minari environment to a separate JAX process."""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np

try:
    from .calql_minari_protocol import (
        HANDSHAKE,
        MAGIC,
        OP_CLOSE,
        OP_RESET,
        OP_STEP,
        RESET_REQUEST,
        STATUS,
        STATUS_OK,
        STEP_REQUEST,
        STEP_RESULT,
        read_exact,
        write_error,
    )
except ImportError:  # Direct script execution.
    from calql_minari_protocol import (
        HANDSHAKE,
        MAGIC,
        OP_CLOSE,
        OP_RESET,
        OP_STEP,
        RESET_REQUEST,
        STATUS,
        STATUS_OK,
        STEP_REQUEST,
        STEP_RESULT,
        read_exact,
        write_error,
    )


def flatten_observation(observation, keys):
    if not isinstance(observation, dict):
        raise TypeError("expected a Dict observation from the recovered Minari env")
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys]
    ).astype(np.float32, copy=False)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--observation-keys",
        default="achieved_goal,desired_goal,observation",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    keys = tuple(part.strip() for part in args.observation_keys.split(",") if part)

    import minari

    dataset = minari.load_dataset(args.dataset_id, download=False)
    env = dataset.recover_environment(eval_env=True)
    first_observation, _ = env.reset(seed=0)
    observation_dim = int(flatten_observation(first_observation, keys).size)
    action_dim = int(np.prod(env.action_space.shape))
    horizon = int(env.spec.max_episode_steps)

    reader = sys.stdin.buffer
    writer = sys.stdout.buffer
    writer.write(HANDSHAKE.pack(MAGIC, observation_dim, action_dim, horizon))
    writer.flush()

    try:
        while True:
            (opcode,) = STEP_REQUEST.unpack(read_exact(reader, STEP_REQUEST.size))
            try:
                if opcode == OP_RESET:
                    seed_bytes = read_exact(reader, RESET_REQUEST.size - STEP_REQUEST.size)
                    (seed,) = np.frombuffer(seed_bytes, dtype="<i8", count=1)
                    observation, _ = env.reset(seed=None if seed < 0 else int(seed))
                    flat = flatten_observation(observation, keys)
                    writer.write(STATUS.pack(STATUS_OK))
                    writer.write(flat.astype("<f4", copy=False).tobytes())
                    writer.flush()
                elif opcode == OP_STEP:
                    action_bytes = read_exact(reader, action_dim * 4)
                    action = np.frombuffer(action_bytes, dtype="<f4").copy()
                    observation, reward, terminated, truncated, _ = env.step(action)
                    flat = flatten_observation(observation, keys)
                    writer.write(STATUS.pack(STATUS_OK))
                    writer.write(flat.astype("<f4", copy=False).tobytes())
                    writer.write(
                        STEP_RESULT.pack(
                            float(reward), bool(terminated), bool(truncated)
                        )
                    )
                    writer.flush()
                elif opcode == OP_CLOSE:
                    writer.write(STATUS.pack(STATUS_OK))
                    writer.flush()
                    break
                else:
                    raise ValueError("unknown opcode: {}".format(opcode))
            except Exception:
                write_error(writer, traceback.format_exc())
    finally:
        env.close()


if __name__ == "__main__":
    main()
