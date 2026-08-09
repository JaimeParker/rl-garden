#!/usr/bin/env python3
"""Export rl-garden's canonical Minari Cal-QL arrays for official JAX code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.mc_buffer import MCDictReplayBuffer
from rl_garden.buffers.minari_dataset import (
    infer_specs_from_minari,
    load_minari_dataset_to_replay_buffer,
)


ARRAY_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "dones",
    "mc_returns",
)


def _chronological_indices(buffer):
    if buffer.full and buffer.pos:
        return torch.cat(
            (torch.arange(buffer.pos, buffer.per_env_buffer_size), torch.arange(buffer.pos))
        )
    return torch.arange(buffer.size)


def flatten_buffer_observations(buffer, observation_space, indices, field):
    storage = getattr(buffer, field)
    parts = []
    for key in observation_space.spaces:
        value = storage[key][indices, 0].cpu().numpy().astype(np.float32, copy=False)
        parts.append(value.reshape(value.shape[0], -1))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def export_dataset(dataset_id, output):
    import minari

    observation_space, action_space = infer_specs_from_minari(dataset_id)
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError("expected a Dict observation space, got {!r}".format(observation_space))

    dataset = minari.load_dataset(dataset_id, download=True)
    buffer = MCDictReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=int(dataset.total_steps),
        gamma=0.99,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        sparse_negative_reward=-5.0,
        success_threshold=0.5,
    )
    loaded = load_minari_dataset_to_replay_buffer(
        buffer,
        dataset_id,
        reward_scale=10.0,
        reward_bias=-5.0,
    )
    if loaded != dataset.total_steps:
        raise RuntimeError(
            "loaded {} transitions, expected {}".format(loaded, dataset.total_steps)
        )
    if buffer._mc_table is None:
        raise RuntimeError("canonical Minari loader did not populate MC returns")

    indices = _chronological_indices(buffer)
    arrays = {
        "observations": flatten_buffer_observations(
            buffer, observation_space, indices, "obs"
        ),
        "actions": buffer.actions[indices, 0].cpu().numpy().astype(np.float32),
        "next_observations": flatten_buffer_observations(
            buffer, observation_space, indices, "next_obs"
        ),
        "rewards": buffer.rewards[indices, 0].cpu().numpy().astype(np.float32),
        "dones": buffer.dones[indices, 0].cpu().numpy().astype(np.float32),
        "mc_returns": buffer._mc_table[indices, 0].cpu().numpy().astype(np.float32),
    }

    output = Path(output)
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)

    manifest = {
        "dataset_id": dataset_id,
        "dataset_minari_version": dataset.spec.minari_version,
        "total_episodes": int(dataset.total_episodes),
        "total_steps": int(dataset.total_steps),
        "observation_keys": list(observation_space.spaces.keys()),
        "preprocessing": {
            "reward_scale": 10.0,
            "reward_bias": -5.0,
            "done_source": "terminations",
            "include_truncation_transitions": True,
            "action_clip": None,
            "gamma": 0.99,
            "failed_episode_mc_return": -500.0,
        },
        "packages": {
            name: _package_version(name)
            for name in ("minari", "gymnasium", "gymnasium-robotics", "mujoco")
        },
        "requirements": list(dataset._data.metadata.get("requirements", [])),
        "env_spec": dataset.spec.env_spec.to_json(),
        "eval_env_spec": (
            dataset._eval_env_spec.to_json()
            if getattr(dataset, "_eval_env_spec", None) is not None
            else None
        ),
        "arrays": {
            key: {
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
                "sha256": array_sha256(arrays[key]),
            }
            for key in ARRAY_KEYS
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": str(output), "manifest": str(manifest_path), **manifest}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="D4RL/antmaze/large-diverse-v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_dataset(args.dataset_id, args.output)


if __name__ == "__main__":
    main()
