#!/usr/bin/env python3
"""Print legacy D4RL environment and official Cal-QL dataset parity facts."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

from rl_garden.buffers.d4rl_legacy_dataset import (
    _make_legacy_env,
    _official_calql_antmaze_dataset,
)


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _json_array(value: Any) -> list:
    return np.asarray(value).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="antmaze-large-diverse-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=10)
    args = parser.parse_args()

    env = _make_legacy_env(args.env_id)
    try:
        raw = env.get_dataset()
        dataset = _official_calql_antmaze_dataset(
            env,
            reward_scale=10.0,
            reward_bias=-5.0,
            clip_action=0.99999,
            gamma=0.99,
        )

        env.seed(args.seed)
        observation = env.reset()
        action_rng = np.random.RandomState(args.seed)
        rollout_observations = [np.asarray(observation)]
        rollout_rewards = []
        rollout_dones = []
        for _ in range(args.rollout_steps):
            action = action_rng.uniform(
                env.action_space.low, env.action_space.high
            ).astype(env.action_space.dtype)
            observation, reward, done, _ = env.step(action)
            rollout_observations.append(np.asarray(observation))
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            if done:
                break

        model = env.model
        dt = float(env.dt)
        result = {
            "environment": {
                "env_id": args.env_id,
                "observation_shape": list(env.observation_space.shape),
                "action_shape": list(env.action_space.shape),
                "max_episode_steps": int(env.spec.max_episode_steps),
                "model_timestep": float(model.opt.timestep),
                "frame_skip": int(env.frame_skip),
                "control_dt": dt,
                "horizon_physical_seconds": dt * int(env.spec.max_episode_steps),
                "actuator_ctrlrange": _json_array(model.actuator_ctrlrange),
                "actuator_gear": _json_array(model.actuator_gear),
            },
            "dataset": {
                "raw_transitions": int(raw["rewards"].shape[0]),
                "loaded_transitions": int(dataset["rewards"].shape[0]),
                "reward_values": np.unique(dataset["rewards"]).tolist(),
                "terminal_count": int(dataset["terminals"].sum()),
                "max_abs_action": float(np.abs(dataset["actions"]).max()),
                "mc_min": float(dataset["mc_returns"].min()),
                "mc_max": float(dataset["mc_returns"].max()),
                "mc_quantiles": np.quantile(
                    dataset["mc_returns"], [0.0, 0.25, 0.5, 0.75, 1.0]
                ).tolist(),
                "hashes": {
                    key: _array_hash(value) for key, value in dataset.items()
                },
            },
            "seeded_rollout": {
                "steps": len(rollout_rewards),
                "observation_hash": _array_hash(np.asarray(rollout_observations)),
                "rewards": rollout_rewards,
                "dones": rollout_dones,
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
