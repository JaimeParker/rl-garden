"""Official Cal-QL dataset semantics for legacy D4RL AntMaze."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import _add_flat_transitions
from rl_garden.buffers.base import BaseReplayBuffer


def _make_legacy_env(env_id: str):
    try:
        import d4rl  # noqa: F401
        import gym as legacy_gym
    except ImportError as exc:
        raise ImportError(
            "Loading a d4rl_legacy dataset requires the `d4rl-legacy` optional "
            "dependencies. Install rl-garden with `[d4rl-legacy]`."
        ) from exc
    return legacy_gym.make(env_id).unwrapped


def _box_from_legacy(space: Any) -> spaces.Box:
    if not hasattr(space, "low") or not hasattr(space, "high"):
        raise TypeError(
            f"Legacy D4RL space must be Box-like, got {type(space).__name__}."
        )
    return spaces.Box(
        low=np.asarray(space.low, dtype=np.float32),
        high=np.asarray(space.high, dtype=np.float32),
        shape=space.shape,
        dtype=np.float32,
    )


def infer_specs_from_d4rl_legacy(env_id: str) -> tuple[spaces.Box, spaces.Box]:
    env = _make_legacy_env(env_id)
    try:
        return _box_from_legacy(env.observation_space), _box_from_legacy(env.action_space)
    finally:
        env.close()


def _mc_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    *,
    gamma: float,
    negative_reward: float,
) -> np.ndarray:
    if np.all(rewards == negative_reward):
        return np.full_like(rewards, negative_reward / (1.0 - gamma), dtype=np.float32)

    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running * (1.0 - terminals[index])
        returns[index] = running
    return returns


def _official_calql_antmaze_dataset(
    env: Any,
    *,
    reward_scale: float,
    reward_bias: float,
    clip_action: float,
    gamma: float,
    num_episodes: Optional[int] = None,
) -> dict[str, np.ndarray]:
    raw = env.get_dataset()
    current: defaultdict[str, list[Any]] = defaultdict(list)
    episodes: list[dict[str, np.ndarray]] = []
    episode_step = 0
    use_timeouts = "timeouts" in raw

    for index in range(raw["rewards"].shape[0]):
        done = bool(raw["terminals"][index])
        final_timestep = (
            bool(raw["timeouts"][index])
            if use_timeouts
            else episode_step == env._max_episode_steps - 1
        )

        if not (final_timestep or index == raw["rewards"].shape[0] - 1):
            for key in (
                "actions",
                "next_observations",
                "observations",
                "rewards",
                "terminals",
                "timeouts",
            ):
                if key in raw:
                    current[key].append(raw[key][index])
            if "next_observations" not in raw:
                current["next_observations"].append(raw["observations"][index + 1])
            episode_step += 1

        if (done or final_timestep) and episode_step > 0:
            episode = {key: np.asarray(value) for key, value in current.items()}
            rewards = episode["rewards"] * reward_scale + reward_bias
            terminals = episode["terminals"].astype(np.float32)
            episode["rewards"] = rewards
            episode["terminals"] = terminals
            episode["actions"] = np.clip(episode["actions"], -clip_action, clip_action)
            episode["mc_returns"] = _mc_returns(
                rewards,
                terminals,
                gamma=gamma,
                negative_reward=reward_bias,
            )
            episodes.append(episode)
            current = defaultdict(list)
            episode_step = 0
            if num_episodes is not None and len(episodes) >= num_episodes:
                break

    if not episodes:
        raise ValueError("No complete trajectories found in legacy D4RL dataset.")

    keys = (
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "terminals",
        "mc_returns",
    )
    return {
        key: np.concatenate([episode[key] for episode in episodes], axis=0).astype(
            np.float32
        )
        for key in keys
    }


def load_d4rl_legacy_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    env_id: str,
    *,
    num_episodes: Optional[int] = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    del success_key
    if "antmaze" not in env_id.lower():
        raise NotImplementedError(
            "The official Cal-QL d4rl_legacy loader currently supports AntMaze only."
        )

    env = _make_legacy_env(env_id)
    try:
        dataset = _official_calql_antmaze_dataset(
            env,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            clip_action=0.99999,
            gamma=float(buffer.gamma),
            num_episodes=num_episodes,
        )
    finally:
        env.close()

    device = buffer.storage_device
    rewards = torch.as_tensor(dataset["rewards"], device=device)
    successes = (rewards > reward_bias).float()
    return _add_flat_transitions(
        buffer,
        torch.as_tensor(dataset["observations"], device=device),
        torch.as_tensor(dataset["next_observations"], device=device),
        torch.as_tensor(dataset["actions"], device=device),
        rewards,
        torch.as_tensor(dataset["terminals"], device=device),
        torch.as_tensor(dataset["mc_returns"], device=device),
        successes,
    )
