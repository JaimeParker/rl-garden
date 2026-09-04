"""Cal-QL dataset semantics for supported legacy D4RL task families."""
from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import _add_flat_transitions
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dataset_backend_registry import (
    DatasetBackend,
    DatasetRequest,
    register_dataset_backend,
)


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


def _finite_returns(rewards: np.ndarray, terminals: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running * (1.0 - terminals[index])
        returns[index] = running
    return returns


def _infinite_tail_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = float(rewards[-1]) / (1.0 - gamma)
    returns[-1] = running
    for index in range(len(rewards) - 2, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def _qlearning_episodes(
    env: Any,
    *,
    reward_scale: float,
    reward_bias: float,
    clip_action: float,
    num_episodes: int | None,
) -> list[dict[str, np.ndarray]]:
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
            for key in ("actions", "observations", "rewards", "terminals"):
                current[key].append(raw[key][index])
            if "next_observations" in raw:
                current["next_observations"].append(raw["next_observations"][index])
            else:
                current["next_observations"].append(raw["observations"][index + 1])
            episode_step += 1

        if (done or final_timestep) and episode_step > 0:
            episode = {key: np.asarray(value) for key, value in current.items()}
            episode["rewards"] = (
                episode["rewards"] * reward_scale + reward_bias
            ).astype(np.float32)
            episode["terminals"] = episode["terminals"].astype(np.float32)
            episode["actions"] = np.clip(
                episode["actions"], -clip_action, clip_action
            )
            episode_end = np.zeros_like(episode["terminals"], dtype=np.float32)
            episode_end[-1] = 1.0
            episode["episode_end"] = episode_end
            episodes.append(episode)
            current = defaultdict(list)
            episode_step = 0
            if num_episodes is not None and len(episodes) >= num_episodes:
                break

    if not episodes:
        raise ValueError("No complete trajectories found in legacy D4RL dataset.")
    return episodes


def _concatenate_episodes(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = (
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "terminals",
        "mc_returns",
        "episode_end",
    )
    return {
        key: np.concatenate([episode[key] for episode in episodes], axis=0).astype(
            np.float32
        )
        for key in keys
    }


def _standard_d4rl_dataset(
    env: Any,
    *,
    reward_scale: float,
    reward_bias: float,
    clip_action: float,
    gamma: float,
    num_episodes: int | None = None,
) -> dict[str, np.ndarray]:
    episodes = _qlearning_episodes(
        env,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        clip_action=clip_action,
        num_episodes=num_episodes,
    )
    for episode in episodes:
        episode["mc_returns"] = _finite_returns(
            episode["rewards"], episode["terminals"], gamma
        )
    return _concatenate_episodes(episodes)


def _official_calql_kitchen_dataset(
    env: Any,
    *,
    reward_scale: float,
    reward_bias: float,
    clip_action: float,
    gamma: float,
    num_episodes: int | None = None,
) -> dict[str, np.ndarray]:
    episodes = _qlearning_episodes(
        env,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        clip_action=clip_action,
        num_episodes=num_episodes,
    )
    for episode in episodes:
        episode["mc_returns"] = _infinite_tail_returns(episode["rewards"], gamma)
    return _concatenate_episodes(episodes)


def _official_calql_antmaze_dataset(
    env: Any,
    *,
    reward_scale: float,
    reward_bias: float,
    clip_action: float,
    gamma: float,
    num_episodes: int | None = None,
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
            # True episode boundary (this episode's own last stored step),
            # independent of `terminals` -- an episode closed by `final_timestep`
            # (timeout) has `terminals[-1] == False` but is still a real MC
            # recursion boundary.
            episode_end = np.zeros_like(terminals)
            episode_end[-1] = 1.0
            episode["episode_end"] = episode_end
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
        "episode_end",
    )
    return {
        key: np.concatenate([episode[key] for episode in episodes], axis=0).astype(
            np.float32
        )
        for key in keys
    }


_LOCOMOTION_PREFIXES = ("halfcheetah-", "hopper-", "walker2d-")


def is_locomotion_env_id(env_id: str) -> bool:
    """True for D4RL Gym MuJoCo locomotion tasks (dense reward, no terminal)."""
    return env_id.lower().startswith(_LOCOMOTION_PREFIXES)


def load_d4rl_legacy_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    env_id: str,
    *,
    num_episodes: int | None = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    del success_key
    env_name = env_id.lower()
    if "antmaze" in env_name:
        family = "antmaze"
    elif "kitchen" in env_name:
        family = "kitchen"
    elif env_name.startswith(("pen-", "door-", "relocate-")) and "binary" not in env_name:
        family = "adroit"
    elif is_locomotion_env_id(env_name):
        family = "locomotion"
        warnings.warn(
            "Loading a MuJoCo locomotion D4RL dataset: mc_returns are "
            "computed as a finite-horizon backward sum, which underestimates "
            "the true value for timeout-truncated episodes (all of "
            "HalfCheetah, and the non-falling tail of Hopper/Walker2d) since "
            "there is no terminal state to anchor an infinite-horizon "
            "return. Treat mc_returns for this family as unreliable unless "
            "you have accounted for this.",
            RuntimeWarning,
        )
    else:
        raise NotImplementedError(
            "The d4rl_legacy loader supports AntMaze, Adroit, Kitchen, and "
            "Locomotion only."
        )

    gamma = float(getattr(buffer, "gamma", 0.99))
    env = _make_legacy_env(env_id)
    try:
        loader = {
            "antmaze": _official_calql_antmaze_dataset,
            "adroit": _standard_d4rl_dataset,
            "kitchen": _official_calql_kitchen_dataset,
            "locomotion": _standard_d4rl_dataset,
        }[family]
        dataset = loader(
            env,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            clip_action=0.99999,
            gamma=gamma,
            num_episodes=num_episodes,
        )
    finally:
        env.close()

    device = buffer.storage_device
    rewards = torch.as_tensor(dataset["rewards"], device=device)
    successes = (
        (rewards > reward_bias).float()
        if family == "antmaze" and hasattr(buffer, "_step_success")
        else None
    )
    return _add_flat_transitions(
        buffer,
        torch.as_tensor(dataset["observations"], device=device),
        torch.as_tensor(dataset["next_observations"], device=device),
        torch.as_tensor(dataset["actions"], device=device),
        rewards,
        torch.as_tensor(dataset["terminals"], device=device),
        torch.as_tensor(dataset["mc_returns"], device=device),
        successes,
        episode_ends=torch.as_tensor(dataset["episode_end"], device=device).bool(),
    )


class D4RLLegacyDatasetBackend(DatasetBackend):
    @classmethod
    def infer_specs(cls, req: DatasetRequest):
        return infer_specs_from_d4rl_legacy(req.path)

    @classmethod
    def load(cls, buffer, req: DatasetRequest) -> int:
        return load_d4rl_legacy_dataset_to_replay_buffer(
            buffer,
            req.path,
            num_episodes=req.num_traj,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            success_key=req.success_key,
        )


register_dataset_backend("d4rl_legacy", D4RLLegacyDatasetBackend)
