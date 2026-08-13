"""Adapter for the Gym 0.23 D4RL environments used by official Cal-QL."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.envs.vector_env import TorchVectorEnvAdapter


def _convert_space(space: Any) -> spaces.Space:
    if hasattr(space, "spaces"):
        return spaces.Dict({key: _convert_space(value) for key, value in space.spaces.items()})
    if hasattr(space, "n") and not hasattr(space, "low"):
        return spaces.Discrete(int(space.n))
    if hasattr(space, "low") and hasattr(space, "high"):
        dtype = np.dtype(space.dtype)
        return spaces.Box(
            low=np.asarray(space.low, dtype=dtype),
            high=np.asarray(space.high, dtype=dtype),
            shape=space.shape,
            dtype=dtype,
        )
    raise TypeError(f"Unsupported legacy Gym space: {type(space).__name__}")


class _GymV21Adapter(gym.Env):
    """Translate an old Gym reset/step API without changing environment logic."""

    def __init__(self, env: Any) -> None:
        super().__init__()
        self.legacy_env = env
        self.observation_space = _convert_space(env.observation_space)
        self.action_space = _convert_space(env.action_space)
        self.metadata = dict(getattr(env, "metadata", {}))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.legacy_env.seed(seed)
        return self.legacy_env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.legacy_env.step(action)
        return obs, reward, bool(done), False, info

    def render(self):
        return self.legacy_env.render()

    def close(self) -> None:
        self.legacy_env.close()


class _AdroitBinaryTermination(gym.Wrapper):
    """Treat the legacy Binary task's goal signal as a true termination."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal_achieved = bool(np.asarray(info.get("goal_achieved", False)).reshape(-1)[0])
        return obs, reward, terminated or goal_achieved, truncated, info


class _D4RLEpisodeMetrics(gym.Wrapper):
    """Attach task-family metrics to terminal episode statistics."""

    def __init__(self, env: gym.Env, env_id: str, legacy_env: Any) -> None:
        super().__init__(env)
        self.env_id = env_id.lower()
        self.legacy_env = legacy_env
        self.last_reward = 0.0

    def reset(self, **kwargs):
        self.last_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_reward = float(np.asarray(reward).reshape(-1)[0])
        if (terminated or truncated) and "episode" in info:
            info = dict(info)
            episode = dict(info["episode"])
            episode_return = float(np.asarray(episode["r"]).reshape(-1)[0])
            if "antmaze" in self.env_id:
                episode["success_at_end"] = np.float32(episode_return > 0.0)
            elif "binary" in self.env_id:
                episode["success_at_end"] = np.float32(
                    bool(np.asarray(info.get("goal_achieved", False)).reshape(-1)[0])
                )
            elif "kitchen" in self.env_id:
                episode["num_stages_solved"] = np.float32(self.last_reward)
                episode["normalized_score"] = np.float32(self.last_reward * 25.0)
            elif self.env_id.startswith(("pen-", "door-", "relocate-")):
                normalized = self.legacy_env.get_normalized_score(episode_return)
                episode["normalized_score"] = np.float32(float(normalized) * 100.0)
            info["episode"] = episode
        return obs, reward, terminated, truncated, info


def _make_legacy_env(env_id: str):
    try:
        import d4rl  # noqa: F401
        import gym as legacy_gym
    except ImportError as exc:
        raise ImportError(
            "The d4rl_legacy backend requires the `d4rl-legacy` optional "
            "dependencies. Install rl-garden with `[d4rl-legacy]`."
        ) from exc
    if "binary" in env_id.lower():
        try:
            import mj_envs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Adroit Binary environments require the `d4rl-legacy-manip` "
                "optional dependencies."
            ) from exc
    return legacy_gym.make(env_id).unwrapped


def _make_env_fn(env_id: str):
    def _make_env():
        legacy_env = _make_legacy_env(env_id)
        max_episode_steps = int(legacy_env.spec.max_episode_steps)
        env: gym.Env = _GymV21Adapter(legacy_env)
        if "binary" in env_id.lower():
            env = _AdroitBinaryTermination(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return _D4RLEpisodeMetrics(env, env_id, legacy_env)

    return _make_env


def make_d4rl_legacy_env(cfg):
    env_fns = [_make_env_fn(cfg.env_id) for _ in range(cfg.num_envs)]
    vec_env = gym.vector.SyncVectorEnv(
        env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    adapter: gym.vector.VectorEnv = TorchVectorEnvAdapter(vec_env, device=cfg.device)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import (
            RewardScaleBiasVectorWrapper,
        )

        adapter = RewardScaleBiasVectorWrapper(
            adapter, scale=cfg.reward_scale, bias=cfg.reward_bias
        )
    return adapter
