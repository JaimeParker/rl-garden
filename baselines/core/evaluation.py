"""Episode-rollout evaluation helpers shared across baseline orchestrators."""
from __future__ import annotations

import math
import time

import numpy as np

from baselines.core.env_bridge import GymnasiumEnvBridge


def wilson_interval(successes, episodes, z=1.959963984540054):
    if episodes <= 0:
        return [float("nan"), float("nan")]
    proportion = float(successes) / float(episodes)
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return [center - radius, center + radius]


def summarize_episodes(returns, lengths, normalized_returns=None):
    successes = int(sum(value > 0.5 for value in returns))
    result = {
        "episodes": len(returns),
        "successes": successes,
        "success_rate": successes / float(len(returns)),
        "success_rate_wilson95": wilson_interval(successes, len(returns)),
        "average_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "average_traj_length": float(np.mean(lengths)),
        "traj_length_std": float(np.std(lengths)),
    }
    if normalized_returns is not None:
        result["average_normalized_return"] = float(np.mean(normalized_returns))
    return result


def evaluate_bridge_policy(policy, bridge_kwargs, episodes):
    """Run ``episodes`` rollouts of ``policy`` against a ``GymnasiumEnvBridge``."""
    returns = []
    lengths = []
    started = time.time()
    with GymnasiumEnvBridge(**bridge_kwargs) as env:
        for _ in range(episodes):
            observation = env.reset()
            episode_return = 0.0
            episode_length = 0
            for _ in range(env.horizon):
                action = policy(
                    observation.reshape(1, -1), deterministic=True
                ).reshape(-1)
                observation, reward, terminated, truncated = env.step(action)
                episode_return += reward
                episode_length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
    result = summarize_episodes(returns, lengths)
    result["elapsed_seconds"] = time.time() - started
    return result


def evaluate_legacy_gym_policy(policy, env_id, episodes, seed, *, observation_adapter=None):
    """Run ``episodes`` rollouts of ``policy`` against a classic ``gym.make(env_id)`` env.

    ``observation_adapter(raw_observation, env)`` transforms each raw
    observation before it's passed to ``policy`` (e.g. Cal-QL's AntMaze
    goal-concatenation trick) -- defaults to identity for baselines that
    don't need one.
    """
    import d4rl  # noqa: F401
    import gym

    if observation_adapter is None:
        observation_adapter = lambda raw, env: raw  # noqa: E731

    env = gym.make(env_id).unwrapped
    if hasattr(env, "seed"):
        env.seed(seed)
    returns = []
    normalized_returns = []
    lengths = []
    started = time.time()
    try:
        for _ in range(episodes):
            raw_observation = env.reset()
            observation = observation_adapter(raw_observation, env)
            episode_return = 0.0
            episode_length = 0
            for _ in range(env.spec.max_episode_steps):
                action = policy(
                    observation.reshape(1, -1), deterministic=True
                ).reshape(-1)
                raw_observation, reward, done, _ = env.step(action)
                observation = observation_adapter(raw_observation, env)
                episode_return += float(reward)
                episode_length += 1
                if done:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            normalized_returns.append(env.get_normalized_score(episode_return))
    finally:
        env.close()
    result = summarize_episodes(returns, lengths, normalized_returns)
    result["elapsed_seconds"] = time.time() - started
    return result
