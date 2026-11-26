"""GAIL reward-substitution wrapper: replaces env reward with a
discriminator-derived reward computed from ``(obs_before_action, action)``.

Same family as ``rl_garden.envs.wrappers.reward_classifier.RewardClassifierWrapper``
(replace env reward with a learned model's output at ``step()``), but the
discriminator needs the *pre-step* observation plus the action rather than
just the post-step observation, so it caches ``obs`` across ``reset()``/
``step()`` calls instead. ``GAIL._setup_model`` applies it to the training env only.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from rl_garden.envs.wrappers.batched_env import BatchedEnvWrapper

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _clone_obs(obs):
    """Clone a Dict or plain-tensor obs."""
    if isinstance(obs, dict):
        return {k: v.clone() for k, v in obs.items()}
    return obs.clone()


class GAILRewardWrapper(BatchedEnvWrapper):
    """``reward_fn(obs, action) -> reward`` (batched, shape ``(num_envs,)``),
    substituted for the wrapped vector env's own reward every ``step()``."""

    def __init__(self, env: Any, reward_fn: RewardFn) -> None:
        super().__init__(env)
        self.reward_fn = reward_fn
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = _clone_obs(obs)
        return obs, info

    def step(self, actions):
        with torch.no_grad():
            reward = self.reward_fn(self._last_obs, actions)
        # Clone before storing: some GPU vec envs reuse the same obs buffer
        # across step() calls (same aliasing concern off_policy.py's
        # learn()/_clone_obs() guards against for real_next_obs).
        obs, _, terminated, truncated, info = self.env.step(actions)
        self._last_obs = _clone_obs(obs)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
