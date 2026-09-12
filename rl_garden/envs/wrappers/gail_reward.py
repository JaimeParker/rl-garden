"""GAIL reward-substitution wrapper: replaces env reward with a
discriminator-derived reward computed from ``(obs_before_action, action)``.

Same family as ``rl_garden.envs.wrappers.reward_classifier.RewardClassifierWrapper``
(replace env reward with a learned model's output at ``step()``), but the
discriminator needs the *pre-step* observation plus the action rather than
just the post-step observation, so it caches ``obs`` across ``reset()``/
``step()`` calls instead.

Plain duck-typed proxy (attribute passthrough via ``__getattr__``), not a
``gym.vector.VectorWrapper`` subclass -- unlike
``rl_garden.envs.wrappers.reward_transform.RewardScaleBiasVectorWrapper``,
this wraps ``self.env`` from inside ``BaseAlgorithm``/``GAIL._setup_model``,
where ``self.env`` is already the boundary's own
``rl_garden.envs.wrappers.dict_state.VectorizedDictStateWrapper`` for any
backend that starts out Box -- itself a duck-typed proxy, not a genuine
``gymnasium.vector.VectorEnv`` instance, which ``VectorWrapper.__init__``
hard-asserts (mirrors ``VectorizedDictStateWrapper``'s own reasoning for not
subclassing ``gym.Wrapper``/``VectorWrapper``).
"""
from __future__ import annotations

from typing import Callable

import torch

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _clone_obs(obs):
    """Clone a Dict or plain-tensor obs (the env boundary always normalizes
    to Dict now, but this stays generic rather than assuming a key set)."""
    if isinstance(obs, dict):
        return {k: v.clone() for k, v in obs.items()}
    return obs.clone()


class GAILRewardWrapper:
    """``reward_fn(obs, action) -> reward`` (batched, shape ``(num_envs,)``),
    substituted for the wrapped vector env's own reward every ``step()``."""

    def __init__(self, env, reward_fn: RewardFn) -> None:
        self.env = env
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

    def __getattr__(self, name: str):
        # Only reached when normal attribute lookup (instance __dict__, class
        # dict) fails -- ``self.env`` is set in __init__ and found there
        # directly, so this does not recurse. Guarded anyway in case an
        # instance is accessed before __init__ runs (e.g. during unpickling).
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
