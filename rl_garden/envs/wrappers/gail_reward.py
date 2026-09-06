"""GAIL reward-substitution wrapper: replaces env reward with a
discriminator-derived reward computed from ``(obs_before_action, action)``.

Same family as ``rl_garden.envs.wrappers.reward_classifier.RewardClassifierWrapper``
(replace env reward with a learned model's output at ``step()``), but the
discriminator needs the *pre-step* observation plus the action rather than
just the post-step observation, so it caches ``obs`` across ``reset()``/
``step()`` calls instead. Built on ``gym.vector.VectorWrapper`` rather than
``gym.Wrapper`` -- same reasoning as
``rl_garden.envs.wrappers.reward_transform.RewardScaleBiasVectorWrapper``:
``gym.Wrapper.__init__`` asserts its wrapped env is a ``gymnasium.Env``, but
``gymnasium.vector.VectorEnv`` (what every env backend GAIL targets actually
returns) does not subclass it.
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import torch

# gymnasium >=1.0 exposes the vector wrapper base class as ``gym.vector.VectorWrapper``;
# gymnasium 0.29.x (pinned by ManiSkill) only has the equivalent ``VectorEnvWrapper``.
_VectorWrapperBase = getattr(gym.vector, "VectorWrapper", None) or gym.vector.VectorEnvWrapper

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class GAILRewardWrapper(_VectorWrapperBase):
    """``reward_fn(obs, action) -> reward`` (batched, shape ``(num_envs,)``),
    substituted for the wrapped vector env's own reward every ``step()``."""

    def __init__(self, env: gym.vector.VectorEnv, reward_fn: RewardFn) -> None:
        super().__init__(env)
        self.reward_fn = reward_fn
        self._last_obs: torch.Tensor | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.clone()
        return obs, info

    def step(self, actions):
        with torch.no_grad():
            reward = self.reward_fn(self._last_obs, actions)
        # Clone before storing: some GPU vec envs reuse the same obs buffer
        # across step() calls (same aliasing concern off_policy.py's
        # learn()/_clone_obs() guards against for real_next_obs).
        obs, _, terminated, truncated, info = self.env.step(actions)
        self._last_obs = obs.clone()
        return obs, reward, terminated, truncated, info
