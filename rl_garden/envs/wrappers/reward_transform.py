"""Per-step reward scale/bias wrapper.

Applies ``reward = scale * raw_reward + bias`` on each ``step()``. Mirrors
``ScaledRewardWrapper`` from the WSRL JAX reference; needed when reproducing
D4RL antmaze results where rewards are 0/1 but Cal-QL expects ``-r_neg`` style
returns. The same transform must be applied consistently to online env rewards
and offline H5-loaded rewards via ``load_maniskill_h5_to_replay_buffer``.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class RewardScaleBiasWrapper(gym.Wrapper):
    """Multiply reward by ``scale`` and add ``bias`` on each step.

    Works with ManiSkill envs that emit torch tensor rewards because the
    arithmetic ``scale * reward + bias`` is identical for tensors and scalars.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__(env)
        self.reward_scale = float(scale)
        self.reward_bias = float(bias)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward_scale * reward + self.reward_bias, terminated, truncated, info


class SuccessRewardOverrideWrapper(gym.Wrapper):
    """Override reward values for states marked successful by the environment."""

    def __init__(self, env: gym.Env, reward: float) -> None:
        super().__init__(env)
        self.success_reward = float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = info.get("success")
        if success is None:
            raise KeyError(
                "success_reward_override requires info['success'] on every step"
            )
        if torch.is_tensor(reward):
            success = torch.as_tensor(success, dtype=torch.bool, device=reward.device)
            reward = torch.where(
                success,
                torch.as_tensor(
                    self.success_reward, dtype=reward.dtype, device=reward.device
                ),
                reward,
            )
        elif isinstance(reward, np.ndarray):
            reward = np.where(success, self.success_reward, reward)
        elif bool(success):
            reward = self.success_reward
        return obs, reward, terminated, truncated, info
