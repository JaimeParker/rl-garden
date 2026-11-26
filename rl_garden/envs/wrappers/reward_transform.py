"""Per-step reward scale/bias wrapper.

Applies ``reward = scale * raw_reward + bias`` on each ``step()``. Mirrors
``ScaledRewardWrapper`` from the WSRL JAX reference; needed when reproducing
D4RL antmaze results where rewards are 0/1 but Cal-QL expects ``-r_neg`` style
returns. The same transform must be applied consistently to online env rewards
and offline H5-loaded rewards via ``load_h5_dataset_to_replay_buffer``.
"""
from __future__ import annotations

from typing import Any

from rl_garden.envs.wrappers.batched_env import BatchedEnvWrapper


class RewardScaleBiasWrapper(BatchedEnvWrapper):
    """Multiply reward by ``scale`` and add ``bias`` on each step.

    Works on tensor, numpy and scalar rewards.
    """

    def __init__(self, env: Any, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__(env)
        self.reward_scale = float(scale)
        self.reward_bias = float(bias)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward_scale * reward + self.reward_bias, terminated, truncated, info


def apply_reward_transform(env: Any, *, scale: float, bias: float) -> Any:
    """``env`` in a ``RewardScaleBiasWrapper``, or ``env`` itself when the
    transform is the identity."""
    if scale == 1.0 and bias == 0.0:
        return env
    return RewardScaleBiasWrapper(env, scale=scale, bias=bias)
