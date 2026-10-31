"""Binary success-classifier reward wrapper (SERL's reward-classifier
component): replaces an env's placeholder/hand-written reward with a
learned binary success signal computed from the current observation, and
(by default, matching SERL) ends the episode on success.
"""
from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode

ClassifierFn = Callable[[dict], torch.Tensor]


class RewardClassifierWrapper(gym.Wrapper):
    """``classifier_fn(obs) -> success_prob`` (batched, any shape that
    reshapes to ``(num_envs,)``), thresholded into a ``{0, 1}`` reward.

    This wrapper ORs success into ``terminated`` but supplies no
    ``final_observation`` / ``final_info`` and never resets the env, so it is
    only valid on an env without autoreset (the caller resets on
    ``terminated`` / ``truncated``). Construction raises ``ValueError`` if
    the wrapped env declares ``AutoresetMode.SAME_STEP``. The classifier may
    run on a different device than the env; its output is moved to the
    env's device.
    """

    def __init__(
        self,
        env: gym.Env,
        classifier_fn: ClassifierFn,
        threshold: float = 0.5,
        terminate_on_success: bool = True,
    ) -> None:
        if env.metadata.get("autoreset_mode") == AutoresetMode.SAME_STEP:
            raise ValueError(
                "RewardClassifierWrapper requires an env without autoreset; "
                "the caller resets on terminated/truncated."
            )
        super().__init__(env)
        self.classifier_fn = classifier_fn
        self.threshold = threshold
        self.terminate_on_success = terminate_on_success

    def __getattr__(self, name: str):
        # gymnasium.Wrapper (>=1.0) no longer forwards arbitrary attributes to
        # ``self.env`` -- but this repo's env-backend contract (num_envs,
        # single_observation_space, ...) relies on direct attribute access,
        # not ``get_wrapper_attr()``, so this wrapper must still be
        # transparent to algorithm code built against the unwrapped env.
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        with torch.no_grad():
            success_prob = self.classifier_fn(obs).reshape(-1).to(terminated.device)
        success = success_prob > self.threshold
        classified_reward = success.to(reward.dtype)
        info = dict(info)
        info["classifier_success_prob"] = success_prob
        if self.terminate_on_success:
            terminated = terminated | success
        return obs, classified_reward, terminated, truncated, info
