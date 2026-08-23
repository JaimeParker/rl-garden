"""Sequential multi-stage success-classifier reward wrapper (HIL-SERL's
``MultiStageBinaryRewardClassifierWrapper``, ``3rd_party/hil-serl/
serl_robot_infra/franka_env/envs/wrappers.py``): unlike
``RewardClassifierWrapper``'s single-stage case, this tracks several
per-stage classifiers, each of which can fire at most once (cumulative, not
re-triggerable), and only ends the episode once every stage has fired.
"""
from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import torch

ClassifierFn = Callable[[dict], torch.Tensor]


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    """``classifier_fns[i](obs) -> success_prob`` (batched, any shape that
    reshapes to ``(num_envs,)``) for each stage. A stage's reward (1.0) is
    granted exactly once, the first step its classifier crosses
    ``threshold``; the step reward is the sum of newly-fired stages this
    step. Episode ends once every stage has fired, in addition to whatever
    the wrapped env's own ``terminated`` already signals.
    """

    def __init__(
        self,
        env: gym.Env,
        classifier_fns: Sequence[ClassifierFn],
        threshold: float = 0.75,
    ) -> None:
        super().__init__(env)
        self.classifier_fns = tuple(classifier_fns)
        self.threshold = threshold
        self._received: torch.Tensor | None = None  # (num_envs, num_stages) bool

    def __getattr__(self, name: str):
        # See RewardClassifierWrapper -- gymnasium.Wrapper (>=1.0) no longer
        # forwards arbitrary attributes to self.env.
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._received = torch.zeros(
            self.num_envs, len(self.classifier_fns), dtype=torch.bool
        )
        info = dict(info)
        info["succeed"] = self._received.all(dim=-1)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        with torch.no_grad():
            probs = torch.stack(
                [fn(obs).reshape(-1) for fn in self.classifier_fns], dim=-1
            )
        newly = (probs >= self.threshold) & ~self._received
        self._received = self._received | newly
        stage_reward = newly.to(reward.dtype).sum(dim=-1)
        all_received = self._received.all(dim=-1)
        info = dict(info)
        info["succeed"] = all_received
        return obs, reward + stage_reward, terminated | all_received, truncated, info
