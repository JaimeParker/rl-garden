"""Forward/backward (FWBW) reset-free direction wrapper, SERL's
``FWBWFrontCameraBinaryRewardClassifierWrapper`` equivalent: tracks which of
two task directions ("forward"/"backward") is currently active, flipping to
the other one whenever the current episode ends in success, so training can
run indefinitely without a human resetting the scene between "move object
A->B" and "move it back" runs.

Must wrap *outside* ``RewardClassifierWrapper`` (applied after it in the
stack) so ``terminated`` has already been set by the classifier's success
signal by the time this wrapper sees it.

This wrapper only tracks and reports the active direction via
``info["fwbw_direction"]`` -- actually running two different policies and
routing transitions to two different learners is the actor process's job:
``rl_garden.real_world.actor_loop.FWBWActorLoop`` reads this info key each
step to decide which of its two (policy, learner-sync-client) pairs to use,
since that coordination needs two full pairs and can't be owned by a single
env wrapper.
"""
from __future__ import annotations

from typing import Literal

import gymnasium as gym

Direction = Literal["forward", "backward"]


class FWBWResetFreeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, initial_direction: Direction = "forward") -> None:
        super().__init__(env)
        assert initial_direction in ("forward", "backward")
        self._direction: Direction = initial_direction

    @property
    def direction(self) -> Direction:
        return self._direction

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["fwbw_direction"] = self._direction
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        terminated_any = bool(terminated[0]) if hasattr(terminated, "__len__") else bool(terminated)
        if terminated_any:
            self._direction = "backward" if self._direction == "forward" else "forward"
        info["fwbw_direction"] = self._direction
        return obs, reward, terminated, truncated, info
