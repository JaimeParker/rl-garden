"""``BatchedEnvWrapper``: base class of every wrapper above the numpy-to-torch
adapter.

Which base a wrapper uses follows from where it sits:

- at and below ``TorchVectorEnvAdapter``, and directly on a torch-native vector
  env, wrappers are ``gymnasium.vector.VectorWrapper``s (or ``gym.Wrapper``s on
  each single env before vectorization, and on a ManiSkill env below
  ``ManiSkillVectorEnv``);
- above that, every wrapper derives from ``BatchedEnvWrapper``. It accepts any
  batched env, including another ``BatchedEnvWrapper``, so these wrappers stack
  in any order. Orders can differ in meaning: a reward transform inside a
  wrapper that sums rewards adds its bias once per summed step, outside it once
  per sum. The result is a ``gym.Wrapper``, never a ``VectorEnv``; code above
  the adapter must not test ``isinstance(env, VectorEnv)``.

A batched env is a ``gymnasium.Env`` that carries ``num_envs`` (the Isaac Lab
adapter, a real-robot env), a ``gymnasium.vector.VectorEnv``
(``ManiSkillVectorEnv``, ``TorchVectorEnvAdapter``, the mujoco_warp tasks), or
an object with the same attributes (``OfflineEnvSpec``). Gymnasium keeps the two
types unrelated: ``gym.Wrapper`` accepts only the first, ``VectorWrapper`` only
the second.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym


class BatchedEnvWrapper(gym.Wrapper):
    """``gym.Wrapper`` without the ``gymnasium.Env`` check, forwarding every
    attribute it does not define (``num_envs``, ``device``,
    ``single_observation_space``, ``single_action_space``,
    ``post_update_sync``, ...) to the wrapped env on each access.

    A subclass that changes the observation or action contract must set both
    the ``single_*_space`` and the batched space, or the wrapped env's space is
    the one its callers read.

    Writes do not forward: assigning an attribute, also through
    ``set_wrapper_attr``, sets it on this wrapper. Assign on the owning env.
    """

    def __init__(self, env: Any) -> None:
        # Exactly the attributes gym.Wrapper.__init__ sets; not called, because
        # it rejects a VectorEnv. A subclass over a gym.Env may call that one.
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None
        self._cached_spec = None

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails. "env" is guarded so a lookup
        # before __init__ sets it (e.g. during unpickling) cannot recurse.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
