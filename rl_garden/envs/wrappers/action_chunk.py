"""Batched action-chunking wrapper for GPU-vectorized, SAME_STEP-autoreset
envs.

Executes ``act_steps`` inner ``env.step()`` calls per outer ``step()`` call
and returns one aggregated transition -- mirrors ``3rd_party/dppo``'s own
``env/gym_utils/wrapper/multi_step.py::MultiStep``, which hides action
chunking entirely inside the env so DPPO's training loop never needs to
issue more than one ``env.step()`` per rollout step. That reference wrapper
targets a single non-vectorized env and simply stops stepping once it's
done; a batched vector env can't skip individual envs from one shared
``step()`` call, so this version tracks a per-env ``active`` mask instead:
an env past its done sub-step keeps getting stepped (unavoidable -- it's
already SAME_STEP-autoreset into a new episode by the underlying env), but
its reward stops counting toward the chunk sum, and its
``final_observation``/``final_info`` are frozen at the sub-step it actually
terminated on rather than overwritten by the new episode's data.
``final_observation`` is reported only if the inner env reported it: a
post-reset observation is never passed off as a terminal one. An env that ends
a second time within one chunk loses that episode end; such drops are counted
in ``dropped_episode_ends`` (see ``ChunkAccumulator``).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.envs.wrappers._batch_utils import ChunkAccumulator


def _repeat_bound(bound: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(bound, (n,) + bound.shape).copy()


class ActionChunkWrapper(gym.Wrapper):
    """``action`` has shape ``(num_envs, act_steps) + single_action_space.shape``.

    Reward is summed across the chunk; ``terminated``/``truncated`` are
    combined with logical-or across sub-steps.
    """

    def __init__(self, env: gym.Env, act_steps: int) -> None:
        if act_steps < 1:
            raise ValueError(f"act_steps must be >= 1, got {act_steps}.")
        super().__init__(env)
        self.act_steps = int(act_steps)
        # gym.Wrapper has no __getattr__ fallback for these rl-garden/ManiSkill
        # vector-env conventions (single_observation_space, num_envs, ...) --
        # every other wrapper in this package is applied *before* the final
        # adapter that sets them, but this one must be one of the outermost
        # (it changes the action contract itself), so it forwards them
        # explicitly instead of relying on attribute delegation that doesn't
        # exist in this gymnasium version.
        self.num_envs = env.num_envs
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        single = env.single_action_space
        self.single_action_space = spaces.Box(
            low=_repeat_bound(single.low, self.act_steps),
            high=_repeat_bound(single.high, self.act_steps),
            shape=(self.act_steps,) + single.shape,
            dtype=single.dtype,
        )
        self.action_space = batch_space(self.single_action_space, env.num_envs)
        self._chunk = ChunkAccumulator()

    @property
    def dropped_episode_ends(self) -> torch.Tensor | None:
        return self._chunk.dropped_episode_ends

    def step(self, action: torch.Tensor):
        self._chunk.begin()
        for i in range(self.act_steps):
            obs, reward, terminated, truncated, infos = self.env.step(action[:, i])
            self._chunk.add(reward, terminated, truncated, infos)
        return obs, *self._chunk.finish(infos)
