"""Action repeat for batched, SAME_STEP-autoreset torch vector envs.

One outer ``step(action)`` runs ``repeat`` inner ``env.step(action)`` calls
with the same action and returns one transition per decision, for tasks whose
control holds a command for several env steps (e.g. a task that holds a
trajectory chunk). Unlike ``ActionChunkWrapper`` the action space is
unchanged: the policy picks one action, the wrapper repeats it.

Contract (the inner env and the output both follow it):

- SAME_STEP autoreset: an env that ends on a sub-step is reset inside that
  sub-step and its next observation belongs to the new episode.
- ``reward`` is the sum over the sub-steps up to and including the one the
  env ends on. An env that ends on sub-step ``j < repeat`` keeps running the
  old action in its new episode for the remaining ``repeat - j`` sub-steps
  (a batched env cannot skip single envs); those rewards are not counted,
  and the returned ``obs`` is from the new episode after those sub-steps.
- ``terminated``/``truncated`` are the flags of the env's first episode end
  in the chunk. ``final_info`` and ``final_observation`` are frozen at that
  sub-step; ``_final_info``/``_final_observation`` mark every env that ended.
  ``final_observation`` is reported only if the inner env reported it on
  some sub-step: a post-reset observation is never passed off as a terminal
  one.
- Inner episode statistics (e.g. ``EpisodeStatistics`` in the Isaac Lab
  adapter) count env steps, including the uncounted sub-steps above, so a
  logged episode ``return``/``episode_len`` is per env step, not per decision.
- An env that ends a second time within the same chunk loses that episode
  end: its flags, ``final_info`` and ``final_observation`` stay those of the
  first. Such drops are counted per env on the device in
  ``dropped_episode_ends`` (allocated on the first step, cumulative since
  then, not cleared by ``reset()``); episodes much longer than ``repeat``
  keep it at zero.
- ``infos["log"]`` is the whole dict from the last sub-step that reported
  one. A task that refreshes ``extras["log"]`` every N env steps needs
  ``N >= repeat``, or the earlier refreshes in a chunk are lost.
- Every inner ``step()`` must go through this wrapper. Code that keeps a
  step count aligned to the decision grid (e.g. a chunk hold timer counted
  in env steps) relies on it.

No host sync in ``step()``: all masking is tensor ops on the env device.
Attributes not set here (``num_envs``, ``device``, ``single_action_space``,
``single_observation_space``, ``post_update_sync``, ...) are read from the
wrapped env on every access, so a wrapper above that rewrites the inner
env's observation space (``ImageFrameStackWrapper``) is seen here too.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from rl_garden.envs.wrappers._batch_utils import ChunkAccumulator


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for ``repeat`` env steps and sum the rewards (see
    module docstring for the episode-boundary contract)."""

    def __init__(self, env: gym.Env, repeat: int) -> None:
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}.")
        super().__init__(env)
        self.repeat = int(repeat)
        self._chunk = ChunkAccumulator()

    @property
    def dropped_episode_ends(self) -> torch.Tensor | None:
        return self._chunk.dropped_episode_ends

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails. "env" is guarded so a lookup
        # before __init__ sets it (e.g. during unpickling) cannot recurse.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, action: Any):
        self._chunk.begin()
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, infos = self.env.step(action)
            self._chunk.add(reward, terminated, truncated, infos)
        return obs, *self._chunk.finish(infos)
