"""On-policy buffer for teacher-student policy distillation.

Deliberately NOT a ``RolloutBuffer`` subclass: ``RolloutBuffer``/
``RecurrentRolloutBuffer`` are committed to PPO's field set (``values``,
``log_probs``, ``advantages``, ``returns``) that distillation never produces
-- there is no critic and no reward-based objective (see
``rl_garden/algorithms/policy_distillation.py``'s module docstring). This
buffer stores only what a pure action-regression loss needs: the raw obs and
the frozen teacher's action, computed once per rollout step (cheap, since the
teacher is frozen -- recomputing it every gradient step would cost one extra
teacher forward pass per minibatch per epoch for no benefit).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.dict_buffer import DictArray
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.types import Obs


@dataclass
class DistillationRolloutBufferSample:
    obs: Obs
    teacher_actions: torch.Tensor


class DistillationRolloutBuffer:
    """Fixed-size on-policy buffer with ``(num_steps, num_envs, ...)`` layout.

    ``dones`` is stored only for a future recurrent student (episode-boundary
    hidden-state resets); the stateless student built now never reads it.

    TODO(off-policy-variant): a growing/replay buffer (cf.
    ``rl_garden/buffers/demo_intervention.py``'s ``DemoInterventionMixin``)
    would replace this for a SAC-actor-style student.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_steps: int,
        num_envs: int,
        device: torch.device | str = "cuda",
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "DistillationRolloutBuffer requires a Dict observation space "
                "(needs both teacher and student obs keys), got "
                f"{type(observation_space)}."
            )
        if not isinstance(action_space, spaces.Box):
            raise TypeError(
                "DistillationRolloutBuffer only supports Box action spaces."
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.pos = 0
        self.full = False

        shape = (num_steps, num_envs)
        self.obs = DictArray(shape, observation_space, device=self.device)
        self.teacher_actions = torch.zeros(
            shape + tuple(action_space.shape), dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(shape, dtype=torch.float32, device=self.device)

    @property
    def buffer_size(self) -> int:
        return self.num_steps * self.num_envs

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def add(
        self, obs: Obs, teacher_actions: torch.Tensor, dones: torch.Tensor
    ) -> None:
        if self.pos >= self.num_steps:
            raise RuntimeError(
                "DistillationRolloutBuffer is full; call reset() before adding more."
            )
        assert isinstance(obs, dict)
        self.obs[self.pos] = {
            key: value.to(self.device) for key, value in obs.items()
        }
        self.teacher_actions[self.pos] = teacher_actions.to(self.device)
        self.dones[self.pos] = dones.reshape(self.num_envs).float().to(self.device)
        self.pos += 1
        self.full = self.pos == self.num_steps

    def get(
        self, batch_size: Optional[int] = None
    ) -> Iterator[DistillationRolloutBufferSample]:
        if not self.full:
            raise RuntimeError(
                "DistillationRolloutBuffer must be full before sampling."
            )
        if batch_size is None:
            batch_size = self.buffer_size
        flat_obs = flatten_leading_dims(self.obs)
        flat_teacher_actions = self.teacher_actions.reshape(
            (-1,) + self.teacher_actions.shape[2:]
        )

        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            mb_inds = indices[start : start + batch_size]
            yield DistillationRolloutBufferSample(
                obs=index_obs(flat_obs, mb_inds),
                teacher_actions=flat_teacher_actions[mb_inds],
            )
