"""``BoundTaskWorldModel``: a ``MultitaskWorldModel`` with one task id bound,
exposed through the plain ``WorldModel`` signatures.

``MultitaskWorldModel`` needs a ``task`` tensor on ``encode``/``step``/
``reward``, so generic consumers that only know the base signatures
(``rl_garden.world_models.imagine.imagine``, ``rl_garden.planners.mppi
.MPPIPlanner``) cannot drive it directly. This adapter fills ``task`` with
``task_idx`` for every batch column and forwards everything else unchanged.

The wrapped model is registered as the ``world_model`` submodule, so this
adapter's ``state_dict`` keys are ``world_model.*``. It holds no parameters
of its own and has no running observation normalizer
(``update_normalizer`` is a no-op). Do not use it as a ``BasePolicy``
extractor; it exists for planning/imagination over a trained multitask model.
"""
from __future__ import annotations

import dataclasses
from typing import Iterator, Optional

import torch
import torch.nn as nn

from rl_garden.common.types import Obs
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.batch import SequenceBatch
from rl_garden.world_models.multitask_latent_consistency import MultitaskWorldModel


class BoundTaskWorldModel(WorldModel):
    def __init__(self, world_model: MultitaskWorldModel, task_idx: int) -> None:
        super().__init__()
        if not 0 <= task_idx < world_model.num_tasks:
            raise ValueError(f"task_idx must be in [0, {world_model.num_tasks}), got {task_idx}.")
        self.world_model = world_model
        self.task_idx = int(task_idx)
        self.latent_dim = world_model.latent_dim
        # Read-only metadata the latent-consistency consumers (MPPIPlanner,
        # TD-MPC2 policies) take from a model.
        self.action_dim = world_model.action_dim
        self.num_bins = world_model.num_bins
        self.vmin = world_model.vmin
        self.vmax = world_model.vmax
        self.bin_size = world_model.bin_size
        self.episodic = world_model.episodic

    def _task_for(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,)`` task ids for an ``x`` of shape ``(B, D)`` or ``(T, B, D)``."""
        return torch.full((x.shape[-2],), self.task_idx, dtype=torch.long, device=x.device)

    def update_normalizer(self, obs: Obs) -> None:
        del obs

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode(obs, self._task_for(obs))

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        return self.world_model.initial_state(batch_size, device)

    def observe(
        self, state: Optional[State], action: Optional[torch.Tensor], embed: torch.Tensor, is_first: torch.Tensor
    ) -> State:
        return self.world_model.observe(state, action, embed, is_first, task=self._task_for(embed))

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        return self.world_model.step(state, action, sample=sample, task=self._task_for(state["z"]))

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        return self.world_model.reward(state, action, task=self._task_for(state["z"]))

    def reward_value(
        self, state: State, action: torch.Tensor, *, symlog_space: bool = False
    ) -> torch.Tensor:
        return self.world_model.reward_value(
            state, action, symlog_space=symlog_space, task=self._task_for(state["z"])
        )

    def features(self, state: State) -> torch.Tensor:
        return self.world_model.features(state)

    def parameter_groups(self) -> dict[str, Iterator[nn.Parameter]]:
        return self.world_model.parameter_groups()

    def model_loss(self, batch: SequenceBatch) -> tuple[dict[str, torch.Tensor], State]:
        task = torch.full(
            (batch.action.shape[1],), self.task_idx, dtype=torch.long, device=batch.action.device
        )
        return self.world_model.model_loss(dataclasses.replace(batch, task=task))
