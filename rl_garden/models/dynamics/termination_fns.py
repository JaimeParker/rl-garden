"""Per-env termination heuristics for dynamics-model rollouts. Torch port of
`3rd_party/Uni-O4/transition_model/utils/termination_fns.py`, scoped to the
three D4RL locomotion tasks already in scope for BPPO/Uni-O4
(halfcheetah/hopper/walker2d) -- exact thresholds preserved.
"""
from __future__ import annotations

from typing import Callable

import torch

TerminationFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def termination_fn_halfcheetah(
    obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    del obs, action
    not_done = torch.all((next_obs > -100) & (next_obs < 100), dim=-1)
    return (~not_done).unsqueeze(-1)


def termination_fn_hopper(
    obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    del obs, action
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(dim=-1)
        & (next_obs[:, 1:].abs() < 100).all(dim=-1)
        & (height > 0.7)
        & (angle.abs() < 0.2)
    )
    return (~not_done).unsqueeze(-1)


def termination_fn_walker2d(
    obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    del obs, action
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.all((next_obs > -100) & (next_obs < 100), dim=-1)
        & (height > 0.8)
        & (height < 2.0)
        & (angle > -1.0)
        & (angle < 1.0)
    )
    return (~not_done).unsqueeze(-1)


_TERMINATION_FNS: dict[str, TerminationFn] = {
    "halfcheetah": termination_fn_halfcheetah,
    "hopper": termination_fn_hopper,
    "walker2d": termination_fn_walker2d,
}


def get_termination_fn(task: str) -> TerminationFn:
    for key, fn in _TERMINATION_FNS.items():
        if key in task:
            return fn
    raise ValueError(
        f"No termination_fn for task {task!r}; supported: "
        f"{sorted(_TERMINATION_FNS)} (matches BPPO/Uni-O4's locomotion scope)."
    )
