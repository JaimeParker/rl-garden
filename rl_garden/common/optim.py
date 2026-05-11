"""Optimizer and LR-scheduler factories.

Mirrors the WSRL JAX reference's ``make_optimizer`` (warmup-cosine + AdamW)
in PyTorch. The factory routes through ``torch.optim.AdamW`` when
``weight_decay > 0`` or ``use_adamw=True`` and otherwise uses ``Adam``.

LR schedules supported:
- ``"constant"``: returns ``None`` (caller must skip ``scheduler.step()``)
- ``"linear_warmup"``: linear ramp from 0 to peak over ``warmup_steps``
- ``"warmup_cosine"``: linear warmup followed by cosine decay to
  ``min_lr_ratio * peak`` over ``decay_steps``

All schedules step per gradient-update; ``warmup_steps`` and ``decay_steps`` are
counted in optimizer steps, not env steps.
"""
from __future__ import annotations

import math
from typing import Iterable, Literal, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

ScheduleType = Literal["constant", "linear_warmup", "warmup_cosine"]


def make_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    *,
    weight_decay: float = 0.0,
    use_adamw: bool = False,
) -> torch.optim.Optimizer:
    """Build an Adam/AdamW optimizer.

    AdamW is selected automatically when ``weight_decay > 0`` (since plain
    ``Adam`` couples weight decay into gradient via L2 reg, which is usually
    not what callers want).
    """
    if use_adamw or weight_decay > 0:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule_type: ScheduleType = "constant",
    warmup_steps: int = 0,
    decay_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> Optional[LambdaLR]:
    """Return a ``LambdaLR`` scheduler matching the WSRL reference shapes.

    Returns ``None`` for ``schedule_type="constant"`` so callers can cheaply
    skip ``scheduler.step()`` on each gradient update.
    """
    if schedule_type == "constant":
        return None
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if schedule_type == "warmup_cosine" and decay_steps <= 0:
        raise ValueError(
            "warmup_cosine requires decay_steps > 0; "
            f"got decay_steps={decay_steps}."
        )
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1]; got {min_lr_ratio}.")

    if schedule_type == "linear_warmup":

        def lr_lambda(step: int) -> float:
            if warmup_steps == 0:
                return 1.0
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

    elif schedule_type == "warmup_cosine":

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type!r}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
