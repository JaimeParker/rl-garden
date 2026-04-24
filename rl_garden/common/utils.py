"""Misc utilities: seeding, device resolution, polyak updates, lr schedules."""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def get_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """In-place Polyak update: target = tau * param + (1 - tau) * target."""
    for p, tp in zip(params, target_params):
        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def constant_schedule(value: float):
    def _f(_progress_remaining: float) -> float:
        return value

    return _f
