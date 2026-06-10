"""Entropy-temperature tuning utilities for SAC."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

AlphaTuning = Literal["legacy_exp", "log_alpha", "lagrange_softplus"]


def parse_auto_alpha_init(ent_coef: float | str) -> tuple[bool, float]:
    """Return (autotune, init_value) from SAC's ent_coef argument."""
    if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
        init = 1.0
        if "_" in ent_coef:
            init = float(ent_coef.split("_", 1)[1])
        if init <= 0:
            raise ValueError(f"auto entropy coefficient init must be positive, got {init}.")
        return True, init
    return False, float(ent_coef)


def softplus_inverse(x: float) -> float:
    if x <= 0:
        raise ValueError(f"softplus inverse requires x > 0, got {x}.")
    # log(expm1(x)) = x + log1p(-exp(-x)); the latter form avoids expm1(x)
    # overflowing to inf for large x (e.g. x >= ~89 in float32).
    x_t = torch.tensor(float(x), dtype=torch.float32)
    return float((x_t + torch.log1p(-torch.exp(-x_t))).item())


class AlphaTuner(nn.Module):
    """Auto-tune SAC's entropy coefficient with selectable parameterization."""

    def __init__(
        self,
        tuning: AlphaTuning,
        init_value: float = 1.0,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if init_value <= 0:
            raise ValueError(f"alpha init value must be positive, got {init_value}.")
        if tuning not in ("legacy_exp", "log_alpha", "lagrange_softplus"):
            raise ValueError(f"Unknown alpha_tuning mode: {tuning!r}.")
        self.tuning = tuning
        if tuning in ("legacy_exp", "log_alpha"):
            value = torch.log(torch.ones(1, device=device) * init_value)
            self.log_alpha = nn.Parameter(value)
            self.raw_alpha = None
        else:
            value = torch.tensor([softplus_inverse(init_value)], device=device)
            self.raw_alpha = nn.Parameter(value)
            self.log_alpha = None

    def current_alpha(self) -> torch.Tensor:
        if self.tuning in ("legacy_exp", "log_alpha"):
            return self.log_alpha.exp()
        return F.softplus(self.raw_alpha)

    def loss(self, log_prob_detached: torch.Tensor, target_entropy: float) -> torch.Tensor:
        gap = (log_prob_detached + target_entropy).detach()
        alpha = self.current_alpha()
        if self.tuning == "legacy_exp":
            return -(alpha * gap).mean()
        if self.tuning == "log_alpha":
            return -(self.log_alpha * gap).mean()
        entropy = -log_prob_detached.detach().mean()
        return alpha * (entropy - target_entropy)
