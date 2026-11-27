"""HILP learning settings shared by constructors and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HILPParams:
    """Defaults matching HILP construction."""

    skill_dim: int = 32
    value_hidden_dims: tuple[int, ...] = (512, 512, 512)
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512)
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.95
    skill_expectile: float = 0.9
    skill_temperature: float = 10.0
    skill_discount: float = 0.99
    p_currgoal: float = 0.0
    p_trajgoal: float = 0.625
    p_randomgoal: float = 0.375
    lr: float = 0.0003
