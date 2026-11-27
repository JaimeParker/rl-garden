"""OPAL learning settings shared by constructors and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OPALParams:
    """Defaults matching OPAL construction."""

    skill_dim: int = 8
    chunk_size: int = 4
    hidden_size: int = 256
    vae_hidden_dims: tuple[int, ...] = (256, 256)
    kl_coef: float = 0.1
    lr: float = 0.0003
