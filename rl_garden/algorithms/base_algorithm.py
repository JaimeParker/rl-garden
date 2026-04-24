"""Base algorithm: seeding / logger / device / checkpoint I/O.

Minimal counterpart to SB3's ``BaseAlgorithm``. We deliberately keep this
thin so GPU-parallel specifics live in ``OffPolicyAlgorithm``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from rl_garden.common.logger import Logger
from rl_garden.common.utils import get_device, seed_everything
from rl_garden.policies.base import BasePolicy


class BaseAlgorithm(ABC):
    policy: BasePolicy

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        self.device = get_device(device)
        self.logger = logger

        seed_everything(seed)
        self._global_step = 0
        self._global_update = 0

    @abstractmethod
    def _setup_model(self) -> None: ...

    @abstractmethod
    def learn(self, total_timesteps: int) -> "BaseAlgorithm": ...

    # --- checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        return {"policy": self.policy.state_dict(), "global_step": self._global_step}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self._global_step = int(sd.get("global_step", 0))
