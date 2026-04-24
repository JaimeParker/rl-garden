"""Thin TensorBoard + optional wandb logger. Lifted in spirit from ManiSkill sac.py."""
from __future__ import annotations

from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
        self,
        tensorboard: Optional[SummaryWriter] = None,
        log_wandb: bool = False,
    ) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.log_wandb:
            import wandb  # lazy import

            wandb.log({tag: value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
