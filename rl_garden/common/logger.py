"""Unified logger backend for TensorBoard / W&B / stdout-only modes."""
from __future__ import annotations

import os
from typing import Any, Literal, Optional

from torch.utils.tensorboard import SummaryWriter

LogType = Literal["tensorboard", "wandb", "none"]


class Logger:
    def __init__(
        self,
        tensorboard: Optional[SummaryWriter] = None,
        wandb_run: Optional[Any] = None,
        log_type: LogType = "none",
    ) -> None:
        self.writer = tensorboard
        self.wandb_run = wandb_run
        self.log_type = log_type

    @staticmethod
    def _parse_keywords(log_keywords: Optional[str]) -> list[str]:
        if log_keywords is None:
            return []
        tags: list[str] = []
        for raw in log_keywords.split(","):
            tag = raw.strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    @classmethod
    def create(
        cls,
        *,
        log_type: str,
        log_dir: str,
        run_name: str,
        config: Optional[dict[str, Any]] = None,
        start_time: Optional[str] = None,
        log_keywords: Optional[str] = None,
        wandb_project: str = "rl-garden",
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
    ) -> "Logger":
        normalized = log_type.lower()
        if normalized == "tensorboard":
            writer = SummaryWriter(os.path.join(log_dir, run_name))
            return cls(tensorboard=writer, log_type="tensorboard")
        if normalized == "none":
            return cls(log_type="none")
        if normalized != "wandb":
            raise ValueError(
                f"Unsupported log_type={log_type!r}. Choose from: tensorboard, wandb, none."
            )

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "log_type=wandb requires `wandb` to be installed. "
                "Install with: pip install -e '.[wandb]'"
            ) from exc

        api_key = os.getenv("WANDB_API_KEY") or getattr(getattr(wandb, "api", None), "api_key", None)
        if not api_key:
            raise RuntimeError(
                "log_type=wandb requires wandb authentication. "
                "Run `wandb login` or set WANDB_API_KEY."
            )

        tags = cls._parse_keywords(log_keywords)
        wandb_name = run_name
        if start_time:
            wandb_name = f"{wandb_name}__{start_time}"
        if tags:
            wandb_name = f"{wandb_name}__{'-'.join(tags)}"

        init_kwargs: dict[str, Any] = {
            "project": wandb_project,
            "name": wandb_name,
            "config": config or {},
        }
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity
        if wandb_group:
            init_kwargs["group"] = wandb_group
        if tags:
            init_kwargs["tags"] = tags

        run = wandb.init(**init_kwargs)
        if run is None:
            raise RuntimeError("wandb.init() returned None; failed to initialize wandb run.")
        return cls(wandb_run=run, log_type="wandb")

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log({tag: value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str) -> None:
        if self.wandb_run is not None:
            self.wandb_run.summary[tag] = text
        if self.writer is not None:
            self.writer.add_text(tag, text)

    def add_summary(self, tag: str, value: Any) -> None:
        """Record run-level metadata without creating sparse scalar curves."""
        if self.wandb_run is not None:
            self.wandb_run.summary[tag] = value
        elif self.writer is not None:
            self.writer.add_text(tag, str(value))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
