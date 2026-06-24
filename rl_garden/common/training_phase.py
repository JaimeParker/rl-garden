"""Initial off-policy training phase configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingUpdateMask:
    """Optimizer/gradient switches for the current training phase."""

    update_actor: bool = True
    update_critic: bool = True
    update_encoder: bool = True


STANDARD_UPDATE_MASK = TrainingUpdateMask()


@dataclass(frozen=True)
class InitialTrainingPhase:
    """One bounded phase before standard joint actor-critic training."""

    duration_steps: int
    update_actor: bool
    update_critic: bool
    update_encoder: bool
    random_action_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_steps < 0:
            raise ValueError(
                f"duration_steps must be non-negative, got {self.duration_steps}"
            )
        if not 0.0 <= self.random_action_prob <= 1.0:
            raise ValueError(
                "random_action_prob must be in [0, 1], got "
                f"{self.random_action_prob}"
            )
        if self.update_encoder and not self.update_critic:
            raise ValueError(
                "update_encoder=True requires update_critic=True because the "
                "shared encoder has no independent phase objective"
            )

    @property
    def update_mask(self) -> TrainingUpdateMask:
        return TrainingUpdateMask(
            update_actor=self.update_actor,
            update_critic=self.update_critic,
            update_encoder=self.update_encoder,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
