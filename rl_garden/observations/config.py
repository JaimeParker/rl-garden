"""Declarative "what is observed" configuration.

``ObservationConfig`` is the CLI/YAML-facing description of which observation
modalities an environment or dataset must produce. It says nothing about how
those modalities are encoded (see ``rl_garden/encoders``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ObservationConfig:
    """What an environment/dataset must observe.

    ``rgb`` and ``depth`` are camera names; they are rendered into keys
    ``rgb_<cam>`` / ``depth_<cam>``. ``image_size`` is ``(H, W)``; ``None``
    lets the backend pick its own default. ``frame_stack`` stacks image
    frames into a leading time dimension at the env (distinct from
    ``cond_steps`` used by chunked-BC algorithms for observation history).
    """

    state: bool = True
    rgb: tuple[str, ...] = ()
    depth: tuple[str, ...] = ()
    image_size: Optional[tuple[int, int]] = None
    frame_stack: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "rgb", tuple(self.rgb))
        object.__setattr__(self, "depth", tuple(self.depth))
        if self.image_size is not None:
            object.__setattr__(self, "image_size", tuple(self.image_size))

        if self.frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {self.frame_stack}")
        if not (self.state or self.rgb or self.depth):
            raise ValueError("at least one of state, rgb, or depth must be set")
        if self.image_size is not None:
            if len(self.image_size) != 2:
                raise ValueError(f"image_size must be (H, W), got {self.image_size}")
            h, w = self.image_size
            if h <= 0 or w <= 0:
                raise ValueError(f"image_size must be positive, got {self.image_size}")
        for camera in (*self.rgb, *self.depth):
            if not camera or "/" in camera:
                raise ValueError(f"invalid camera name {camera!r}")

    @property
    def is_visual(self) -> bool:
        return bool(self.rgb) or bool(self.depth)

    @property
    def image_keys(self) -> tuple[str, ...]:
        """``rgb_<cam>`` keys first, then ``depth_<cam>`` keys."""
        return tuple(f"rgb_{cam}" for cam in self.rgb) + tuple(
            f"depth_{cam}" for cam in self.depth
        )

    @property
    def expected_keys(self) -> tuple[str, ...]:
        """Image keys, then ``"state"`` if requested."""
        keys = self.image_keys
        if self.state:
            keys = keys + ("state",)
        return keys
