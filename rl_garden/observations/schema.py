"""Observation contract: strict key vocabulary, schema derivation, validation.

Every observation space in rl-garden is (or is normalized to) a
``spaces.Dict`` using exactly the keys ``"state"``, ``"state_<name>"``,
``"rgb_<cam>"``, and ``"depth_<cam>"``. This module is the single place that
vocabulary and the per-modality shape/dtype rules are enforced.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
from gymnasium import spaces


class ObservationContractError(ValueError):
    """Raised when an observation space or key violates the rl-garden contract."""


class Modality(str, Enum):
    STATE = "state"
    RGB = "rgb"
    DEPTH = "depth"


def key_modality(key: str) -> Modality:
    """Classify an observation key. ``"state"``, ``"state_<name>"``,
    ``"rgb_<cam>"``, and ``"depth_<cam>"`` are valid; anything else (including
    bare rgb/depth) raises.
    """
    if key == "state":
        return Modality.STATE
    if key.startswith("state_"):
        name = key[len("state_"):]
        if not name or "/" in name:
            raise ObservationContractError(
                f"invalid state key {key!r}; expected 'state_<name>' with a "
                "non-empty <name> containing no '/'"
            )
        return Modality.STATE
    if key.startswith("rgb_"):
        return Modality.RGB
    if key.startswith("depth_"):
        return Modality.DEPTH
    raise ObservationContractError(
        f"unknown observation key {key!r}; expected 'state', 'state_<name>', "
        "'rgb_<cam>', or 'depth_<cam>'"
    )


@dataclass(frozen=True)
class ObsEntry:
    key: str
    modality: Modality
    shape: tuple[int, ...]
    dtype: np.dtype
    stacked: bool  # True => leading dim is a frame_stack time dimension


def _is_stacked_image(shape: tuple[int, ...], frame_stack: int) -> bool:
    return len(shape) == 4 or frame_stack > 1


@dataclass(frozen=True)
class ObservationSchema:
    """``shape_meta`` equivalent, derived from a validated observation space."""

    entries: Mapping[str, ObsEntry]

    @classmethod
    def from_space(
        cls, space: "spaces.Box | spaces.Dict", *, frame_stack: int = 1
    ) -> "ObservationSchema":
        if isinstance(space, spaces.Box):
            entry = ObsEntry(
                key="state",
                modality=Modality.STATE,
                shape=tuple(space.shape),
                dtype=space.dtype,
                stacked=False,
            )
            return cls(entries={"state": entry})

        if isinstance(space, spaces.Dict):
            entries: dict[str, ObsEntry] = {}
            for key, sp in space.spaces.items():
                modality = key_modality(key)
                shape = tuple(sp.shape)
                stacked = (
                    modality != Modality.STATE
                    and _is_stacked_image(shape, frame_stack)
                )
                entries[key] = ObsEntry(
                    key=key,
                    modality=modality,
                    shape=shape,
                    dtype=sp.dtype,
                    stacked=stacked,
                )
            return cls(entries=entries)

        raise ObservationContractError(
            f"observation space must be Box or Dict, got {type(space).__name__}"
        )

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.entries.keys())

    @property
    def image_keys(self) -> tuple[str, ...]:
        """``rgb_<cam>`` keys then ``depth_<cam>`` keys, in space order."""
        rgb = [k for k, e in self.entries.items() if e.modality == Modality.RGB]
        depth = [k for k, e in self.entries.items() if e.modality == Modality.DEPTH]
        return tuple(rgb + depth)

    @property
    def state_keys(self) -> tuple[str, ...]:
        """``"state"`` first if present, then other ``state_<name>`` keys in
        space order."""
        others = [
            k for k, e in self.entries.items() if e.modality == Modality.STATE and k != "state"
        ]
        state = ["state"] if "state" in self.entries else []
        return tuple(state + others)

    @property
    def has_images(self) -> bool:
        return bool(self.image_keys)

    @property
    def has_state(self) -> bool:
        return bool(self.state_keys)

    def subset(self, keys: "tuple[str, ...] | list[str]") -> "ObservationSchema":
        unknown = [k for k in keys if k not in self.entries]
        if unknown:
            raise ObservationContractError(
                f"unknown observation key(s) {unknown!r}; schema has {self.keys!r}"
            )
        return ObservationSchema(entries={k: self.entries[k] for k in keys})

    def image_size(self, key: str) -> tuple[int, int]:
        if key not in self.entries:
            raise ObservationContractError(f"unknown observation key {key!r}")
        entry = self.entries[key]
        if entry.modality == Modality.STATE:
            raise ObservationContractError(f"{key!r} is not an image key")
        if len(entry.shape) == 4:
            return entry.shape[1], entry.shape[2]
        if len(entry.shape) == 3:
            return entry.shape[0], entry.shape[1]
        raise ObservationContractError(
            f"image key {key!r} has an unsupported shape {entry.shape!r}"
        )


def normalize_observation_space(space: "spaces.Box | spaces.Dict") -> spaces.Dict:
    """Coerce a bare ``Box`` (plain state env) into ``Dict({"state": Box})``.

    A ``Dict`` space is validated and returned unchanged.
    """
    validate_observation_space(space)
    if isinstance(space, spaces.Box):
        return spaces.Dict({"state": space})
    return space


def _validate_state_box(space: spaces.Box, *, key: str) -> None:
    if len(space.shape) != 1:
        raise ObservationContractError(
            f"{key!r} must be a 1-D Box, got shape {space.shape!r}"
        )
    # Any float dtype is accepted here (plain gymnasium envs are commonly
    # float64); every backend/dataset loader casts to float32 itself before
    # data reaches a buffer or encoder (see e.g. _dataset_common._to_tensor).
    if not np.issubdtype(space.dtype, np.floating):
        raise ObservationContractError(
            f"{key!r} must be a float dtype, got {space.dtype}"
        )


def _validate_rgb_box(space: spaces.Box, *, key: str) -> None:
    if space.dtype != np.uint8:
        raise ObservationContractError(f"{key!r} must be uint8, got {space.dtype}")
    if len(space.shape) not in (3, 4):
        raise ObservationContractError(
            f"{key!r} must be HWC or (T, H, W, C), got shape {space.shape!r}"
        )
    if space.shape[-1] != 3:
        raise ObservationContractError(
            f"{key!r} must have 3 channels (RGB), got shape {space.shape!r}"
        )


def _validate_depth_box(space: spaces.Box, *, key: str) -> None:
    if space.dtype != np.float32:
        raise ObservationContractError(f"{key!r} must be float32, got {space.dtype}")
    if len(space.shape) not in (3, 4):
        raise ObservationContractError(
            f"{key!r} must be HW1 or (T, H, W, 1), got shape {space.shape!r}"
        )
    if space.shape[-1] != 1:
        raise ObservationContractError(
            f"{key!r} must have 1 channel (depth), got shape {space.shape!r}"
        )


def validate_observation_space(space: "spaces.Box | spaces.Dict") -> None:
    """Raise ``ObservationContractError`` unless ``space`` matches the
    rl-garden observation contract."""
    if isinstance(space, spaces.Box):
        _validate_state_box(space, key="state")
        return

    if not isinstance(space, spaces.Dict):
        raise ObservationContractError(
            f"observation space must be Box or Dict, got {type(space).__name__}"
        )

    for key, sp in space.spaces.items():
        if isinstance(sp, spaces.Dict):
            raise ObservationContractError(
                f"nested Dict observation spaces are not supported (key {key!r})"
            )
        if not isinstance(sp, spaces.Box):
            raise ObservationContractError(
                f"observation key {key!r} must be a Box, got {type(sp).__name__}"
            )
        modality = key_modality(key)
        if modality == Modality.STATE:
            _validate_state_box(sp, key=key)
        elif modality == Modality.RGB:
            _validate_rgb_box(sp, key=key)
        else:
            _validate_depth_box(sp, key=key)
