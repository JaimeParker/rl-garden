"""Growing human-intervention "demo" buffer, layered on top of
:class:`~rl_garden.buffers.prior_data_replay.PriorDataReplayMixin`.

HIL-SERL's ``train_rlpd.py`` keeps a demo buffer that grows with every
human-intervened transition and is mixed into training at a fixed ratio
alongside the online replay buffer -- structurally identical to
``PriorDataReplayMixin``'s existing "static offline prior data, mixed from
step 0" machinery (``_sample_train_batch``/``_concat_replay_samples``/
``_shuffle_batch``), just populated incrementally instead of loaded once from
a file. Reuses that machinery unchanged via the same
``offline_replay_buffer``/``offline_data_ratio`` slot rather than adding a
third buffer -- a real-world run with a growing demo buffer doesn't also load
a static ``offline_dataset`` prior dataset in this round, so the two
never need to coexist independently.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin
from rl_garden.common.checkpoint import (
    load_checkpoint_file,
    load_replay_buffer_file,
    replay_buffer_path_for_checkpoint,
    save_replay_buffer_file,
)


def _demo_buffer_path_beside(replay_path: Path) -> Path:
    return replay_path.with_name(replay_path.name.replace("replay_buffer_", "demo_buffer_", 1))


def demo_buffer_path_for_checkpoint(path: str | Path) -> Path:
    return _demo_buffer_path_beside(replay_buffer_path_for_checkpoint(path))


class DemoInterventionMixin(PriorDataReplayMixin):
    """Adds an empty, incrementally-growing demo buffer in the same slot
    :meth:`PriorDataReplayMixin.load_offline_replay_buffer` uses for a static
    dataset. Do not call both on the same instance -- they share
    ``offline_replay_buffer``/``offline_data_ratio``.

    Unlike a static dataset, the demo buffer cannot be loaded again, so it is
    saved next to the replay buffer (``demo_buffer_<suffix>.pt``) and referenced
    from the checkpoint."""

    _has_demo_buffer: bool = False
    # Names the file in a checkpoint being written with the demo buffer.
    _demo_buffer_file: Optional[str] = None

    def init_demo_buffer(self, buffer_size: int, demo_data_ratio: float = 0.5) -> None:
        if not (0.0 <= demo_data_ratio <= 1.0):
            raise ValueError(f"demo_data_ratio must be in [0, 1], got {demo_data_ratio}.")
        if self.offline_replay_buffer is not None:
            raise RuntimeError(
                "offline_replay_buffer is already populated (e.g. via "
                "load_offline_replay_buffer/--offline_dataset) -- "
                "init_demo_buffer() would silently discard it. RLPD's static "
                "offline dataset and HIL-SERL's demo buffer share this one "
                "slot and can't both be used on the same instance."
            )
        self.offline_replay_buffer = self._build_prior_data_buffer(int(buffer_size))
        self.offline_data_ratio = float(demo_data_ratio)
        self._has_demo_buffer = True

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "demo_buffer_path": self._demo_buffer_file,
        }

    def save(self, path: str | Path, include_replay_buffer: bool = False) -> Path:
        demo_path = demo_buffer_path_for_checkpoint(path)
        if include_replay_buffer and self._has_demo_buffer:
            self._demo_buffer_file = demo_path.name
        try:
            checkpoint_path = super().save(path, include_replay_buffer=include_replay_buffer)
            if self._demo_buffer_file is not None:
                save_replay_buffer_file(demo_path, self.offline_replay_buffer)
        finally:
            self._demo_buffer_file = None
        return checkpoint_path

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_replay_buffer: bool = True,
        load_optimizers: bool = True,
    ):
        super().load(
            path,
            strict=strict,
            load_replay_buffer=load_replay_buffer,
            load_optimizers=load_optimizers,
        )
        if load_replay_buffer and self._has_demo_buffer:
            self._load_demo_buffer_file(Path(path), strict=strict)
        return self

    def _load_demo_buffer_file(self, path: Path, *, strict: bool) -> None:
        checkpoint = load_checkpoint_file(path, map_location="cpu")
        replay_ref = checkpoint["metadata"].get("replay_buffer_path")
        replay_path = (
            replay_buffer_path_for_checkpoint(path)
            if replay_ref is None
            else path.parent / replay_ref
        )
        extra = checkpoint["state"].get("extra", {})
        demo_ref = extra.get("demo_buffer_path")
        demo_path = (
            _demo_buffer_path_beside(replay_path) if demo_ref is None else path.parent / demo_ref
        )
        if demo_path.exists():
            load_replay_buffer_file(demo_path, self.offline_replay_buffer, strict=strict)
        elif demo_ref is not None:
            raise FileNotFoundError(
                f"Checkpoint references demo buffer {demo_path}, but it does not exist."
            )
        elif "demo_buffer_path" not in extra and replay_path.exists():
            warnings.warn(
                f"{path} does not record whether it was saved with a demo buffer and "
                f"{demo_path} does not exist: continuing with an empty demo buffer.",
                stacklevel=3,
            )

    def add_demo_transition(self, obs, next_obs, action, reward, done, **extra) -> None:
        self.offline_replay_buffer.add(obs, next_obs, action, reward, done, **extra)
