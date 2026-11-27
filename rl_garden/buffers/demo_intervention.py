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

from pathlib import Path
from typing import Any

from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin
from rl_garden.common.checkpoint import SidecarBuffer, load_replay_buffer_file


class DemoInterventionMixin(PriorDataReplayMixin):
    """Adds an empty, incrementally-growing demo buffer in the same slot
    :meth:`PriorDataReplayMixin.load_offline_replay_buffer` uses for a static
    dataset. Do not call both on the same instance -- they share
    ``offline_replay_buffer``/``offline_data_ratio``.

    Unlike a static dataset, the demo buffer cannot be loaded again, so it is
    a sidecar buffer (``_sidecar_buffers``): saved next to the replay buffer
    (``demo_buffer_<suffix>.pt``) and named at ``state["extra"]["demo_buffer_path"]``.
    A checkpoint without that slot reads the file named like its replay file.
    Its running episode ends at a restore, as the replay buffer's does when
    ``learn()`` resets the env, because the rows added next are another one's."""

    _has_demo_buffer: bool = False

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
        # save() names the file here when it writes one.
        return {**super()._extra_checkpoint_state(), "demo_buffer_path": None}

    def _sidecar_buffers(self) -> tuple[SidecarBuffer, ...]:
        sidecars = super()._sidecar_buffers()
        if not self._has_demo_buffer:
            return sidecars
        demo_buffer = SidecarBuffer(
            prefix="demo_buffer",
            reference_key=("demo_buffer_path",),
            buffer=self.offline_replay_buffer,
            restore=self._load_demo_buffer_file,
            unrecorded_beside_replay=True,
        )
        return (*sidecars, demo_buffer)

    def _load_demo_buffer_file(self, path: Path, strict: bool) -> None:
        load_replay_buffer_file(path, self.offline_replay_buffer, strict=strict)
        self.offline_replay_buffer.cut_running_episodes()

    def add_demo_transition(self, obs, next_obs, action, reward, done, **extra) -> None:
        self.offline_replay_buffer.add(obs, next_obs, action, reward, done, **extra)
