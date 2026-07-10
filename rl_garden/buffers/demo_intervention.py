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
a static ``offline_dataset_path`` prior dataset in this round, so the two
never need to coexist independently.
"""
from __future__ import annotations

from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin


class DemoInterventionMixin(PriorDataReplayMixin):
    """Adds an empty, incrementally-growing demo buffer in the same slot
    :meth:`PriorDataReplayMixin.load_offline_replay_buffer` uses for a static
    dataset. Do not call both on the same instance -- they share
    ``offline_replay_buffer``/``offline_data_ratio``."""

    def init_demo_buffer(self, buffer_size: int, demo_data_ratio: float = 0.5) -> None:
        if not (0.0 <= demo_data_ratio <= 1.0):
            raise ValueError(f"demo_data_ratio must be in [0, 1], got {demo_data_ratio}.")
        self.offline_replay_buffer = self._build_prior_data_buffer(int(buffer_size))
        self.offline_data_ratio = float(demo_data_ratio)

    def add_demo_transition(self, obs, next_obs, action, reward, done, **extra) -> None:
        self.offline_replay_buffer.add(obs, next_obs, action, reward, done, **extra)
