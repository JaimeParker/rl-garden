"""ResidualSAC variant with HIL-SERL-style growing intervention demo buffer."""
from __future__ import annotations

from rl_garden.algorithms.residual import ResidualSAC


class ResidualHilSerlSAC(ResidualSAC):
    """ResidualSAC plus an incrementally populated demo buffer.

    The demo buffer reuses ResidualSAC's existing offline replay slot and
    sampling logic. It stores residual replay samples, so callers must provide
    ``base_actions`` and ``next_base_actions`` for every demo transition.
    """

    _compatible_checkpoint_algorithms = ("ResidualHilSerlSAC", "ResidualSAC")

    def init_demo_buffer(self, buffer_size: int, demo_data_ratio: float = 0.5) -> None:
        if not (0.0 <= demo_data_ratio <= 1.0):
            raise ValueError(f"demo_data_ratio must be in [0, 1], got {demo_data_ratio}.")
        if self.offline_replay_buffer is not None:
            raise RuntimeError(
                "offline_replay_buffer is already populated (e.g. via "
                "load_offline_replay_buffer/--offline_dataset_path) -- "
                "init_demo_buffer() would silently discard it. Residual offline "
                "datasets and HIL-SERL's demo buffer share this one slot and "
                "can't both be used on the same instance."
            )
        self.offline_replay_buffer = self._make_residual_replay_buffer(int(buffer_size))
        self.offline_data_ratio = float(demo_data_ratio)

    def add_demo_transition(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        *,
        base_actions,
        next_base_actions,
    ) -> None:
        if self.offline_replay_buffer is None:
            raise RuntimeError("init_demo_buffer() must be called before add_demo_transition().")
        self.offline_replay_buffer.add(
            obs,
            next_obs,
            action,
            reward,
            done,
            base_actions=base_actions,
            next_base_actions=next_base_actions,
        )
