"""SPOT offline-to-online, built on the shared shell.

Unlike ``Off2OnIQL``/``Off2OnAWAC`` (no algorithm-specific change at the
offline->online switch), ``Off2OnSPOT`` inherits
``_SPOTRolloutTrainingShell``'s override of ``_apply_online_regularizer_override``
(optimizer reset + discount swap, reproducing ``spot.py``'s "Resetting
optimizers" block) and ``_rollout_action`` (TD3-style exploration noise) --
both are shell-level behavior, so this class adds only a construction-time
preset, matching ``Off2OnIQL``'s shape.
"""
from __future__ import annotations

import warnings
from typing import Any

from rl_garden.algorithms.spot import _SPOTRolloutTrainingShell


class Off2OnSPOT(_SPOTRolloutTrainingShell):
    """SPOT offline pretraining + online fine-tuning."""

    _compatible_checkpoint_algorithms = ("Off2OnSPOT", "SPOT")

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state = super()._extra_checkpoint_state()
        state["spot_online_update_start"] = self._spot_online_update_start
        state["spot_max_online_updates"] = self.max_online_updates
        state["spot_gamma"] = self.gamma
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        self._spot_online_update_start = state.get("spot_online_update_start")
        if self._online_start_step is None:
            return
        self.gamma = state.get("spot_gamma", self.online_discount)
        self.max_online_updates = state.get("spot_max_online_updates", self.max_online_updates)
        if self.lambd_cool and self._spot_online_update_start is None:
            warnings.warn(
                "This online-stage checkpoint does not record where lambd cooling started; "
                "lambd stays at its initial value.",
                UserWarning,
                stacklevel=2,
            )

    def pretrain_vae(self) -> dict[str, float]:
        if self.offline_replay_buffer is None:
            return super().pretrain_vae()
        online_buffer = self.replay_buffer
        self.replay_buffer = self.offline_replay_buffer
        try:
            return super().pretrain_vae()
        finally:
            self.replay_buffer = online_buffer
