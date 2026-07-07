"""Generic offline-to-online (off2on) transition machinery.

``Off2OnReplayMixin`` owns every offline->online transition mechanic that is
generic across off2on algorithm families (Cal-QL, IQL, ...): offline/online
replay bookkeeping (``switch_to_online_mode``, mixed-batch sampling with a
fixed or adaptive ``offline_data_ratio``), checkpoint state for the
transition, offline-probe diagnostics scaffolding, and phase logging.

Algorithm-family-specific "what changes at the online switch" behavior is
expressed via two hooks, both no-ops by default:

- ``_apply_online_regularizer_override(online_replay_mode)``: mutate any
  algorithm-specific regularizer state at the switch (e.g. CQL/Cal-QL's
  ``use_cql_loss``/``cql_alpha`` swap). IQL needs no such override (confirmed
  against the reference WSRL/Cal-QL JAX implementation: IQL is treated
  identically to plain SAC at the offline->online switch).
- ``_offline_probe_metrics()``: compute diagnostic metrics from a cached
  offline batch (e.g. CQL/Cal-QL's predicted-Q/target-Q/TD-RMSE probe, which
  needs ``_critic_forward``/``_target_q`` — methods IQL does not have).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.combined import ImageEncoderFactory, default_image_encoder_factory


class Off2OnReplayMixin:
    """Shared off2on replay/transition machinery. See module docstring."""

    def _init_off2on_params(
        self,
        *,
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
    ) -> None:
        self.offline_sampling: Literal["with_replace", "without_replace"] = offline_sampling
        self._online_start_step: int | None = None
        self._offline_probe_batch: Any = None
        self.offline_replay_buffer: Optional[Any] = None
        self.offline_data_ratio: float | Literal["auto"] = 0.0

    def _configure_observation_kwargs(
        self,
        env: Any,
        *,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        detach_encoder_on_actor: bool = True,
    ) -> None:
        """Validate Box/Dict obs kwargs and set ``self._is_dict_obs``/image attrs.

        Must be called before ``super().__init__()``: the off-policy rollout
        shell's own ``__init__`` calls ``_setup_model()`` internally, which
        reads these attributes, so the concrete subclass has no later point
        to set them.
        """
        obs_space = env.single_observation_space
        image_kwargs_explicit = {
            "image_encoder_factory": image_encoder_factory,
            "image_keys": image_keys,
            "state_key": state_key,
            "use_proprio": use_proprio,
            "proprio_latent_dim": proprio_latent_dim,
            "image_fusion_mode": image_fusion_mode,
            "enable_stacking": enable_stacking,
        }
        explicitly_set = [k for k, v in image_kwargs_explicit.items() if v is not None]
        class_name = type(self).__name__

        if isinstance(obs_space, spaces.Box):
            if explicitly_set:
                raise ValueError(
                    f"{class_name} with Box observation space does not accept "
                    f"image-related kwargs (got {explicitly_set}). Use a Dict "
                    f"observation space, or remove these kwargs."
                )
            self._is_dict_obs = False
        elif isinstance(obs_space, spaces.Dict):
            if not detach_encoder_on_actor:
                raise ValueError(
                    f"{class_name} always uses stop_gradient=True on the actor "
                    "image path for Dict observations so image encoders are "
                    "trained only by critic loss."
                )
            self._is_dict_obs = True
            self._image_encoder_factory = (
                image_encoder_factory or default_image_encoder_factory()
            )
            self._image_keys = image_keys if image_keys is not None else ("rgb", "depth")
            self._state_key = state_key if state_key is not None else "state"
            self._use_proprio = use_proprio if use_proprio is not None else True
            self._proprio_latent_dim = (
                proprio_latent_dim if proprio_latent_dim is not None else 64
            )
            self._image_fusion_mode = (
                image_fusion_mode if image_fusion_mode is not None else "stack_channels"
            )
            self._enable_stacking = enable_stacking if enable_stacking is not None else False
        else:
            raise TypeError(
                f"{class_name} supports Box or Dict observation spaces, got {type(obs_space)}"
            )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
            "offline_sampling": self.offline_sampling,
        }
        if self._is_dict_obs:
            meta.update(
                {
                    "image_keys": self._image_keys,
                    "state_key": self._state_key,
                    "use_proprio": self._use_proprio,
                    "proprio_latent_dim": self._proprio_latent_dim,
                    "image_fusion_mode": self._image_fusion_mode,
                    "enable_stacking": self._enable_stacking,
                }
            )
        return meta

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state = super()._extra_checkpoint_state()
        state["online_start_step"] = self._online_start_step
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        start = state.get("online_start_step")
        self._online_start_step = None if start is None else int(start)

    def _should_start_initial_training_phase_on_learn(self) -> bool:
        # Off2on algorithms start any configured initial phase at the
        # offline-to-online mode switch, not at learn() start. This is a
        # no-op for subclasses that never configure ``initial_training_phase``.
        return False

    def _actor_stop_gradient(self) -> bool:
        return self._is_dict_obs

    def _clear_replay_buffer(self) -> int:
        previous_len = len(self.replay_buffer)
        self.replay_buffer.pos = 0
        self.replay_buffer.full = False
        if hasattr(self.replay_buffer, "_mc_table"):
            self.replay_buffer._mc_table = None
        return previous_len

    def _sample_batch(self, batch_size: int):
        in_offline_phase = self._online_start_step is None
        if in_offline_phase and self.offline_sampling == "without_replace":
            return self.replay_buffer.sample_without_repeat(batch_size)
        if (
            not in_offline_phase
            and self.offline_replay_buffer is not None
            and (self.offline_data_ratio == "auto" or self.offline_data_ratio > 0.0)
        ):
            return self._sample_mixed_batch(batch_size)
        return self.replay_buffer.sample(batch_size)

    def _sample_train_batch(self, batch_size: int):
        return self._sample_batch(batch_size)

    def _resolve_offline_data_ratio(self) -> float:
        """Resolve ``offline_data_ratio`` to a numeric fraction.

        ``"auto"`` reproduces the official Cal-QL formula
        (``offline_size / (offline_size + online_size)``), so the offline
        share decays automatically as online data accumulates, instead of a
        fixed constant.
        """
        if self.offline_data_ratio == "auto":
            offline_n = len(self.offline_replay_buffer)
            online_n = len(self.replay_buffer)
            total = offline_n + online_n
            return offline_n / total if total > 0 else 1.0
        return self.offline_data_ratio

    def _sample_mixed_batch(self, batch_size: int):
        ratio = self._resolve_offline_data_ratio()
        n_online = batch_size - int(round(batch_size * ratio))
        online_size = len(self.replay_buffer)
        if online_size == 0:
            return self.offline_replay_buffer.sample(batch_size)
        n_offline = batch_size - n_online
        if n_online == 0:
            return self.offline_replay_buffer.sample(batch_size)
        if n_offline == 0:
            return self.replay_buffer.sample(batch_size)

        online_sample = self.replay_buffer.sample(n_online)
        offline_sample = self.offline_replay_buffer.sample(n_offline)
        return self._concat_replay_samples(online_sample, offline_sample)

    @staticmethod
    def _concat_replay_samples(a, b):
        """Concatenate two same-type replay samples along the batch dim.

        Field-generic (via ``dataclasses.fields``) so it works for both the
        plain ``ReplayBufferSample`` (used by IQL) and its MC-return
        superset ``MCReplayBufferSample`` (used by Cal-QL), without hardcoding
        either field list.
        """

        def _cat(x, y):
            if isinstance(x, dict):
                return {k: torch.cat([x[k], y[k]], dim=0) for k in x}
            if x is None:
                return None
            return torch.cat([x, y], dim=0)

        kwargs = {
            f.name: _cat(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
        }
        return type(a)(**kwargs)

    @staticmethod
    def canonical_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
        out = dict(metrics)
        success = metrics.get("success_at_end", metrics.get("success_once"))
        if success is not None:
            out["normalized_score"] = float(success) * 100.0
        return out

    def set_offline_probe_batch(self, batch: Any) -> None:
        self._offline_probe_batch = batch

    def _offline_probe_metrics(self) -> dict[str, float]:
        """Diagnostic metrics computed from the cached offline probe batch.

        No-op by default; algorithm families whose critic interface supports
        it (Cal-QL: ``_critic_forward``/``_target_q``) override this.
        """
        return {}

    def _apply_online_regularizer_override(self, online_replay_mode: str) -> None:
        """Mutate algorithm-specific regularizer state at the online switch.

        No-op by default (correct for IQL, which needs no change at the
        offline->online switch). Cal-QL family overrides this to swap
        ``use_cql_loss``/``cql_alpha`` to their online values.
        """

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in self.canonical_eval_metrics(metrics).items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _log_rollout_metric(self, key: str, value: float, step: int) -> None:
        if self.logger is None:
            return
        self.logger.add_scalar(f"train/{key}", value, step)
        if key in {"success_at_end", "success_once"}:
            self.logger.add_scalar("train/normalized_score", value * 100.0, step)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        self.logger.log_metrics(metrics, step)

        # Not propagated into Logger.log_metrics so SAC online and offline
        # runs of algorithms without a td_loss metric do not start showing
        # q/td_rmse.
        td_loss = metrics.get("td_loss")
        if isinstance(td_loss, (int, float)):
            self.logger.add_scalar(
                "q/td_rmse", float(np.sqrt(max(td_loss, 0.0))), step
            )

        online_start = self._online_start_step
        is_online = online_start is not None and step >= online_start
        offline_step = min(step, online_start) if online_start is not None else step
        online_step = max(0, step - online_start) if online_start is not None else 0
        is_warmup = self._active_initial_training_phase() is not None
        self.logger.add_scalar("phase/is_online", float(is_online), step)
        self.logger.add_scalar("phase/warmup", float(is_warmup), step)
        self.logger.add_scalar("phase/offline_step", float(offline_step), step)
        self.logger.add_scalar("phase/online_step", float(online_step), step)

        if is_online:
            for tag, value in self._offline_probe_metrics().items():
                self.logger.add_scalar(tag, value, step)
            if self.offline_data_ratio == "auto" and self.offline_replay_buffer is not None:
                self.logger.add_scalar(
                    "phase/offline_mix_ratio", self._resolve_offline_data_ratio(), step
                )

    def switch_to_online_mode(
        self,
        online_replay_mode: Literal["empty", "append", "mixed"] = "append",
        offline_data_ratio: float | Literal["auto"] = 0.0,
    ) -> None:
        if self._online_start_step is not None:
            return
        if isinstance(offline_data_ratio, str):
            if offline_data_ratio != "auto":
                raise ValueError(
                    f"offline_data_ratio must be a float in [0, 1] or 'auto'; "
                    f"got {offline_data_ratio!r}."
                )
        elif not (0.0 <= offline_data_ratio <= 1.0):
            raise ValueError(f"offline_data_ratio must be in [0, 1]; got {offline_data_ratio}.")
        if online_replay_mode not in {"empty", "append", "mixed"}:
            raise ValueError(f"Unknown online_replay_mode: {online_replay_mode!r}")

        self._online_start_step = self._global_step
        self._start_initial_training_phase(self._global_step)
        self.offline_replay_buffer = None
        self.offline_data_ratio = 0.0
        cleared_transitions = 0
        if online_replay_mode == "empty":
            cleared_transitions = self._clear_replay_buffer()
        elif online_replay_mode == "append":
            pass
        elif online_replay_mode == "mixed":
            self.offline_replay_buffer = self.replay_buffer
            self.replay_buffer = self._build_replay_buffer()
            self.offline_data_ratio = offline_data_ratio

        self._apply_online_regularizer_override(online_replay_mode)

        if self.logger:
            self.logger.add_summary("off2on/online_start_step", self._global_step)
            warmup_steps = (
                self.initial_training_phase.duration_steps
                if self.initial_training_phase is not None
                else 0
            )
            self.logger.add_summary("off2on/warmup_steps", warmup_steps)
            if self._initial_phase_start_step is not None and warmup_steps > 0:
                self.logger.add_summary(
                    "off2on/warmup_end_step",
                    self._initial_phase_start_step + warmup_steps,
                )
            self.logger.add_summary("off2on/online_replay_mode", online_replay_mode)
            self.logger.add_summary(
                "off2on/online_replay_cleared", online_replay_mode == "empty"
            )
            if online_replay_mode == "empty":
                self.logger.add_summary(
                    "off2on/online_replay_size_before_clear", cleared_transitions
                )
            if online_replay_mode == "mixed":
                self.logger.add_summary("off2on/offline_data_ratio", offline_data_ratio)
