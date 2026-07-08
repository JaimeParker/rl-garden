"""Static prior-data + growing online-buffer mixing, shared by any algorithm
that samples every training batch as a ratio mix of a fixed offline dataset
and an online replay buffer, from the first gradient step (no offline-only
pretraining phase, no explicit online-switch event).

Written fresh rather than imported from ``ResidualSAC`` (residual-specific
buffer/action fields) or ``Off2OnReplayMixin`` (built around an offline
pretrain phase + explicit ``switch_to_online_mode()``, which doesn't apply
here). Both of those already implement closely related mixing logic; this
mixin is the generic version, intended for any future algorithm that needs
the same "static prior data mixed at a fixed ratio from step 0" shape.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.maniskill_h5 import load_maniskill_h5_to_replay_buffer
from rl_garden.buffers.minari_dataset import load_minari_dataset_to_replay_buffer
from rl_garden.buffers.nstep_buffer import NStepDictReplayBuffer
from rl_garden.buffers.nstep_tensor_buffer import NStepTensorReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer

DatasetSource = Literal["maniskill_h5", "minari"]


class PriorDataReplayMixin:
    """Shared prior-data replay/mixing machinery. See module docstring."""

    def _init_prior_data_params(self) -> None:
        self.offline_replay_buffer: Optional[Any] = None
        self.offline_data_ratio: float = 0.0

    def _build_prior_data_buffer(self, buffer_size: int):
        """Same buffer type as ``self.replay_buffer`` (Tensor/Dict, N-step or
        not -- mirrors ``SAC._build_replay_buffer``'s branching exactly so
        ``.sample()`` returns the same dataclass shape on both buffers;
        ``_concat_replay_samples`` requires matching fields, e.g. ``nstep>1``
        adds a ``discounts`` field), sized independently for a static,
        single-environment offline dataset."""
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            if self.nstep > 1:
                return NStepDictReplayBuffer(
                    observation_space=obs_space,
                    action_space=self.env.single_action_space,
                    num_envs=1,
                    buffer_size=buffer_size,
                    nstep=self.nstep,
                    gamma=self.gamma,
                    storage_device=self.buffer_device,
                    sample_device=self.device,
                )
            return DictReplayBuffer(
                observation_space=obs_space,
                action_space=self.env.single_action_space,
                num_envs=1,
                buffer_size=buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        if self.nstep > 1:
            return NStepTensorReplayBuffer(
                observation_space=obs_space,
                action_space=self.env.single_action_space,
                num_envs=1,
                buffer_size=buffer_size,
                nstep=self.nstep,
                gamma=self.gamma,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        return TensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def load_offline_replay_buffer(
        self,
        path: str | Path,
        *,
        buffer_size: int,
        source: DatasetSource = "maniskill_h5",
        num_traj: Optional[int] = None,
        offline_data_ratio: float = 0.5,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        success_key: Optional[str] = None,
    ) -> int:
        if not (0.0 <= offline_data_ratio <= 1.0):
            raise ValueError(
                f"offline_data_ratio must be in [0, 1], got {offline_data_ratio}."
            )
        self.offline_replay_buffer = self._build_prior_data_buffer(int(buffer_size))
        if source == "maniskill_h5":
            loaded = load_maniskill_h5_to_replay_buffer(
                self.offline_replay_buffer,
                path,
                num_traj=num_traj,
                reward_scale=reward_scale,
                reward_bias=reward_bias,
                success_key=success_key,
            )
        elif source == "minari":
            loaded = load_minari_dataset_to_replay_buffer(
                self.offline_replay_buffer,
                str(path),
                num_episodes=num_traj,
                reward_scale=reward_scale,
                reward_bias=reward_bias,
                success_key=success_key,
            )
        else:
            raise ValueError(f"Unknown dataset source: {source!r}")

        self.offline_data_ratio = float(offline_data_ratio)
        if self.logger is not None:
            self.logger.add_summary("prior_data/offline_loaded_transitions", loaded)
            self.logger.add_summary(
                "prior_data/offline_data_ratio", self.offline_data_ratio
            )
            self.logger.add_summary("prior_data/offline_buffer_size", int(buffer_size))
        return loaded

    def _sample_train_batch(self, batch_size: int):
        if self.offline_replay_buffer is None or self.offline_data_ratio <= 0.0:
            return self.replay_buffer.sample(batch_size)
        if len(self.offline_replay_buffer) == 0:
            return self.replay_buffer.sample(batch_size)
        if len(self.replay_buffer) == 0:
            return self.offline_replay_buffer.sample(batch_size)

        n_offline = int(round(batch_size * self.offline_data_ratio))
        n_offline = min(max(n_offline, 0), batch_size)
        n_online = batch_size - n_offline
        if n_offline == 0:
            return self.replay_buffer.sample(batch_size)
        if n_online == 0:
            return self.offline_replay_buffer.sample(batch_size)

        online_sample = self.replay_buffer.sample(n_online)
        offline_sample = self.offline_replay_buffer.sample(n_offline)
        combined = self._concat_replay_samples(online_sample, offline_sample)
        return self._shuffle_batch(combined, batch_size)

    def _shuffle_batch(self, sample, batch_size: int):
        # train_high_utd (sac_core.py) samples one batch and slices it
        # sequentially into per-gradient-step minibatches, without shuffling.
        # Without this shuffle, a block-concatenated [online; offline] batch
        # would produce minibatches that are each nearly-pure online or
        # nearly-pure offline under high UTD, breaking RLPD's "every gradient
        # step sees a representative mix" invariant.
        perm = torch.randperm(batch_size, device=sample.rewards.device)

        def _idx(x):
            if isinstance(x, dict):
                return {k: v[perm] for k, v in x.items()}
            if x is None:
                return None
            return x[perm]

        kwargs = {f.name: _idx(getattr(sample, f.name)) for f in dataclasses.fields(sample)}
        return type(sample)(**kwargs)

    @staticmethod
    def _concat_replay_samples(a, b):
        # TODO: duplicated from Off2OnReplayMixin._concat_replay_samples
        # (off2on.py) / ResidualSAC._concat_replay_samples (residual.py) --
        # consider extracting a single shared helper once a third caller
        # shows up.
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

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {**super()._checkpoint_metadata(), "offline_data_ratio": self.offline_data_ratio}
        return meta
