"""ExPLORe (Li et al. 2023): RLPD + optimistic reward relabeling of the
unlabeled offline prior data, plus an optional RND novelty bonus.

RLPD (``rl_garden/algorithms/rlpd.py``) already provides the full backbone
(SAC + high UTD + REDQ-style critic ensemble subsampling + LayerNorm +
static offline/online prior-data mixing via ``PriorDataReplayMixin``) --
ExPLORe only adds a ``RewardMaskRelabeler`` (``rl_garden/networks/reward_mask_relabeler.py``)
that relabels the offline half of every training batch, and an optional
``RNDBonus`` (``rl_garden/networks/rnd.py``) added to either half's reward.
Both are generic, standalone components (not ExPLORe-specific), per the
project's "components + algorithms" plan for the ExPLORe/SUPE/IDQL/HILP
family -- SUPE will be their second consumer.

State-based only: ``RewardMaskRelabeler``/``RNDBonus`` take flat
``(obs, action)`` tensors, matching ExPLORe's own state-based reference
(the pixel variant additionally needs ICVF representation pretraining, out
of scope here).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.rlpd import RLPD
from rl_garden.common.optim import make_optimizer
from rl_garden.networks import Activation, KernelInit, RewardMaskRelabeler, RNDBonus

OfflineRelabelType = Literal["gt", "pred", "min"]


class ExPLORe(RLPD):
    """ExPLORe as a thin extension of RLPD. See module docstring."""

    _compatible_checkpoint_algorithms = ("ExPLORe",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        offline_relabel_type: OfflineRelabelType = "pred",
        use_rnd_offline: bool = False,
        use_rnd_online: bool = False,
        rnd_coeff: float = 1.0,
        relabeler_hidden_dims: Sequence[int] = (256, 256),
        relabeler_lr: float = 3e-4,
        relabeler_activation_fn: Optional[Activation] = None,
        relabeler_kernel_init: Optional[KernelInit] = None,
        rnd_hidden_dims: Sequence[int] = (256, 256),
        rnd_feature_dim: int = 256,
        rnd_lr: float = 3e-4,
        rnd_activation_fn: Optional[Activation] = None,
        rnd_kernel_init: Optional[KernelInit] = None,
        **rlpd_kwargs: Any,
    ) -> None:
        if isinstance(env.single_observation_space, spaces.Dict):
            raise NotImplementedError(
                "ExPLORe currently supports flat (Box) observation spaces only -- "
                "the pixel variant additionally needs ICVF representation "
                "pretraining, which this port does not implement."
            )
        self.offline_relabel_type = offline_relabel_type
        self.use_rnd_offline = use_rnd_offline
        self.use_rnd_online = use_rnd_online
        self.rnd_coeff = rnd_coeff
        self.relabeler_hidden_dims = tuple(relabeler_hidden_dims)
        self.relabeler_lr = relabeler_lr
        self.relabeler_activation_fn = relabeler_activation_fn
        self.relabeler_kernel_init = relabeler_kernel_init
        self.rnd_hidden_dims = tuple(rnd_hidden_dims)
        self.rnd_feature_dim = rnd_feature_dim
        self.rnd_lr = rnd_lr
        self.rnd_activation_fn = rnd_activation_fn
        self.rnd_kernel_init = rnd_kernel_init
        self._relabeler: Optional[RewardMaskRelabeler] = None
        self._rnd: Optional[RNDBonus] = None
        self._offline_min_reward: Optional[torch.Tensor] = None
        super().__init__(env, eval_env, **rlpd_kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        obs_dim = int(np.prod(self.env.single_observation_space.shape))
        action_dim = int(np.prod(self.env.single_action_space.shape))
        if self.offline_relabel_type != "gt":
            self._relabeler = RewardMaskRelabeler(
                obs_dim,
                action_dim,
                net_arch=self.relabeler_hidden_dims,
                activation_fn=self.relabeler_activation_fn,
                kernel_init=self.relabeler_kernel_init,
            ).to(self.device)
            self._relabeler_optimizer = make_optimizer(
                list(self._relabeler.parameters()), lr=self.relabeler_lr
            )
        if self.use_rnd_offline or self.use_rnd_online:
            self._rnd = RNDBonus(
                obs_dim,
                action_dim,
                feature_dim=self.rnd_feature_dim,
                net_arch=self.rnd_hidden_dims,
                activation_fn=self.rnd_activation_fn,
                kernel_init=self.rnd_kernel_init,
            ).to(self.device)
            self._rnd_optimizer = make_optimizer(
                list(self._rnd.predictor.parameters()), lr=self.rnd_lr
            )

    def load_offline_replay_buffer(self, *args: Any, **kwargs: Any) -> int:
        loaded = super().load_offline_replay_buffer(*args, **kwargs)
        if self.offline_relabel_type == "min":
            full = self.offline_replay_buffer.sample(len(self.offline_replay_buffer))
            self._offline_min_reward = full.rewards.min()
        return loaded

    def _relabel(self, sample):
        if self.offline_relabel_type == "gt" or self._relabeler is None:
            return sample
        if self.offline_relabel_type == "min":
            assert self._offline_min_reward is not None, (
                "offline_relabel_type='min' requires load_offline_replay_buffer() "
                "to have been called first."
            )
            rewards = torch.full_like(sample.rewards, self._offline_min_reward.item())
        else:  # "pred"
            rewards = self._relabeler.predict_reward(sample.obs, sample.actions)
        mask = self._relabeler.predict_mask(sample.obs, sample.actions)
        return dataclasses.replace(sample, rewards=rewards, dones=1.0 - mask)

    def _maybe_add_rnd_bonus(self, sample, enabled: bool):
        if self._rnd is None or not enabled:
            return sample
        bonus = self.rnd_coeff * self._rnd.bonus(sample.obs, sample.actions)
        return dataclasses.replace(sample, rewards=sample.rewards + bonus)

    def _update_relabeler_and_rnd(self, online_sample) -> None:
        n_online = online_sample.rewards.shape[0]
        high_utd_ratio = int(self.utd) if float(self.utd).is_integer() else 1
        steps = max(1, min(high_utd_ratio, n_online))
        minibatch_size = max(1, n_online // steps)
        for j in range(steps):
            mb = self._slice_batch(online_sample, j * minibatch_size, minibatch_size)
            if self._relabeler is not None:
                loss, _ = self._relabeler.loss(mb.obs, mb.actions, mb.rewards, 1.0 - mb.dones)
                self._relabeler_optimizer.zero_grad()
                loss.backward()
                self._relabeler_optimizer.step()
            if self._rnd is not None:
                rnd_loss = self._rnd.loss(mb.obs, mb.actions)
                self._rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self._rnd_optimizer.step()

    def _sample_train_batch(self, batch_size: int):
        # Intentional near-duplication of PriorDataReplayMixin._sample_train_batch
        # (rl_garden/buffers/prior_data_replay.py:133-152) rather than a new
        # parent hook there -- AGENTS.md's parent-hook rule needs >=2 existing
        # subclasses needing it, and ExPLORe was the only one at the time this
        # was written. Resolved: SUPE (rl_garden/algorithms/supe.py) needs the
        # identical offline-relabel-before-mix shape and simply subclasses
        # ExPLORe directly, inheriting this method verbatim -- no new parent
        # hook was needed after all.
        if self.offline_replay_buffer is None or self.offline_data_ratio <= 0.0:
            return self.replay_buffer.sample(batch_size)
        if len(self.offline_replay_buffer) == 0:
            return self.replay_buffer.sample(batch_size)
        if len(self.replay_buffer) == 0:
            offline_sample = self._relabel(self.offline_replay_buffer.sample(batch_size))
            return self._maybe_add_rnd_bonus(offline_sample, self.use_rnd_offline)

        n_offline = int(round(batch_size * self.offline_data_ratio))
        n_offline = min(max(n_offline, 0), batch_size)
        n_online = batch_size - n_offline
        if n_offline == 0:
            return self.replay_buffer.sample(batch_size)
        if n_online == 0:
            offline_sample = self._relabel(self.offline_replay_buffer.sample(batch_size))
            return self._maybe_add_rnd_bonus(offline_sample, self.use_rnd_offline)

        online_sample = self.replay_buffer.sample(n_online)
        offline_sample = self.offline_replay_buffer.sample(n_offline)
        self._update_relabeler_and_rnd(online_sample)
        offline_sample = self._relabel(offline_sample)
        online_sample = self._maybe_add_rnd_bonus(online_sample, self.use_rnd_online)
        offline_sample = self._maybe_add_rnd_bonus(offline_sample, self.use_rnd_offline)
        combined = self._concat_replay_samples(online_sample, offline_sample)
        return self._shuffle_batch(combined, batch_size)

    def _optimizer_names(self) -> tuple[str, ...]:
        names = super()._optimizer_names()
        if self._relabeler is not None:
            names = (*names, "_relabeler_optimizer")
        if self._rnd is not None:
            names = (*names, "_rnd_optimizer")
        return names

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state = super()._extra_checkpoint_state()
        if self._relabeler is not None:
            state["relabeler_state_dict"] = self._relabeler.state_dict()
        if self._rnd is not None:
            state["rnd_state_dict"] = self._rnd.state_dict()
        if self._offline_min_reward is not None:
            state["offline_min_reward"] = self._offline_min_reward
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        if "relabeler_state_dict" in state and self._relabeler is not None:
            self._relabeler.load_state_dict(state["relabeler_state_dict"])
        if "rnd_state_dict" in state and self._rnd is not None:
            self._rnd.load_state_dict(state["rnd_state_dict"])
        if "offline_min_reward" in state:
            self._offline_min_reward = state["offline_min_reward"]

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "offline_relabel_type": self.offline_relabel_type,
            "use_rnd_offline": self.use_rnd_offline,
            "use_rnd_online": self.use_rnd_online,
            "rnd_coeff": self.rnd_coeff,
            "relabeler_hidden_dims": self.relabeler_hidden_dims,
            "relabeler_lr": self.relabeler_lr,
            "rnd_hidden_dims": self.rnd_hidden_dims,
            "rnd_feature_dim": self.rnd_feature_dim,
            "rnd_lr": self.rnd_lr,
        }
