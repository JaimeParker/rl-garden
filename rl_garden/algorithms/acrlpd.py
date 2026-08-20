"""ACRLPD: Q-chunking's action-chunked RLPD (``3rd_party/qc/agents/acrlpd.py``).

Extends rl-garden's existing ``RLPD`` (itself a thin ``SAC`` extension --
ensemble critics, LayerNorm, high UTD, ``PriorDataReplayMixin``'s
ratio-mixed offline/online sampling from step 0) with action chunking:
``horizon_length`` consecutive raw actions are folded into one flat
macro-action for the actor and critic, and the critic bootstraps off an
H-step-ahead observation instead of the immediate next one.

Reused from ``RLPD``/``SAC``/``SACPolicy`` verbatim (verified against
``3rd_party/qc/agents/acrlpd.py`` directly, not from memory):
  - ``SquashedGaussianActor``/``EnsembleQCritic`` are already
    action-dim-agnostic (``act_dim = prod(action_space.shape)``) -- built
    with a synthetic flat ``(horizon_length * action_dim,)`` action space via
    ``_policy_action_space()``, no new policy/network classes needed.
  - ``_actor_loss``'s formula shape, ``alpha*log_prob - Q``, is identical to
    ``acrlpd.py:71-84``'s ``actor_loss`` -- just operating over the joint
    macro-action distribution.
  - ``PriorDataReplayMixin``'s two-buffer, ratio-mixed-from-step-0 sampling
    already matches ``main_online.py``'s 50/50 offline/online split-batch
    model (no offline-only pretraining phase, matching ACRLPD's own
    reference exactly, unlike ``Off2OnReplayMixin``'s phase-switch shape
    used for ``ACFQL``).

Three numerically load-bearing divergences from plain ``RLPD``/``SAC``,
each verified against ``acrlpd.py`` source directly:
  1. **No entropy correction in the critic's Bellman target by default**
     (``backup_entropy=False``) -- ``acrlpd.py:40-58``'s ``target_q`` has no
     ``-alpha*next_log_prob`` term, unlike standard SAC. Kept as a live,
     overridable flag (not hardcoded) via the existing
     ``_backup_entropy_enabled()`` hook, rather than silently dropping the
     capability.
  2. **Mean, not min, ensemble aggregation by default** (``q_agg="mean"``)
     for the critic's target -- ``acrlpd.py:47``: ``next_qs.mean(axis=0)``,
     not rl-garden's ``SACPolicy.min_q_value`` (which is hardcoded to
     ``.min(dim=0)`` and shared by many other policy families -- not touched
     here; this class calls the more primitive ``q_values_subsampled``
     directly and aggregates inline instead). Note ``acrlpd.py:66``'s actor
     loss hardcodes ``q = jnp.mean(qs, axis=0)`` unconditionally -- ``q_agg``
     only ever gates the critic's bootstrap target, never the actor's own
     policy-improvement Q; ``ACRLPDCore._actor_loss`` reproduces this by
     calling ``qs.mean(dim=0)`` directly rather than the config-gated
     ``_aggregate_q`` helper (that helper is only correct for the critic
     target, ``_target_q``/``_aggregate_q``, below).
  3. **No REDQ critic subsampling by default** (``critic_subsample_size=
     None``) -- ``subsample_ensemble`` exists in QC's own
     ``rlpd_networks/ensemble.py`` but is never called in ``acrlpd.py``;
     the reference uses its full 10-critic ensemble every update, unlike
     plain rl-garden ``RLPD``'s default (``critic_subsample_size=2``).

``target_entropy`` is computed explicitly as
``-target_entropy_multiplier * horizon_length * action_dim``
(``target_entropy_multiplier=0.5`` default, ``acrlpd.py:191-199``) and passed
as a literal float to ``SAC.__init__`` -- NOT ``"auto"``, which would compute
``-prod(action_space.shape)`` from the (chunked) policy action space, giving
a different multiplier (1.0) than QC's own default (0.5).

Rollout: no env wrapper. ``ChunkedRolloutMixin`` handles the
receding-horizon action queue (see ``_chunked_rollout.py``); the raw env's
``env.step()`` stays exactly 1:1 with ``OffPolicyAlgorithm.learn()``'s
existing per-step loop, matching QC's own single-env-step rollout
(``main_online.py:179-190``) rather than an atomic multi-step wrapper.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms._chunked_rollout import ChunkedRolloutMixin
from rl_garden.algorithms.rlpd import RLPD
from rl_garden.buffers.chunked_replay_buffer import ChunkedTensorReplayBuffer


class ACRLPDCore:
    """Chunked-critic/actor loss overrides, mixed into ``RLPD``'s MRO ahead
    of ``SACCore`` so these take precedence over the base SAC loss shape."""

    def _init_acrlpd_params(
        self,
        *,
        horizon_length: int,
        q_agg: Literal["mean", "min"],
        bc_alpha: float,
    ) -> None:
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}")
        if bc_alpha < 0:
            raise ValueError(f"bc_alpha must be >= 0, got {bc_alpha}")
        self.horizon_length = horizon_length
        self.q_agg = q_agg
        self.bc_alpha = bc_alpha
        # SAC.__init__ only appends "discounts" here when self.nstep > 1 (a
        # knob ACRLPD doesn't use -- chunking always needs windowed fields,
        # unconditionally). Set before SAC.__init__ runs so its own
        # nstep==1 no-op branch leaves this untouched.
        self._extra_batch_slice_keys = ("discounts", "valid")

    # --- action-space / buffer plumbing ---

    def _policy_action_space(self) -> spaces.Box:
        raw = self.env.single_action_space
        assert isinstance(raw, spaces.Box), "ACRLPD requires a flat Box action space."
        low = np.tile(np.asarray(raw.low, dtype=np.float32).reshape(-1), self.horizon_length)
        high = np.tile(np.asarray(raw.high, dtype=np.float32).reshape(-1), self.horizon_length)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_replay_buffer(self) -> ChunkedTensorReplayBuffer:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("ACRLPD is state-only (Box observations); vision is out of scope.")
        return ChunkedTensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            horizon_length=self.horizon_length,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _build_prior_data_buffer(self, buffer_size: int) -> ChunkedTensorReplayBuffer:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("ACRLPD is state-only (Box observations); vision is out of scope.")
        return ChunkedTensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=buffer_size,
            horizon_length=self.horizon_length,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _replay_buffer_step_kwargs(
        self, terminations: torch.Tensor, truncations: torch.Tensor
    ) -> dict[str, Any]:
        return {"episode_end": terminations | truncations}

    def _sample_train_batch(self, batch_size: int):
        # Flatten the (B, horizon_length, action_dim) window gathered by
        # ChunkedTensorReplayBuffer into the (B, horizon_length*action_dim)
        # macro-action the (chunked-action-space) critic/actor expect --
        # matches QC's own `batch_actions = reshape(actions, (B, -1))`
        # (acrlpd.py:44), done once here so every downstream loss sees an
        # already-flat `data.actions`.
        data = super()._sample_train_batch(batch_size)
        flat_actions = data.actions.reshape(data.actions.shape[0], -1)
        return dataclasses.replace(data, actions=flat_actions)

    # --- chunked losses ---
    #
    # No `valid[..., -1]`-style critic-loss masking here (unlike
    # acrlpd.py:44's `* batch['valid'][..., -1]`): QC's reference always
    # reads a full horizon_length window and discards the ENTIRE sample when
    # a terminal/truncation falls before the last position. This buffer
    # instead stops reward/discount accumulation exactly at the first
    # terminal-or-truncation (see chunked_replay_buffer.py's module
    # docstring) and produces a correct, unbiased partial-length target for
    # every sampled window -- nothing here is ever "garbage that needs
    # masking out", so every sample is already usable as-is.

    def _target_discounts(self, data) -> torch.Tensor:
        return data.discounts.reshape(-1, 1)

    def _aggregate_q(self, q_all: torch.Tensor) -> torch.Tensor:
        return q_all.mean(dim=0) if self.q_agg == "mean" else q_all.min(dim=0).values

    def _target_q(self, data) -> torch.Tensor:
        alpha = self._current_alpha().detach()
        with torch.no_grad():
            next_action, next_log_prob, next_features = self._target_action_log_prob(data)
            next_qs = self.policy.q_values_subsampled(
                next_features,
                next_action,
                subsample_size=self._target_critic_subsample_size(),
                target=True,
            )
            next_q = self._aggregate_q(next_qs)
            if self._backup_entropy_enabled():
                next_q = next_q - alpha * next_log_prob
            target = data.rewards.reshape(-1, 1) + self._target_discounts(data) * next_q
        return target

    def _actor_loss(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._current_alpha().detach()
        action, log_prob, features = self._actor_action_log_prob(
            obs, stop_gradient=self._actor_stop_gradient()
        )
        qs = self.policy.q_values_subsampled(features, action, subsample_size=None, target=False)
        # acrlpd.py:84's actor_loss hardcodes `q = jnp.mean(qs, axis=0)` --
        # unlike the critic target, q_agg ("min" vs "mean") only applies to
        # the critic's bootstrap, never to the actor's own policy-improvement
        # Q. Using self._aggregate_q here (q_agg-gated) would be a no-op
        # under the default q_agg="mean" but silently switch to min-Q for the
        # actor under q_agg="min", diverging from the reference.
        q = qs.mean(dim=0)
        return (alpha * log_prob - q).mean(), log_prob.detach()

    def _actor_loss_from_batch(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        actor_loss, log_prob_detached = self._actor_loss(data.obs)
        if self.bc_alpha <= 0.0:
            return actor_loss, log_prob_detached
        # data.actions is already flattened by _sample_train_batch.
        bc_log_prob = self.policy.evaluate_action_log_prob(
            data.obs, data.actions, stop_gradient=self._actor_stop_gradient()
        )
        bc_loss = -bc_log_prob.mean() * self.bc_alpha
        return actor_loss + bc_loss, log_prob_detached

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_length": self.horizon_length,
            "q_agg": self.q_agg,
            "bc_alpha": self.bc_alpha,
        }


class ACRLPD(ChunkedRolloutMixin, ACRLPDCore, RLPD):
    """Q-chunking's action-chunked RLPD. See module docstring."""

    _compatible_checkpoint_algorithms = ("ACRLPD",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        horizon_length: int = 5,
        target_entropy_multiplier: float = 0.5,
        q_agg: Literal["mean", "min"] = "mean",
        bc_alpha: float = 0.0,
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = None,
        critic_use_layer_norm: bool = True,
        backup_entropy: bool = False,
        target_entropy: float | str = "auto",
        **rlpd_kwargs: Any,
    ) -> None:
        self._init_acrlpd_params(horizon_length=horizon_length, q_agg=q_agg, bc_alpha=bc_alpha)
        self._init_chunked_rollout()

        if target_entropy == "auto":
            action_dim = int(np.prod(env.single_action_space.shape))
            target_entropy = -target_entropy_multiplier * horizon_length * action_dim

        super().__init__(
            env,
            eval_env,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            critic_use_layer_norm=critic_use_layer_norm,
            backup_entropy=backup_entropy,
            target_entropy=target_entropy,
            **rlpd_kwargs,
        )

    def _sample_action_chunk(self, obs) -> torch.Tensor:
        flat_action = self.policy.predict(obs, deterministic=False)
        act_dim = int(np.prod(self.env.single_action_space.shape))
        return flat_action.reshape(flat_action.shape[0], self.horizon_length, act_dim)
