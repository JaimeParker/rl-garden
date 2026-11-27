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

Verified critic and actor choices relative to plain ``RLPD``/``SAC``,
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
  4. **Mean over critics and the whole batch**, with incomplete windows
     weighted zero, without renormalizing by the number of complete windows
     (``3rd_party/qc/agents/acrlpd.py:61-62``). Every online critic is updated;
     target subsampling does not change the loss reduction.

``target_entropy`` is computed explicitly as
``-target_entropy_multiplier * horizon_length * action_dim``
(``target_entropy_multiplier=0.5`` default, ``acrlpd.py:191-199``) and passed
as a literal float to ``SAC.__init__`` -- NOT ``"auto"``, which would compute
``-prod(env.single_action_space.shape)`` from the raw environment action
space (minus the per-step action dimension), without a horizon multiplier.

Rollout: no env wrapper. ``ChunkedRolloutMixin`` handles the
receding-horizon action queue (see ``chunked_rollout.py``); the raw env's
``env.step()`` stays exactly 1:1 with ``OffPolicyAlgorithm.learn()``'s
existing per-step loop, matching QC's own single-env-step rollout
(``main_online.py:179-190``) rather than an atomic multi-step wrapper.

Action-space tiling, chunked replay construction, training-batch flattening,
evaluation queues and horizon metadata live in ``ActionChunkingMixin``.
"""
from __future__ import annotations

import dataclasses

from typing import Any, Optional

import numpy as np
import torch

from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.sac import ACRLPDParams
from rl_garden.algorithms.chunked_rollout import ChunkedRolloutMixin
from rl_garden.algorithms.rlpd import RLPD
from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.observations import (
    ObservationContractError,
    ObservationSchema,
    normalize_observation_space,
)
from rl_garden.policies.contracts import SACActorCapabilities


class ACRLPDCore:
    """Chunked-critic/actor loss overrides, mixed into ``RLPD``'s MRO ahead
    of ``SACCore`` so these take precedence over the base SAC loss shape."""

    def _init_acrlpd_params(
        self,
        *,
        params: ACRLPDParams,
    ) -> None:
        horizon_length = params.horizon_length
        q_agg = params.q_agg
        bc_alpha = params.bc_alpha
        self._init_action_chunking(horizon_length)
        if bc_alpha < 0:
            raise ValueError(f"bc_alpha must be >= 0, got {bc_alpha}")
        self.q_agg = q_agg
        self.bc_alpha = bc_alpha
        # SAC.__init__ only appends "discounts" here when self.nstep > 1 (a
        # knob ACRLPD doesn't use -- chunking always needs windowed fields,
        # unconditionally). Set before SAC.__init__ runs so its own
        # nstep==1 no-op branch leaves this untouched.
        self._extra_batch_slice_keys = ("discounts", "valid")

    def _check_actor_capabilities(self, capabilities: SACActorCapabilities) -> None:
        super()._check_actor_capabilities(capabilities)
        if self.bc_alpha > 0 and not capabilities.supports_evaluate_log_prob:
            raise ValueError(
                f"bc_alpha={self.bc_alpha} scores the dataset actions with the actor's "
                "evaluate_action_log_prob, but the actor declares "
                "supports_evaluate_log_prob=False; pass bc_alpha=0.0."
            )

    # --- action-space / buffer plumbing ---

    def _setup_model(self) -> None:
        """Reject images before constructing networks, optimizers or buffers."""
        obs_space = normalize_observation_space(self.env.single_observation_space)
        if ObservationSchema.from_space(obs_space).has_images:
            raise ObservationContractError(
                "ACRLPD is state-only; vision is out of scope."
            )
        super()._setup_model()

    def _build_prior_data_buffer(self, buffer_size: int) -> ChunkedReplayBuffer:
        return self._chunked_replay_buffer(num_envs=1, buffer_size=buffer_size)

    # --- chunked losses ---

    def _td_loss(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Mean over critics and batch, with incomplete windows weighted zero
        (``3rd_party/qc/agents/acrlpd.py:62``)."""
        target_q = self._target_q(data)
        complete_window = data.valid[:, -1].to(target_q.dtype).reshape(-1, 1)
        q_loss = torch.stack([
            ((q - target_q).square() * complete_window).mean() for q in q_pred
        ]).mean()
        return q_loss, {
            "td_loss": q_loss.detach(),
            "target_q": target_q.mean().detach(),
        }

    def _target_discounts(self, data) -> torch.Tensor:
        return data.discounts.reshape(-1, 1)

    def _aggregate_q(self, q_all: torch.Tensor) -> torch.Tensor:
        return q_all.mean(dim=0) if self.q_agg == "mean" else q_all.min(dim=0).values

    def _target_q(self, data) -> torch.Tensor:
        alpha = self._current_alpha().detach()
        with torch.no_grad():
            next_action, next_log_prob, next_actor_features = self._target_action_log_prob(data)
            next_critic_features = self.policy.critic_features_for(
                data.next_obs, next_actor_features, stop_gradient=True
            )
            next_qs = self.policy.q_values_subsampled(
                next_critic_features,
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
        # No explicit stop_gradient: SACPolicy.actor_action_log_prob applies
        # BasePolicy.extract_actor_features's encoder_sharing rule by default.
        action, log_prob, actor_features = self._actor_action_log_prob(obs)
        critic_features = self.policy.critic_features_for(obs, actor_features, stop_gradient=True)
        qs = self.policy.q_values_subsampled(
            critic_features, action, subsample_size=None, target=False
        )
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
        bc_log_prob = self.policy.evaluate_action_log_prob(data.obs, data.actions)
        bc_loss = -bc_log_prob.mean() * self.bc_alpha
        return actor_loss + bc_loss, log_prob_detached

    def _q_mc_finish_episode(
        self, env_idx: int, bootstrap_v: Optional[torch.Tensor]
    ) -> None:
        """Keep the decision steps, where the critic scores the chunk chosen for
        that observation: each env starts a chunk at its episode start and then
        every ``horizon_length`` steps."""
        super()._q_mc_finish_episode(env_idx, bootstrap_v)
        for completed in (
            self._q_mc_completed_q,
            self._q_mc_completed_mc,
            self._q_mc_completed_err,
        ):
            completed[-1] = completed[-1][:: self.horizon_length]

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "q_agg": self.q_agg,
            "bc_alpha": self.bc_alpha,
        }


class ACRLPD(ChunkedRolloutMixin, ACRLPDCore, RLPD):
    """Q-chunking's action-chunked RLPD. See module docstring."""

    _compatible_checkpoint_algorithms = ("ACRLPD",)

    params_class = ACRLPDParams

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        initial_training_phase: Optional[InitialTrainingPhase] = None,
        params: ACRLPDParams | None = None,
        **rlpd_kwargs: Any,
    ) -> None:
        parameter_names = self.params_class.__dataclass_fields__
        overrides = {name: rlpd_kwargs.pop(name) for name in tuple(rlpd_kwargs)
                     if name in parameter_names}
        params = resolve_params(self.params_class, params, overrides)
        horizon_length = params.horizon_length
        target_entropy_multiplier = params.target_entropy_multiplier
        target_entropy = params.target_entropy
        actor_q_agg = params.actor_q_agg
        if actor_q_agg != "min":
            raise ValueError(
                f"ACRLPD does not take actor_q_agg={actor_q_agg!r} (--actor-q-agg): its actor "
                "loss always averages the critic ensemble, as its reference does. Its own "
                "ensemble aggregation is q_agg (--q-agg), which applies to the critic target."
            )
        if initial_training_phase is not None:
            raise ValueError(
                "ACRLPD does not take an initial_training_phase (--critic-only-steps): "
                "Q-chunking has no critic-only phase, only random actions before learning "
                "starts (3rd_party/qc/main_online.py:179-188)."
            )
        self._init_acrlpd_params(params=params)

        if target_entropy == "auto":
            action_dim = int(np.prod(env.single_action_space.shape))
            target_entropy = -target_entropy_multiplier * horizon_length * action_dim

        input_params = params
        params = dataclasses.replace(params, target_entropy=target_entropy)
        super().__init__(
            env,
            eval_env,
            params=params,
            **rlpd_kwargs,
        )
        self.params = input_params
