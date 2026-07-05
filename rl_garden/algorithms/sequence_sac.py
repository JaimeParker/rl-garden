"""Encoder-agnostic base for SAC variants with a stateful sequence-latent stage
(RNN, GTrXL, ...) between the encoder and actor/critic heads.

Split out of what was originally all of ``RecurrentSAC`` -- every method here
only ever calls the ``SequenceLatentEncoder`` duck-typed API
(``get_initial_state``/``mask_state``/``step``/``forward_sequence_with_burn_in``)
or ``self.policy``/``self.replay_buffer`` methods that are themselves
encoder-agnostic (see ``RecurrentSACPolicy``). Concrete subclasses
(``RecurrentSAC``, a future ``TransformerSAC``) differ only in their
constructor's encoder hyperparameters, ``_build_sequence_encoder()``,
``_build_replay_buffer()`` (buffer construction needs encoder-shape-specific
parameters that intentionally stay out of this base), and
``_initial_state_from_sample()`` (how a replay sample's stored state is
unpacked into a ``SequenceState``).
"""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.algorithms.sac import SAC
from rl_garden.networks.sequence_encoder import SequenceLatentEncoder, SequenceState
from rl_garden.policies.recurrent_sac_policy import RecurrentSACPolicy


class SequenceSAC(SAC):
    def __init__(
        self,
        env: Any,
        *,
        burn_in_len: int = 40,
        learning_len: int = 40,
        forward_len: int = 5,
        prio_exponent: float = 0.9,
        importance_sampling_exponent: float = 0.6,
        **sac_kwargs: Any,
    ) -> None:
        if sac_kwargs.get("nstep", 1) != 1:
            raise ValueError(
                f"{type(self).__name__} uses forward_len for n-step bootstrapping "
                "(integrated with burn-in), not the flat nstep kwarg."
            )
        utd = sac_kwargs.get("utd", 0.5)
        # Matches SACCore.train()'s own test for entering train_high_utd() --
        # an integer-valued float (e.g. utd=2.0) must be rejected too, not just
        # a Python int, or it silently slips into _slice_batch()'s generic
        # x[start:end] slicing, which would slice (learning_len, B)-shaped
        # fields along the wrong axis instead of the batch axis.
        if float(utd).is_integer() and utd > 1:
            raise ValueError(
                f"{type(self).__name__} v1 does not support integer-valued utd > 1 -- "
                "train_high_utd()'s minibatch splitting is not wired up for "
                "(learning_len, B)-shaped replay fields yet."
            )
        # Must be set before super().__init__(), which ends by calling
        # self._setup_model() -- our _build_policy/_build_replay_buffer
        # overrides below read these attributes.
        self.burn_in_len = burn_in_len
        self.learning_len = learning_len
        self.forward_len = forward_len
        self.prio_exponent = prio_exponent
        self.importance_sampling_exponent = importance_sampling_exponent
        super().__init__(env, **sac_kwargs)

    # --- subclass hooks ---

    def _build_sequence_encoder(self, features_extractor) -> SequenceLatentEncoder:
        raise NotImplementedError

    def _initial_state_from_sample(self, data) -> SequenceState:
        raise NotImplementedError

    # --- model construction (override existing SAC hooks, no _setup_model rewrite) ---

    def _build_policy(self, features_extractor) -> RecurrentSACPolicy:
        if features_extractor.structured_feature_config() is not None:
            raise NotImplementedError(
                f"{type(self).__name__} only supports flat-latent feature "
                "extractors this round (ViT token_and_prop layouts untested)."
            )
        sequence_encoder = self._build_sequence_encoder(features_extractor)
        return RecurrentSACPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            features_extractor=features_extractor,
            recurrent_encoder=sequence_encoder,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            critic_impl=self.critic_impl,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            log_std_min=self.actor_log_std_min,
            log_std_mode=self.actor_log_std_mode,
        )

    # --- rollout hidden-state carry (pure subclass overrides of existing hooks) ---

    def _on_env_reset(self, obs) -> None:
        self._rollout_hidden = self.policy.recurrent_encoder.get_initial_state(
            self.num_envs, self.device
        )
        self._rollout_episode_start = torch.ones(self.num_envs, device=self.device)

    def _rollout_action(self, obs, learning_has_started: bool):
        del learning_has_started  # recurrent hidden state advances regardless of phase
        with torch.no_grad():
            action, new_hidden = self.policy.act_recurrent_step(
                self._obs_to_policy_device(obs),
                self._rollout_hidden,
                self._rollout_episode_start,
                deterministic=False,
            )
        action_context = {"hidden_pre_step": self._rollout_hidden, "new_hidden": new_hidden}
        return action, action, action_context

    def _replay_buffer_add_kwargs(
        self, action_context, obs, next_obs, real_next_obs, infos, need_final_obs
    ) -> dict[str, Any]:
        del obs, next_obs, real_next_obs, infos, need_final_obs
        return {"hidden": action_context["hidden_pre_step"]}

    def _replay_buffer_step_kwargs(self, terminations, truncations) -> dict[str, Any]:
        return {"episode_end": terminations | truncations}

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        del infos
        self._rollout_hidden = action_context["new_hidden"]
        self._rollout_episode_start = (terminations | truncations).float()

    def _eval_start_hook(self) -> None:
        # NOTE: deliberately does NOT call super()._eval_start_hook() (SACCore's
        # q_mc_diagnostics setup) -- SACCore._eval_q_values() assumes a flat
        # features->Q pipeline and calls policy.extract_features() directly,
        # bypassing the sequence encoder entirely, so its Q values would be
        # computed from the wrong (pre-encoder, wrong-dimensioned) features.
        # q_mc_diagnostics is unsupported for sequence SAC variants v1 --
        # silently absent, not silently wrong, matching
        # _compute_actor_diagnostics/_compute_q_landscape_diagnostics' identical
        # scope decision below.
        self._eval_hidden = self.policy.recurrent_encoder.get_initial_state(
            self.eval_env.num_envs, self.device
        )
        self._eval_episode_start = torch.ones(self.eval_env.num_envs, device=self.device)

    def _eval_action_and_critic_action(self, obs):
        with torch.no_grad():
            action, new_hidden = self.policy.act_recurrent_step(
                self._obs_to_policy_device(obs),
                self._eval_hidden,
                self._eval_episode_start,
                deterministic=True,
            )
        self._eval_hidden = new_hidden
        return action, action

    def _eval_step_hook(
        self, obs_before, critic_action, rewards, terminations, truncations, infos
    ) -> None:
        # Reset exactly at a genuine episode boundary (this hook is the only
        # one in the eval loop that actually receives terminations/truncations
        # -- _eval_action_and_critic_action does not, so resetting there was
        # never correct: it would either never reset, or reset every step).
        del obs_before, critic_action, rewards, infos
        self._eval_episode_start = (terminations | truncations).float()

    # --- training-side ---
    # NOTE: _extra_batch_slice_keys (SACCore's _slice_batch() extension point)
    # is intentionally NOT declared here. It exists only to support
    # train_high_utd()'s mid-batch _slice_batch() splitting, which this class
    # permanently rejects at construction (integer-valued utd > 1) precisely
    # because _slice_batch()'s generic x[start:end] slicing would slice
    # (learning_len, B)-shaped fields along the wrong axis -- so the
    # declaration would be dead code with no reachable call site.

    def _critic_loss(self, data):
        stop_gradient = not self._training_update_mask().update_encoder
        initial_hidden = self._initial_state_from_sample(data)
        tail = self.policy.windowed_features(
            data.obs, initial_hidden, data.episode_starts, self.burn_in_len,
            stop_gradient=stop_gradient,
        )
        online = tail[: self.learning_len]
        # Explicit .detach() (NOT a torch.no_grad() around the whole unroll --
        # that would also kill the online slice's gradient to the encoder/RNN,
        # since both slices come from the SAME differentiable unroll).
        target = tail[self.forward_len : self.forward_len + self.learning_len].detach()

        hidden_size = online.shape[-1]
        online_flat = online.reshape(-1, hidden_size)
        actions_flat = data.actions.reshape(-1, data.actions.shape[-1])
        q_pred = self.policy.q_values_all(online_flat, actions_flat, target=False)

        alpha = self._current_alpha().detach()
        target_flat = target.reshape(-1, hidden_size)
        with torch.no_grad():
            next_action, next_log_prob = self.policy.actor.action_log_prob(target_flat)
            min_q_next = self.policy.min_q_value(
                target_flat, next_action,
                subsample_size=self._target_critic_subsample_size(), target=True,
            )
            if self._backup_entropy_enabled():
                min_q_next = min_q_next - alpha * next_log_prob
            target_q = data.rewards.reshape(-1, 1) + data.discounts.reshape(-1, 1) * min_q_next

        # Flatten order is row-major over (learning_len, B): position t*B+b.
        # Broadcast each window's IS weight across its learning_len positions
        # and all critics accordingly.
        is_weights_flat = (
            data.is_weights.unsqueeze(0).expand(self.learning_len, -1).reshape(-1)
        )
        is_weights_bc = is_weights_flat.reshape(1, -1, 1)
        target_q_expanded = target_q.unsqueeze(0).expand_as(q_pred)
        td_loss = (is_weights_bc * (q_pred - target_q_expanded).pow(2)).mean()

        # R2D2-style mixed priority: 0.9*max + 0.1*mean of |td_error| over the
        # learning axis, per sampled window (worker.py's calculate_mixed_td_errors).
        with torch.no_grad():
            batch_size = data.is_weights.shape[0]
            per_position_error = (
                (q_pred.mean(dim=0) - target_q).abs().reshape(self.learning_len, batch_size)
            )
            td_error_priority = (
                0.9 * per_position_error.max(dim=0).values + 0.1 * per_position_error.mean(dim=0)
            )

        info = {
            "td_loss": td_loss.detach(),
            "target_q": target_q.mean().detach(),
            "critic_loss": td_loss.detach(),
            "predicted_q": q_pred.mean().detach(),
            "td_error_priority": td_error_priority,
        }
        return td_loss, info

    def _actor_loss_from_batch(self, data):
        initial_hidden = self._initial_state_from_sample(data)
        # Shorter unroll than the critic's: actor loss never needs the
        # n-step-ahead bootstrap tail.
        actor_window_len = self.burn_in_len + self.learning_len
        obs_window = self._slice_obs_window(data.obs, actor_window_len)
        episode_starts_window = data.episode_starts[:actor_window_len]
        features = self.policy.windowed_features(
            obs_window, initial_hidden, episode_starts_window, self.burn_in_len,
            stop_gradient=False,
        )
        # Detach the encoder's OUTPUT here, not the pre-encoder raw features --
        # passing stop_gradient into windowed_features would only cut gradient
        # into the feature extractor (detaching an upstream tensor never blocks
        # gradient to a downstream layer's own parameters), leaving the
        # sequence encoder itself still trained by the actor loss. Matches
        # RecurrentPPOPolicy's identical "detach latent, not raw features" fix
        # for the same shared-trunk problem.
        if self._actor_stop_gradient():
            features = features.detach()
        hidden_size = features.shape[-1]
        flat_features = features.reshape(-1, hidden_size)

        alpha = self._current_alpha().detach()
        action, log_prob = self.policy.actor.action_log_prob(flat_features)
        min_q = self.policy.min_q_value(flat_features, action, subsample_size=None, target=False)
        actor_loss = (alpha * log_prob - min_q).mean()
        return actor_loss, log_prob.detach()

    def _post_critic_update(self, data, critic_info) -> None:
        self.replay_buffer.update_priorities(
            data.priority_indices, critic_info["td_error_priority"]
        )

    def _compute_actor_diagnostics(self, data) -> dict[str, torch.Tensor]:
        # The default (SACCore._compute_actor_diagnostics) assumes flat
        # per-transition obs; data.obs here is a (window_len, B, ...) window.
        # No-op for v1 -- a documented diagnostics gap, not a correctness gap.
        # Same reasoning applies to _q_landscape_diagnostics below and to
        # _eval_start_hook's skipped q_mc_diagnostics setup above.
        return {}

    def _q_landscape_diagnostics(self, data) -> dict[str, torch.Tensor]:
        # Override the full SACCore method (not just an inner hook) so this
        # stays a pure-addition subclass -- SACCore.q_landscape_diagnostics's
        # flag-check + RNG-fork wrapping assumes data.obs is flat per-transition
        # obs (SACPolicy.q_landscape_diagnostics calls extract_features(obs,
        # ...) directly, bypassing the sequence encoder), which would
        # misinterpret data.obs's (window_len, B, ...) shape here regardless of
        # what happens inside the wrapper. No-op for v1 (only matters if a
        # caller explicitly opts in via q_landscape_diagnostics=True).
        return {}

    @staticmethod
    def _slice_obs_window(obs, length: int):
        if isinstance(obs, dict):
            return {k: v[:length] for k, v in obs.items()}
        return obs[:length]

    # --- checkpointing ---

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "burn_in_len": self.burn_in_len,
            "learning_len": self.learning_len,
            "forward_len": self.forward_len,
            "prio_exponent": self.prio_exponent,
            "importance_sampling_exponent": self.importance_sampling_exponent,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        # NOTE: the live rollout hidden state (self._rollout_hidden) is
        # intentionally NOT persisted here, matching RecurrentPPO's identical
        # precedent -- a resumed run re-zero-initializes it in _on_env_reset().
        # The replay buffer's priority tree / checkpoint side-buffer / episode
        # bookkeeping are also not persisted, matching the pre-existing gap in
        # NStepDictReplayBuffer's checkpoint support.
        return super()._extra_checkpoint_state()
