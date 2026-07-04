"""Encoder-agnostic base for PPO variants with a stateful sequence-latent stage
(RNN, GTrXL, ...) between the encoder and actor/critic heads.

Split out of what was originally all of ``RecurrentPPO`` -- every method here
only ever calls the ``SequenceLatentEncoder`` duck-typed API
(``get_initial_state``/``mask_state``/``index_state``/``step``/
``forward_sequence``) or ``self.policy``/``self.rollout_buffer`` methods that
are themselves encoder-agnostic (see ``RecurrentPPOPolicy``,
``RecurrentRolloutBuffer``). Concrete subclasses (``RecurrentPPO``,
``TransformerPPO``) differ only in their constructor's encoder hyperparameters
and in ``_build_sequence_encoder()``.
"""
from __future__ import annotations

from typing import Any, Iterator

import torch

from rl_garden.algorithms.ppo import PPO
from rl_garden.buffers.recurrent_rollout_buffer import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
    RecurrentRolloutBufferSample,
)
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.networks.sequence_encoder import SequenceLatentEncoder
from rl_garden.policies.recurrent_ppo_policy import RecurrentPPOPolicy


class SequencePPO(PPO):
    def _build_sequence_encoder(self, features_extractor) -> SequenceLatentEncoder:
        raise NotImplementedError

    def _setup_model(self) -> None:
        # num_envs/num_minibatches are set earlier in PPO.__init__ (via
        # OnPolicyAlgorithm.__init__), before this method runs -- safe to
        # validate here, before any sequence-encoder module gets constructed.
        if self.num_envs % self.num_minibatches != 0:
            raise ValueError(
                f"{type(self).__name__} requires num_envs ({self.num_envs}) to be "
                f"divisible by num_minibatches ({self.num_minibatches}) for "
                "env-axis minibatching."
            )
        features_extractor = self._build_features_extractor()
        if features_extractor.structured_feature_config() is not None:
            raise NotImplementedError(
                f"{type(self).__name__} only supports flat-latent feature "
                "extractors this round (ViT token_and_prop layouts untested)."
            )
        sequence_encoder = self._build_sequence_encoder(features_extractor)
        self.policy = RecurrentPPOPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            recurrent_encoder=sequence_encoder,
            net_arch=self.net_arch,
            log_std_init=self.log_std_init,
            actor_use_layer_norm=self.actor_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            value_use_group_norm=self.value_use_group_norm,
            num_groups=self.num_groups,
            actor_dropout_rate=self.actor_dropout_rate,
            value_dropout_rate=self.value_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
        ).to(self.device)
        self.policy_optimizer = make_optimizer(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self._lr_scheduler = make_lr_scheduler(
            self.policy_optimizer,
            schedule_type=self.lr_schedule,
            warmup_steps=self.lr_warmup_steps,
            decay_steps=self.lr_decay_steps,
            min_lr_ratio=self.lr_min_ratio,
        )
        buffer_cls = RecurrentDictRolloutBuffer if self._is_dict_obs else RecurrentRolloutBuffer
        self.rollout_buffer = buffer_cls(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _initial_hidden_state(self, batch_size: int):
        return self.policy.recurrent_encoder.get_initial_state(batch_size, self.device)

    def _snapshot_window_initial_hidden(self, hidden, next_done: torch.Tensor):
        return self.policy.recurrent_encoder.mask_state(hidden, 1.0 - next_done.float())

    def _rollout_step(self, obs, hidden, episode_starts: torch.Tensor):
        with torch.no_grad():
            return self.policy.act_recurrent(
                self._obs_to_policy_device(obs),
                hidden,
                episode_starts,
                deterministic=False,
                stop_gradient_actor=self._actor_stop_gradient(),
            )

    def _compute_final_values(self, infos, done_mask: torch.Tensor, hidden) -> torch.Tensor:
        """Bootstrap value for envs that just finished, using the POST-step,
        UNMASKED hidden state (``final_observation`` continues the same episode,
        it is not a fresh start)."""
        final_values = torch.zeros(self.num_envs, device=self.device)
        if "final_observation" not in infos or not done_mask.any():
            return final_values
        final_obs = infos["final_observation"]
        if isinstance(final_obs, dict):
            final_obs = {k: v[done_mask] for k, v in final_obs.items()}
        else:
            final_obs = final_obs[done_mask]
        done_hidden = self.policy.recurrent_encoder.index_state(hidden, done_mask)
        episode_starts = torch.zeros(int(done_mask.sum().item()), device=self.device)
        with torch.no_grad():
            values, _ = self._predict_values_recurrent(
                self._obs_to_policy_device(final_obs), done_hidden, episode_starts
            )
        final_values[done_mask] = values.view(-1)
        return final_values

    def _predict_last_values(self, obs, hidden) -> torch.Tensor:
        episode_starts = torch.zeros(self.num_envs, device=self.device)
        last_values, _ = self._predict_values_recurrent(obs, hidden, episode_starts)
        return last_values.view(-1)

    def _predict_values_recurrent(self, obs, hidden, episode_starts: torch.Tensor):
        """Private helper reused by _compute_final_values and _predict_last_values."""
        with torch.no_grad():
            return self.policy.predict_values_recurrent(obs, hidden, episode_starts)

    def _iter_minibatches(self) -> Iterator[RecurrentRolloutBufferSample]:
        return self.rollout_buffer.get_sequences(
            self.num_minibatches, initial_hidden=self._rollout_initial_hidden
        )

    def _evaluate_minibatch(self, data: RecurrentRolloutBufferSample):
        values, log_prob, entropy = self.policy.evaluate_actions_sequence(
            data.obs,
            data.actions,
            data.initial_hidden,
            data.episode_starts,
            stop_gradient_actor=self._actor_stop_gradient(),
        )
        # (T,B,1) tensors; T-major flatten so index [t,b] lands at the same flat
        # offset across all seven tensors below.
        return (
            values.flatten(),
            log_prob.flatten(),
            entropy.flatten(),
            data.old_values.reshape(-1),
            data.old_log_prob.reshape(-1),
            data.advantages.reshape(-1),
            data.returns.reshape(-1),
        )

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        # NOTE: the live rollout hidden state (self.learn()'s local `hidden`
        # variable) is intentionally NOT persisted here. A resumed run
        # re-zero-initializes it at the start of learn(); episode-boundary
        # masking self-heals within one episode length. Accepted limitation,
        # not a defect.
        return super()._extra_checkpoint_state()
