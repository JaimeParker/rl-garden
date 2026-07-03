"""Recurrent variant of PPOPolicy: an RNN latent stage between the encoder and heads."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, KernelInit, RecurrentLatentEncoder, RecurrentState
from rl_garden.policies.ppo_policy import PPOPolicy


class RecurrentPPOPolicy(PPOPolicy):
    """PPOPolicy with a RecurrentLatentEncoder between features_extractor and the heads."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        recurrent_encoder: RecurrentLatentEncoder,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        *,
        log_std_init: float = -0.5,
        actor_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        # recurrent_encoder.features_dim is read here (a plain property, safe
        # before nn.Module registration); self.recurrent_encoder is assigned
        # AFTER super().__init__() returns, since nn.Module.__setattr__ requires
        # nn.Module.__init__() (called inside super().__init__()) to have
        # already run before any submodule attribute can be set.
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            net_arch,
            features_dim=recurrent_encoder.features_dim,
            log_std_init=log_std_init,
            actor_use_layer_norm=actor_use_layer_norm,
            value_use_layer_norm=value_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            value_use_group_norm=value_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            value_dropout_rate=value_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.recurrent_encoder = recurrent_encoder

    def act_recurrent(
        self,
        obs: Obs,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
        *,
        stop_gradient_actor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RecurrentState]:
        """Single rollout step. Returns (actions, values, log_prob, entropy, new_hidden)."""
        raw = self._extract_features(obs, stop_gradient=False)
        latent, new_hidden = self.recurrent_encoder.step(raw, hidden, episode_starts)
        actor_latent = latent.detach() if stop_gradient_actor else latent
        actions, log_prob, entropy = self.actor.action_log_prob(
            actor_latent, deterministic=deterministic
        )
        values = self.value_net(latent)
        return actions, values, log_prob, entropy, new_hidden

    def predict_recurrent(
        self,
        obs: Obs,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, RecurrentState]:
        raw = self._extract_features(obs, stop_gradient=False)
        latent, new_hidden = self.recurrent_encoder.step(raw, hidden, episode_starts)
        if deterministic:
            action = self.actor.clamp_action(self.actor.deterministic_action(latent))
        else:
            action, _, _ = self.actor.action_log_prob(latent, deterministic=False)
            action = self.actor.clamp_action(action)
        return action, new_hidden

    def predict_values_recurrent(
        self, obs: Obs, hidden: RecurrentState, episode_starts: torch.Tensor
    ) -> tuple[torch.Tensor, RecurrentState]:
        raw = self._extract_features(obs, stop_gradient=False)
        latent, new_hidden = self.recurrent_encoder.step(raw, hidden, episode_starts)
        return self.value_net(latent), new_hidden

    def evaluate_actions_sequence(
        self,
        obs: Obs,
        actions: torch.Tensor,
        initial_hidden: RecurrentState,
        episode_starts: torch.Tensor,
        *,
        stop_gradient_actor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BPTT update path. obs/actions: (T, B, ...). Returns (values, log_prob,
        entropy), each (T, B, 1).

        The encoder sees no time axis (T*B flattened -> encode -> reshaped back to
        (T, B, -1)); the RNN unrolls over the (T, B, -1) latent.
        """
        num_steps, num_envs = actions.shape[0], actions.shape[1]
        flat_obs = flatten_leading_dims(obs)
        raw = self._extract_features(flat_obs, stop_gradient=False)
        raw = raw.reshape(num_steps, num_envs, -1)
        latent, _ = self.recurrent_encoder.forward_sequence(raw, initial_hidden, episode_starts)

        # Detach the RNN's OUTPUT before the actor head, rather than calling
        # extract_features twice with different stop_gradient settings like the
        # non-recurrent forward()/evaluate_actions() do. Value-loss gradient
        # still flows unconditionally through latent -> RNN -> raw -> encoder;
        # actor-loss gradient is cut right here and therefore never reaches the
        # RNN or encoder either -- topologically identical to two separate
        # calls, computed once. Two calls would double the O(T) RNN unroll cost
        # for no behavioral difference, unlike the flat-MLP-encoder case where a
        # second call is cheap.
        actor_latent = latent.detach() if stop_gradient_actor else latent
        flat_actor_latent = actor_latent.reshape(num_steps * num_envs, -1)
        flat_actions = actions.reshape(num_steps * num_envs, -1)
        log_prob, entropy = self.actor.evaluate_action_log_prob(flat_actor_latent, flat_actions)
        values = self.value_net(latent.reshape(num_steps * num_envs, -1))
        return (
            values.reshape(num_steps, num_envs, -1),
            log_prob.reshape(num_steps, num_envs, -1),
            entropy.reshape(num_steps, num_envs, -1),
        )
