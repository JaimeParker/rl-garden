"""TD3-BC policy: normalized obs + deterministic tanh actor + twin-Q critic.

Every ``extract_*_features`` entry point normalizes the RAW state entries
(CORL's own ``normalize_states`` semantics) before they reach the features
extractor, not the extractor's output -- meaningful for a Dict+image schema
too. All of ``schema.state_keys`` (``"state"`` plus any ``state_<name>``
keys, in that order -- see ``ObservationSchema``) are concatenated into one
vector, normalized together, then split back into their original per-key
shapes so the extractor's own per-key concatenation (``FlattenExtractor``/
``CombinedExtractor``'s proprio branch) sees the same values in the same
layout; any ``rgb_<cam>``/``depth_<cam>`` key reaches the extractor
untouched. Requires at least one state key in the observation schema (see
``rl_garden.common.obs_normalization``).
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.obs_normalization import ObsNormalizingMixin
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import DeterministicTanhActor, EnsembleQCritic, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.observations import ObservationSchema
from rl_garden.observations.schema import ObservationContractError
from rl_garden.policies.base import BasePolicy, EncoderSharing


class TD3BCPolicy(ObsNormalizingMixin, BasePolicy):
    """Deterministic actor + twin-Q critic, both with target copies."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        *,
        actor_extractor: BaseFeaturesExtractor,
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        encoder_sharing: EncoderSharing = "shared_critic_grad",
        net_arch: Sequence[int] = (256, 256),
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        assert isinstance(action_space, spaces.Box), "TD3-BC requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")
        schema = ObservationSchema.from_space(observation_space)
        if not schema.has_state:
            raise ObservationContractError(
                "TD3BCPolicy requires at least one 'state'/'state_<name>' key "
                "in the observation space -- CORL's normalize_states "
                "semantics normalize the raw state entries, not the "
                "post-encoder features."
            )
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        self._state_keys = schema.state_keys
        state_dim = sum(
            int(np.prod(observation_space.spaces[key].shape)) for key in self._state_keys
        )
        self._register_obs_normalizer(state_dim)

        actor_fd = self.actor_features_dim
        critic_fd = self.critic_features_dim
        net_arch = list(net_arch)

        self.actor = DeterministicTanhActor(
            actor_fd,
            action_space,
            hidden_dims=net_arch,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.actor_target = DeterministicTanhActor(
            actor_fd,
            action_space,
            hidden_dims=net_arch,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad_(False)

        self.critic = EnsembleQCritic(
            critic_fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic_target = EnsembleQCritic(
            critic_fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def _normalize_state(self, obs: Obs) -> Obs:
        normalized_obs = dict(obs)
        flat = torch.cat([obs[key] for key in self._state_keys], dim=-1)
        normalized_flat = self._normalize_obs(flat)
        offset = 0
        for key in self._state_keys:
            size = obs[key].shape[-1]
            normalized_obs[key] = normalized_flat[..., offset : offset + size]
            offset += size
        return normalized_obs

    def extract_actor_features(self, obs: Obs) -> torch.Tensor:
        return super().extract_actor_features(self._normalize_state(obs))

    def extract_critic_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return super().extract_critic_features(
            self._normalize_state(obs), stop_gradient=stop_gradient
        )

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Raw, caller-picks-``stop_gradient`` escape hatch over the actor
        extractor -- e.g. SPOT's VAE pretraining, a frozen-encoder phase
        unrelated to the ``encoder_sharing`` rule (which lives in
        ``extract_actor_features``)."""
        return self.actor_extractor.extract(
            self._normalize_state(obs), stop_gradient=stop_gradient
        )

    def critic_features_for(
        self, obs: Obs, actor_features: torch.Tensor, stop_gradient: bool = True
    ) -> torch.Tensor:
        """Critic-role features for ``obs``, for callers that already hold
        actor-role features for the same ``obs`` (the actor loss's
        Q(s, pi(s)) term). Reuses ``actor_features`` when the encoder is
        shared; re-extracts through the critic's own encoder only when it's
        genuinely separate. Available for direct use (tests, other
        callers); ``TD3BC``'s own ``train()`` instead reuses its own
        already-computed critic-role features tensor directly (valid
        regardless of sharing mode, since it's the same obs), rather than
        routing through ``actor_features`` here."""
        if self.critic_extractor is None:
            return actor_features
        return self.extract_critic_features(obs, stop_gradient=stop_gradient)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic  # TD3-BC inference is always deterministic.
        features = self.extract_actor_features(obs)
        return self.actor.deterministic_action(features)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def min_q_value(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = True
    ) -> torch.Tensor:
        return self.q_values_all(features, actions, target=target).min(dim=0).values

    def actor_parameters(self):
        # actor_extractor is trained via critic_and_encoder_parameters' Q-loss
        # in the shared case; when critic_extractor is genuinely separate,
        # nothing else would ever train actor_extractor, so it belongs here.
        if self.critic_extractor is not None:
            yield from self.actor_extractor.parameters()
        yield from self.actor.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()
