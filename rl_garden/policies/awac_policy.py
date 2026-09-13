"""AWAC policy: normalized obs + unsquashed Gaussian actor + twin-Q critic.

State observations only (no images; matches CORL's D4RL MuJoCo scope) --
``state_<name>`` keys and an asymmetric actor/critic (``obs_groups``/
``critic_encoder_config``/``encoder_sharing="separate"``) are supported. No
actor target -- AWAC's critic backup samples ``next_action`` from the
current actor, not a target actor (see ``rl_garden.algorithms.awac`` for
why).
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.obs_normalization import ObsNormalizingMixin
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import EnsembleQCritic, KernelInit, UnsquashedGaussianActor
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.base import BasePolicy, EncoderSharing


class AWACPolicy(ObsNormalizingMixin, BasePolicy):
    """Unsquashed Gaussian actor (no target) + twin-Q critic (with target).

    Registers a second, ``suffix="critic"`` obs-normalizer buffer pair only
    when ``critic_extractor`` is a genuinely separate object from
    ``actor_extractor`` (``encoder_sharing="separate"``) -- its output dim can
    then differ from the actor extractor's. The default (shared) case keeps
    exactly one buffer pair, applied to both roles, unchanged from before this
    policy had two extractors.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        actor_extractor: BaseFeaturesExtractor,
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        encoder_sharing: EncoderSharing = "shared_critic_grad",
        net_arch: Sequence[int] = (256, 256, 256),
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
        std_parameterization: Literal["exp", "uniform"] = "exp",
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        assert isinstance(action_space, spaces.Box), "AWAC requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")

        self._register_obs_normalizer(int(self.actor_features_dim))
        if self.critic_extractor is not None:
            self._register_obs_normalizer(int(self.critic_features_dim), suffix="critic")

        fd = self.actor_features_dim
        critic_fd = self.critic_features_dim
        net_arch = list(net_arch)

        self.actor = UnsquashedGaussianActor(
            fd,
            action_space,
            hidden_dims=net_arch,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
        )

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

    def extract_actor_features(self, obs: Obs) -> torch.Tensor:
        features = super().extract_actor_features(obs)
        return self._normalize_obs(features)

    def extract_critic_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        features = super().extract_critic_features(obs, stop_gradient=stop_gradient)
        if self.critic_extractor is None:
            return self._normalize_obs(features)
        return self._normalize_obs(features, suffix="critic")

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_actor_features(obs)
        if deterministic:
            return self.actor.deterministic_action(features)
        action, _ = self.actor.action_log_prob(features)
        return action

    def behavior_log_prob(
        self, obs: Obs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_actor_features(obs)
        return self.actor.evaluate_action_log_prob(
            features, actions
        ), self.actor.deterministic_action(features)

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
        # actor_extractor is actor-exclusive (nothing else would train it)
        # only when it is genuinely a separate object from critic_extractor;
        # matches SACPolicy.actor_parameters' identical convention.
        if self.critic_extractor is not None and self.critic_extractor is not self.actor_extractor:
            yield from self.actor_extractor.parameters()
        yield from self.actor.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()
