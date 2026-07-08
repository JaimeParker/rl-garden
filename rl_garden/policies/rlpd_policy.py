"""RLPD policy: SACPolicy with an optional pnorm ablation on actor/critic.

Mirrors ``ResidualSACPolicy``'s pattern (build via ``super().__init__()``,
then replace the modules that need algorithm-specific construction) rather
than adding ``use_pnorm`` params to ``SACPolicy.__init__`` itself --
``use_pnorm`` is an RLPD-specific ablation knob with no other caller, unlike
``dropout_rate``/``kernel_init``/``backbone_type`` which were already
generic ``SACPolicy`` capabilities before RLPD existed.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, CriticImpl, KernelInit
from rl_garden.networks.actor_critic import (
    EnsembleQCritic as ContinuousCritic,
    SquashedGaussianActor as Actor,
    get_actor_critic_arch,
)
from rl_garden.policies.sac_policy import SACPolicy


class RLPDPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        critic_impl: CriticImpl = "vmap",
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        use_pnorm: bool = False,
        log_std_min: float = -5.0,
        log_std_mode: Literal["clamp", "tanh"] = "clamp",
        actor_feature_dim: Optional[int] = None,
        critic_spatial_emb_dim: int = 1024,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            critic_impl=critic_impl,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            log_std_min=log_std_min,
            log_std_mode=log_std_mode,
            actor_feature_dim=actor_feature_dim,
            critic_spatial_emb_dim=critic_spatial_emb_dim,
        )
        if not use_pnorm:
            return

        sc = features_extractor.structured_feature_config()
        if sc is not None and sc.get("layout") == "token_and_prop":
            raise ValueError(
                "use_pnorm=True is not supported with a 'token_and_prop' "
                "structured features extractor: that obs layout routes "
                "through SpatialEmbQEnsemble, which EnsembleQCritic's "
                "use_pnorm does not reach."
            )

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        fd = features_extractor.features_dim
        critic_kwargs = dict(
            hidden_dims=critic_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            use_pnorm=True,
            critic_impl=critic_impl,
        )
        self.critic = ContinuousCritic(fd, action_space, **critic_kwargs)
        self.critic_target = ContinuousCritic(fd, action_space, **critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor = Actor(
            self._actor_fd,
            action_space,
            hidden_dims=actor_arch,
            use_layer_norm=actor_use_layer_norm,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            use_pnorm=True,
            log_std_mode=log_std_mode,
            log_std_min=log_std_min,
        )
