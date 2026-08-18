"""SACPolicy variant with a FlowMatchingActor instead of a Gaussian actor.

Reuses SACPolicy's features_extractor/critic/critic_target construction
unchanged; only the actor changes. SACPolicy.__init__ builds ``self.actor``
inline (no ``_build_actor()`` hook), so this subclass lets the parent build
its throwaway SquashedGaussianActor and immediately replaces it -- cheaper
than duplicating SACPolicy's critic-construction logic, and keeps
``sac_policy.py`` untouched.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, CriticImpl, FlowMatchingActor, KernelInit
from rl_garden.policies.sac_policy import SACPolicy


class SACFlowPolicy(SACPolicy):
    """SACPolicy with a flow-matching actor. Flat Box observations only --
    the caller (SACFlow._build_policy) is responsible for rejecting
    ``structured_feature_config() is not None`` before constructing this."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        *,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        critic_use_layer_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        critic_impl: CriticImpl = "vmap",
        flow_hidden_dims: Sequence[int] = (256, 256, 256),
        denoising_steps: int = 4,
        noise_std: float = 0.3,
        flow_use_layer_norm: bool = False,
        flow_kernel_init: Optional[KernelInit] = "xavier_uniform",
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            critic_use_layer_norm=critic_use_layer_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            critic_impl=critic_impl,
        )
        # Parent __init__ already built a throwaway SquashedGaussianActor on
        # self._actor_fd -- replace it rather than duplicating critic setup.
        # flow_kernel_init is intentionally decoupled from the critic/trunk
        # kernel_init above: it defaults to "xavier_uniform" to match RLinf's
        # own (unconditional) flow-actor weight init, independent of whatever
        # init the critic uses.
        self.actor = FlowMatchingActor(
            self._actor_fd,
            action_space,
            hidden_dims=flow_hidden_dims,
            denoising_steps=denoising_steps,
            noise_std=noise_std,
            use_layer_norm=flow_use_layer_norm,
            kernel_init=flow_kernel_init,
        )

    def actor_diagnostics(self, obs: Obs) -> dict[str, torch.Tensor]:
        """Diagnostic-only actor stats. FlowMatchingActor has no closed-form
        mean/log_std to unpack (unlike SquashedGaussianActor), so this reports
        the accumulated log_prob's own mean/std and the sampled action's norm
        instead of a Gaussian entropy decomposition."""
        with torch.no_grad():
            features = self.extract_features(obs, stop_gradient=True)
            actor_input = self._transform_features_for_actor(features)
            action, log_prob = self.actor.action_log_prob(actor_input)
            return {
                "log_prob_mean": log_prob.mean(),
                "log_prob_std": log_prob.std(),
                "action_norm": action.norm(dim=-1).mean(),
            }
