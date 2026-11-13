"""SACPolicy variant with a FlowMatchingActor instead of a Gaussian actor.

Reuses SACPolicy's actor_extractor/critic/critic_target construction
unchanged; only the actor changes, through the ``_build_actor`` seam.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import FlowMatchingActor, KernelInit
from rl_garden.policies.sac_policy import SACPolicy


class SACFlowPolicy(SACPolicy):
    """SACPolicy with a flow-matching actor. Flat Box observations only --
    the caller (SACFlow._policy_kwargs) is responsible for rejecting
    ``structured_feature_config() is not None`` before constructing this.

    Every ``SACPolicy`` keyword is accepted; the Gaussian actor's options
    (``actor_use_layer_norm``, ``log_std_*``, ...) do not reach the flow
    actor, which reads only the ``flow_*`` / ``denoising_steps`` /
    ``noise_std`` keywords below."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        actor_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        *,
        flow_hidden_dims: Sequence[int] = (256, 256, 256),
        denoising_steps: int = 4,
        noise_std: float = 0.3,
        flow_use_layer_norm: bool = False,
        flow_kernel_init: Optional[KernelInit] = "xavier_uniform",
        **sac_policy_kwargs: Any,
    ) -> None:
        if sac_policy_kwargs.get("actor_factory") is not None:
            raise ValueError(
                "SACFlowPolicy builds its own FlowMatchingActor; actor_factory "
                "is not supported (pass the factory to SACPolicy / SAC instead)."
            )
        # Read by _build_actor, which SACPolicy.__init__ calls; a plain
        # attribute may be set before nn.Module.__init__.
        # flow_kernel_init is intentionally decoupled from the critic/trunk
        # kernel_init: it defaults to "xavier_uniform" to match RLinf's own
        # (unconditional) flow-actor weight init, independent of whatever
        # init the critic uses.
        self._flow_actor_options = {
            "hidden_dims": flow_hidden_dims,
            "denoising_steps": denoising_steps,
            "noise_std": noise_std,
            "use_layer_norm": flow_use_layer_norm,
            "kernel_init": flow_kernel_init,
        }
        super().__init__(
            observation_space, action_space, actor_extractor, net_arch, **sac_policy_kwargs
        )

    def _build_actor(
        self, features_dim: int, action_space: spaces.Box, **actor_options: Any
    ) -> nn.Module:
        del actor_options  # the Gaussian actor's options
        return FlowMatchingActor(features_dim, action_space, **self._flow_actor_options)
