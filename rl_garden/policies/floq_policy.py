"""FloQ policy: FQL's twin-Q critic + two flow-matching actor networks, plus
a flow-matching critic vector field over the scalar-return axis.

Extends ``FQLPolicy`` rather than duplicating it: the actor side (BC flow +
one-step distillation) and the plain scalar critic (used as the distillation
target for the actor's Q-loss) are unchanged FQL. The only addition is
``floq``/``floq_target`` (``CriticVectorField``), FloQ's flow-matching TD
critic (see ``rl_garden/algorithms/floq.py`` for the loss).

``FQLPolicy``'s own ``critic_target`` is deleted after ``super().__init__``:
nothing in FloQ updates it (FloQ's only Polyak target is ``floq_target``), so
keeping it around would silently drift out of sync with ``critic`` with no
training signal ever correcting it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, BackboneType, KernelInit
from rl_garden.networks.critic_vector_field import CriticVectorField
from rl_garden.policies.fql_policy import EncoderSharing, FQLPolicy


class FloQPolicy(FQLPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (512, 512, 512, 512),
        *,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = None,
        encoder_sharing: EncoderSharing = "shared",
        actor_bc_flow_encoder: Optional[BaseFeaturesExtractor] = None,
        actor_onestep_flow_encoder: Optional[BaseFeaturesExtractor] = None,
        flow_num_ensembles: int = 2,
        embed_time: bool = True,
        time_embed_dim: int = 64,
        use_prob_embed: bool = True,
        q_min: float,
        q_max: float,
        num_bins: int = 51,
        sigma: float = 16.0,
        critic_flow_net_arch: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
            encoder_sharing=encoder_sharing,
            actor_bc_flow_encoder=actor_bc_flow_encoder,
            actor_onestep_flow_encoder=actor_onestep_flow_encoder,
        )
        del self.critic_target

        fd = features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        critic_flow_hidden_dims = (
            list(critic_flow_net_arch) if critic_flow_net_arch is not None else list(net_arch)
        )

        floq_kwargs = dict(
            num_ensembles=flow_num_ensembles,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
            embed_time=embed_time,
            time_embed_dim=time_embed_dim,
            use_prob_embed=use_prob_embed,
            q_min=q_min,
            q_max=q_max,
            num_bins=num_bins,
            sigma=sigma,
        )
        self.floq = CriticVectorField(fd, action_dim, critic_flow_hidden_dims, **floq_kwargs)
        self.floq_target = CriticVectorField(
            fd, action_dim, critic_flow_hidden_dims, **floq_kwargs
        )
        self.floq_target.load_state_dict(self.floq.state_dict())
        for p in self.floq_target.parameters():
            p.requires_grad_(False)

    def critic_and_encoder_parameters(self):
        yield from super().critic_and_encoder_parameters()
        yield from self.floq.parameters()
