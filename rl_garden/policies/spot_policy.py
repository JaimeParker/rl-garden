"""SPOT policy: TD3-BC's actor/critic plus a behavior-density VAE.

``SPOTPolicy`` extends ``TD3BCPolicy`` unmodified (actor/actor_target,
critic/critic_target, ``extract_features``, ``q_values_all``) and adds
``self.vae`` (a ``BehaviorVAE``), used by SPOT's actor loss as a "support
constraint" in place of TD3-BC's BC-MSE term.
"""
from __future__ import annotations

from typing import Optional, Sequence

from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BehaviorVAE, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.td3_bc_policy import TD3BCPolicy


class SPOTPolicy(TD3BCPolicy):
    """``TD3BCPolicy`` plus a frozen-after-pretraining behavior-density VAE."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
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
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.vae = BehaviorVAE(
            self.features_extractor.features_dim,
            action_space,
            hidden_dim=vae_hidden_dim,
            latent_dim=vae_latent_dim,
        )

    def vae_parameters(self):
        yield from self.vae.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        # The VAE is pretrained once, then frozen (no Dropout/LayerNorm in
        # its trunk today, so this is currently a no-op) -- force eval()
        # unconditionally so a future trunk change can't silently reintroduce
        # train-mode-only behavior (e.g. Dropout) into a network that is no
        # longer being optimized.
        self.vae.eval()
        return self
