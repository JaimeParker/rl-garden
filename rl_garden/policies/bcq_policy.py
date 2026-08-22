"""BCQ policy: normalized obs + jointly-trained VAE + perturbation actor + twin-Q critic.

Verified against a full clone of ``sfujim/BCQ`` (``continuous_BCQ/BCQ.py``,
``continuous_BCQ/main.py`` read in full), not just fetched raw files. Box
observations only (matches BCQ's D4RL MuJoCo scope). Unlike
``SPOTPolicy``'s VAE (frozen after a one-time pretraining phase),
``self.vae`` here trains jointly with
the actor/critic every gradient step -- it is a plain trainable submodule,
never switched to ``eval()``/frozen by this class.

``n_critics`` is fixed at 2 (not configurable): BCQ's target computation
(``BCQCore.train``) needs exactly a ``(q1, q2)`` pair for its soft-clipped
double-Q mixture, not a generic N-way ensemble.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.obs_normalization import ObsNormalizingMixin
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import ConditionalVAE, EnsembleQCritic, KernelInit, PerturbationActor
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.base import BasePolicy

_N_CRITICS = 2


class BCQPolicy(ObsNormalizingMixin, BasePolicy):
    """VAE (jointly trained) + perturbation actor + twin-Q critic, all with target copies (VAE excepted)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (400, 300),
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        phi: float = 0.05,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, spaces.Box), (
            "BCQPolicy requires a Box observation space."
        )
        assert isinstance(action_space, spaces.Box), "BCQ requires a Box action space."

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self._register_obs_normalizer(int(observation_space.shape[0]))

        fd = features_extractor.features_dim
        net_arch = list(net_arch)

        self.vae = ConditionalVAE(
            fd, action_space, hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim
        )

        actor_kwargs = dict(
            hidden_dims=net_arch,
            phi=phi,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.actor = PerturbationActor(fd, action_space, **actor_kwargs)
        self.actor_target = PerturbationActor(fd, action_space, **actor_kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad_(False)

        critic_kwargs = dict(
            hidden_dims=net_arch,
            n_critics=_N_CRITICS,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic = EnsembleQCritic(fd, action_space, **critic_kwargs)
        self.critic_target = EnsembleQCritic(fd, action_space, **critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        obs = self._normalize_obs(obs)
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.critic_target if target else self.critic
        q1, q2 = net(features, actions)
        return q1, q2

    def predict(
        self, obs: Obs, deterministic: bool = False, num_candidates: int = 100
    ) -> torch.Tensor:
        del deterministic  # BCQ inference is always this same candidate-search.
        with torch.no_grad():
            features = self.extract_features(obs)
            batch_size = features.shape[0]
            act_dim = int(self.action_space.shape[0])

            tiled_features = features.repeat_interleave(num_candidates, dim=0)
            sampled_actions = self.vae.decode(tiled_features, clip=0.5)
            perturbed_actions = self.actor(tiled_features, sampled_actions)
            q1 = self.q_values(tiled_features, perturbed_actions, target=False)[0]

            q1 = q1.reshape(batch_size, num_candidates)
            best_idx = q1.argmax(dim=1)
            perturbed_actions = perturbed_actions.reshape(batch_size, num_candidates, act_dim)
            return perturbed_actions[
                torch.arange(batch_size, device=features.device), best_idx
            ]

    def vae_parameters(self):
        yield from self.vae.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()
