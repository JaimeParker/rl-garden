"""PLAS policy: normalized obs + a frozen-after-pretraining VAE + latent-space actor + twin-Q critic.

Verified against a full clone of ``Wenxuan-Zhou/PLAS`` (``algos.py``,
``main.py`` read in full), not just fetched raw files. Box or Dict (vision)
observations via ``actor_extractor``/``critic_extractor`` (schema-driven,
like every other policy in this codebase); either way at least one
``state``/``state_<name>`` key is required (PLAS's D4RL MuJoCo scope has
none of the Dict/vision cases upstream, so this generalization is
rl-garden's own). ``self.vae`` is pretrained once (``PLASCore.pretrain_vae()``,
mirroring ``SPOTPolicy``'s pattern) then frozen -- unlike ``BCQPolicy``'s
jointly trained VAE.

``use_perturbation`` (default ``False``) adds the "-P" variant: a
second-stage ``PerturbationActor`` applied to the VAE's decoded action,
reusing the exact class BCQ's perturbation actor uses (confirmed against
the cloned upstream source: PLAS-P's second stage is BCQ's ``Actor`` class
reused inline as ``ActorPerturbation``'s ``l4-l6``). The default-off
behavior matches upstream's own default -- ``main.py``'s ``--algo_name``
defaults to ``"Latent"``, not ``"LatentPerturbation"`` -- not the CLI's
separate ``--phi 0.`` default, which is only a safety net for running -P
without an explicit ``--phi`` (see ``rl_garden.algorithms.plas``'s module
docstring for the full mechanism). Both the latent actor and the
perturbation net (when present) are trained and target-updated together as
one logical "actor" -- two separate ``nn.Module`` instances sharing one
optimizer and one set of polyak updates, rather than upstream's single
bundled ``ActorPerturbation`` module; confirmed functionally identical
(disjoint, untied parameters; elementwise-independent polyak update; single
scalar actor loss backpropagated through both stages via one optimizer) and
avoids a bespoke two-stage network class.

``n_critics`` is fixed at 2 (not configurable), same reasoning as ``BCQPolicy``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.obs_normalization import ObsNormalizingMixin
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    ConditionalVAE,
    EnsembleQCritic,
    KernelInit,
    LatentActor,
    PerturbationActor,
)
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.observations import ObservationSchema
from rl_garden.observations.schema import ObservationContractError
from rl_garden.policies.base import BasePolicy, EncoderSharing

_N_CRITICS = 2


class PLASPolicy(ObsNormalizingMixin, BasePolicy):
    """Pretrained-frozen VAE + latent-space actor (+ optional perturbation) + twin-Q critic."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        *,
        actor_extractor: BaseFeaturesExtractor,
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        encoder_sharing: EncoderSharing = "shared_critic_grad",
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
        max_latent_action: float = 2.0,
        use_perturbation: bool = False,
        phi: float = 0.05,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
    ) -> None:
        assert isinstance(action_space, spaces.Box), "PLAS requires a Box action space."
        schema = ObservationSchema.from_space(observation_space)
        if not schema.has_state:
            raise ObservationContractError(
                "PLASPolicy requires a 'state' key in the observation space "
                "-- CORL's normalize_states semantics normalize the raw "
                "state entry, not the post-encoder features."
            )
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        self._state_keys = schema.state_keys
        state_dim = sum(int(observation_space.spaces[k].shape[0]) for k in self._state_keys)
        self._register_obs_normalizer(state_dim)
        self.use_perturbation = use_perturbation

        actor_fd = self.actor_features_dim
        critic_fd = self.critic_features_dim
        net_arch = list(net_arch)

        self.vae = ConditionalVAE(
            actor_fd, action_space, hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim
        )

        latent_kwargs = dict(
            hidden_dims=net_arch,
            max_latent_action=max_latent_action,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.latent_actor = LatentActor(actor_fd, self.vae.latent_dim, **latent_kwargs)
        self.latent_actor_target = LatentActor(actor_fd, self.vae.latent_dim, **latent_kwargs)
        self.latent_actor_target.load_state_dict(self.latent_actor.state_dict())
        for p in self.latent_actor_target.parameters():
            p.requires_grad_(False)

        self.perturbation = None
        self.perturbation_target = None
        if use_perturbation:
            perturbation_kwargs = dict(
                hidden_dims=net_arch,
                phi=phi,
                use_layer_norm=actor_use_layer_norm,
                use_group_norm=actor_use_group_norm,
                num_groups=num_groups,
                dropout_rate=actor_dropout_rate,
                kernel_init=kernel_init,
                backbone_type=backbone_type,
            )
            self.perturbation = PerturbationActor(actor_fd, action_space, **perturbation_kwargs)
            self.perturbation_target = PerturbationActor(
                actor_fd, action_space, **perturbation_kwargs
            )
            self.perturbation_target.load_state_dict(self.perturbation.state_dict())
            for p in self.perturbation_target.parameters():
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
        self.critic = EnsembleQCritic(critic_fd, action_space, **critic_kwargs)
        self.critic_target = EnsembleQCritic(critic_fd, action_space, **critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def _normalize_state_obs(self, obs: Obs) -> Obs:
        # See BCQPolicy._normalize_state_obs -- identical pattern.
        normalized = dict(obs)
        raw = torch.cat([obs[k] for k in self._state_keys], dim=-1)
        normalized_concat = self._normalize_obs(raw)
        offset = 0
        for k in self._state_keys:
            dim = obs[k].shape[-1]
            normalized[k] = normalized_concat[..., offset : offset + dim]
            offset += dim
        return normalized

    def extract_actor_features(self, obs: Obs) -> torch.Tensor:
        return super().extract_actor_features(self._normalize_state_obs(obs))

    def extract_critic_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return super().extract_critic_features(
            self._normalize_state_obs(obs), stop_gradient=stop_gradient
        )

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Raw actor-extractor escape hatch (caller picks ``stop_gradient``),
        used by ``predict``/diagnostics/tests -- not the actor-role training
        path, which goes through ``extract_actor_features``."""
        return self.actor_extractor.extract(
            self._normalize_state_obs(obs), stop_gradient=stop_gradient
        )

    def action_from_latent(
        self, features: torch.Tensor, latent: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        decoded = self.vae.decode(features, z=latent)
        if not self.use_perturbation:
            return decoded
        net = self.perturbation_target if target else self.perturbation
        return net(features, decoded)

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.critic_target if target else self.critic
        q1, q2 = net(features, actions)
        return q1, q2

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic  # PLAS inference is always this same deterministic pass.
        features = self.extract_actor_features(obs)
        latent = self.latent_actor.deterministic_action(features)
        return self.action_from_latent(features, latent, target=False)

    def vae_parameters(self):
        yield from self.vae.parameters()

    def actor_parameters(self):
        # actor_extractor trains via this optimizer only when it is genuinely
        # separate from critic_extractor (otherwise it trains via
        # critic_and_encoder_parameters' critic loss under "shared_critic_grad").
        if self.critic_extractor is not None and self.critic_extractor is not self.actor_extractor:
            yield from self.actor_extractor.parameters()
        yield from self.latent_actor.parameters()
        if self.use_perturbation:
            yield from self.perturbation.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        # VAE is pretrained once, then frozen -- force eval() unconditionally,
        # mirroring SPOTPolicy's identical guard.
        self.vae.eval()
        return self
