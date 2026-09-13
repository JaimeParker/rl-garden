"""BCQ policy: normalized obs + jointly-trained VAE + perturbation actor + twin-Q critic.

Verified against a full clone of ``sfujim/BCQ`` (``continuous_BCQ/BCQ.py``,
``continuous_BCQ/main.py`` read in full), not just fetched raw files. Box
or Dict (vision) observations via ``actor_extractor``/``critic_extractor``
(schema-driven, like every other policy in this codebase); either way at
least one ``state``/``state_<name>`` key is required (BCQ's D4RL MuJoCo
scope has none of the Dict/vision cases upstream, so this generalization is
rl-garden's own). Unlike ``SPOTPolicy``'s VAE (frozen after a one-time
pretraining phase), ``self.vae`` here trains jointly with the actor/critic
every gradient step -- it is a plain trainable submodule, never switched to
``eval()``/frozen by this class.

Every ``extract_*_features`` entry point normalizes the concatenated raw
``schema.state_keys`` entries (CORL's own ``normalize_states`` semantics)
before they reach the features extractor(s), not the extractor's output --
meaningful for a Dict+image schema too (only the state keys are normalized;
any ``rgb_<cam>``/``depth_<cam>`` key reaches the extractor untouched).

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
from rl_garden.observations import ObservationSchema
from rl_garden.observations.schema import ObservationContractError
from rl_garden.policies.base import BasePolicy, EncoderSharing

_N_CRITICS = 2


class BCQPolicy(ObsNormalizingMixin, BasePolicy):
    """VAE (jointly trained) + perturbation actor + twin-Q critic, all with target copies (VAE excepted)."""

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
        phi: float = 0.05,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
    ) -> None:
        assert isinstance(action_space, spaces.Box), "BCQ requires a Box action space."
        schema = ObservationSchema.from_space(observation_space)
        if not schema.has_state:
            raise ObservationContractError(
                "BCQPolicy requires a 'state' key in the observation space "
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

        actor_fd = self.actor_features_dim
        critic_fd = self.critic_features_dim
        net_arch = list(net_arch)

        self.vae = ConditionalVAE(
            actor_fd, action_space, hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim
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
        self.actor = PerturbationActor(actor_fd, action_space, **actor_kwargs)
        self.actor_target = PerturbationActor(actor_fd, action_space, **actor_kwargs)
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
        self.critic = EnsembleQCritic(critic_fd, action_space, **critic_kwargs)
        self.critic_target = EnsembleQCritic(critic_fd, action_space, **critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def _normalize_state_obs(self, obs: Obs) -> Obs:
        # Normalize every schema state key (``state``/``state_<name>``) as one
        # concatenated vector -- mean/std buffers are sized to the sum of all
        # state key dims (see __init__), sliced back per-key afterward so the
        # extractor still sees a per-key dict (CombinedExtractor/FlattenExtractor
        # concatenate the state keys themselves, in ``schema.state_keys`` order).
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
            actor_features = self.extract_actor_features(obs)
            critic_features = self.extract_critic_features(obs)
            batch_size = actor_features.shape[0]
            act_dim = int(self.action_space.shape[0])

            tiled_actor_features = actor_features.repeat_interleave(num_candidates, dim=0)
            tiled_critic_features = critic_features.repeat_interleave(num_candidates, dim=0)
            sampled_actions = self.vae.decode(tiled_actor_features, clip=0.5)
            perturbed_actions = self.actor(tiled_actor_features, sampled_actions)
            q1 = self.q_values(tiled_critic_features, perturbed_actions, target=False)[0]

            q1 = q1.reshape(batch_size, num_candidates)
            best_idx = q1.argmax(dim=1)
            perturbed_actions = perturbed_actions.reshape(batch_size, num_candidates, act_dim)
            return perturbed_actions[
                torch.arange(batch_size, device=actor_features.device), best_idx
            ]

    def vae_parameters(self):
        yield from self.vae.parameters()

    def actor_parameters(self):
        # actor_extractor trains via this optimizer only when it is genuinely
        # separate from critic_extractor (otherwise it trains via
        # critic_and_encoder_parameters' critic loss under "shared_critic_grad").
        if self.critic_extractor is not None and self.critic_extractor is not self.actor_extractor:
            yield from self.actor_extractor.parameters()
        yield from self.actor.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()
