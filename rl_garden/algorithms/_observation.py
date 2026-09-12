"""Shared observation-encoder resolution for algorithms.

This module is the Layer C entry point of the observation redesign (see
``.agents``/plan docs): it turns an algorithm's declarative
``encoder_config``/``obs_groups``/``critic_encoder_config`` attributes into
built encoder(s) via the Layer B factory
(``rl_garden.encoders.build_observation_encoder``), and exposes the actor/
critic encoder-sharing convention every algorithm needs
(``EncoderSharing``).

``ObservationEncoderMixin`` is meant to sit on ``BaseAlgorithm`` (see
``rl_garden/algorithms/base_algorithm.py``): every algorithm inherits the
``encoder_sharing`` class attribute and the ``_resolve_observation_encoders``/
``_actor_features``/``_critic_features`` helpers for free, at zero
constructor-signature cost. Concrete algorithms opt in by accepting
``encoder_config``/``obs_groups``/``critic_encoder_config`` constructor
kwargs themselves (see ``SAC`` for the reference implementation) -- these
are *not* threaded through ``BaseAlgorithm.__init__``/
``OffPolicyAlgorithm.__init__``/etc., matching the existing convention where
observation-related kwargs live on the concrete algorithm class, not the
shared training-loop base classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.observations import (
    ObsGroups,
    ObservationSchema,
    normalize_observation_space,
    resolve_obs_groups,
)

# "shared_critic_grad": one encoder; actor path is stop-gradiented, only the
#   critic loss trains it (off-policy default -- SAC/CQL/IQL's existing RGBD
#   convention).
# "shared": one encoder; both actor and critic losses train it (on-policy
#   default -- PPO).
# "separate": two encoders (actor's own + critic's own), each trained only
#   by its own loss. Required whenever obs_groups.actor != obs_groups.critic
#   or a distinct critic_encoder_config is given.
EncoderSharing = Literal["shared_critic_grad", "shared", "separate"]


@dataclass
class ObservationEncoders:
    """Resolved actor/critic feature extractors for one algorithm instance."""

    actor: BaseFeaturesExtractor
    critic: Optional[BaseFeaturesExtractor]
    schema: ObservationSchema
    sharing: EncoderSharing

    @property
    def critic_or_actor(self) -> BaseFeaturesExtractor:
        """The critic's own extractor, or the actor's when shared."""
        return self.critic if self.critic is not None else self.actor


def resolve_observation_encoders(
    observation_space,
    encoder_config: Optional[EncoderConfig],
    obs_groups: Optional[ObsGroups],
    encoder_sharing: EncoderSharing,
    *,
    critic_encoder_config: Optional[EncoderConfig] = None,
    augmentation_seed: Optional[int] = None,
) -> ObservationEncoders:
    """Build the actor (and, when ``encoder_sharing == "separate"``, critic)
    features extractor for ``observation_space``.

    Raises ``ValueError`` when ``obs_groups`` asks for asymmetric actor/critic
    keys, or ``critic_encoder_config`` is given, while ``encoder_sharing`` is
    not ``"separate"`` -- both require two independent encoder instances.
    """
    normalized_space = normalize_observation_space(observation_space)
    schema = ObservationSchema.from_space(normalized_space)
    resolved = resolve_obs_groups(schema, obs_groups)
    actor_schema = resolved["actor"]
    critic_schema = resolved["critic"]
    asymmetric = actor_schema.keys != critic_schema.keys

    if (asymmetric or critic_encoder_config is not None) and encoder_sharing != "separate":
        reason = (
            "obs_groups.actor != obs_groups.critic"
            if asymmetric
            else "a critic_encoder_config was given"
        )
        raise ValueError(
            f"{reason}, which requires two independent encoders; set "
            f"encoder_sharing='separate' (got {encoder_sharing!r})."
        )

    # Pass the ORIGINAL observation_space through: build_observation_encoder
    # re-normalizes internally. Every algorithm's env boundary normalizes a
    # bare Box into Dict unconditionally (BaseAlgorithm.__init__), so this is
    # always a Dict in practice; build_observation_encoder's own raw-Box
    # ("no dict wrapping at runtime") FlattenExtractor mode remains reachable
    # directly for unit tests that construct an extractor without going
    # through an algorithm at all.
    actor_encoder = build_observation_encoder(
        observation_space,
        encoder_config,
        schema=actor_schema,
        augmentation_seed=augmentation_seed,
    )
    critic_encoder: Optional[BaseFeaturesExtractor] = None
    if encoder_sharing == "separate":
        critic_cfg = critic_encoder_config if critic_encoder_config is not None else encoder_config
        critic_encoder = build_observation_encoder(
            observation_space,
            critic_cfg,
            schema=critic_schema,
            augmentation_seed=augmentation_seed,
        )

    return ObservationEncoders(
        actor=actor_encoder, critic=critic_encoder, schema=schema, sharing=encoder_sharing
    )


class ObservationEncoderMixin:
    """Gives an algorithm class the observation-encoder resolution helpers.

    Applied to ``BaseAlgorithm`` so every algorithm inherits it. Concrete
    algorithms set ``self.encoder_config``/``self.obs_groups``/
    ``self.critic_encoder_config`` (all optional; default ``None`` when
    unset) before calling ``self._resolve_observation_encoders(...)`` from
    their own ``_setup_model()``.
    """

    #: Overridable per algorithm class. See module docstring for the three
    #: values' meaning.
    encoder_sharing: EncoderSharing = "shared_critic_grad"

    #: Class-level defaults so a concrete algorithm that never sets one of
    #: these (e.g. a state-only algorithm with no obs_groups/critic-encoder
    #: notion) doesn't need its own `getattr(self, ..., None)` guard;
    #: algorithms that DO support them set the instance attribute in
    #: __init__, which shadows these per ordinary Python attribute lookup.
    encoder_config: Optional[EncoderConfig] = None
    obs_groups: Optional[ObsGroups] = None
    critic_encoder_config: Optional[EncoderConfig] = None

    observation_encoders: ObservationEncoders

    def _resolve_observation_encoders(
        self, observation_space, *, augmentation_seed: Optional[int] = None
    ) -> ObservationEncoders:
        self.observation_encoders = resolve_observation_encoders(
            observation_space,
            self.encoder_config,
            self.obs_groups,
            self.encoder_sharing,
            critic_encoder_config=self.critic_encoder_config,
            augmentation_seed=augmentation_seed,
        )
        return self.observation_encoders

    def _actor_features(self, obs, stop_gradient: Optional[bool] = None) -> torch.Tensor:
        """Actor-role features for ``obs``.

        ``stop_gradient`` defaults to the sharing rule: detached when
        ``encoder_sharing == "shared_critic_grad"`` (the encoder is trained
        only by the critic loss), passed straight through otherwise.
        """
        if stop_gradient is None:
            stop_gradient = self.encoder_sharing == "shared_critic_grad"
        return self.observation_encoders.actor.extract(obs, stop_gradient=stop_gradient)

    def _critic_features(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        """Critic-role features for ``obs`` (the critic's own encoder when
        ``encoder_sharing == "separate"``, else the shared actor encoder)."""
        return self.observation_encoders.critic_or_actor.extract(obs, stop_gradient=stop_gradient)
