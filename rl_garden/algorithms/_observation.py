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
``_policy_extractor_kwargs`` helpers for free, at zero constructor-signature
cost. ``_policy_extractor_kwargs`` returns the
``{"actor_extractor", "critic_extractor", "encoder_sharing"}`` kwargs an
algorithm's ``_setup_model`` passes straight into its ``BasePolicy``
subclass (see ``rl_garden.policies.base.BasePolicy`` for the actor/critic
extractor contract and stop-gradient rule those feed). Concrete algorithms
opt in by accepting ``encoder_config``/``obs_groups``/``critic_encoder_config``
constructor kwargs themselves (see ``SAC`` for the reference implementation)
-- these are *not* threaded through ``BaseAlgorithm.__init__``/
``OffPolicyAlgorithm.__init__``/etc., matching the existing convention where
observation-related kwargs live on the concrete algorithm class, not the
shared training-loop base classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.observations import (
    ObsGroups,
    ObservationContractError,
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

    Raises ``ObservationContractError`` (a ``ValueError`` subclass) when
    ``obs_groups`` asks for asymmetric actor/critic keys, or
    ``critic_encoder_config`` is given, while ``encoder_sharing`` is not
    ``"separate"`` -- both require two independent encoder instances.
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
        raise ObservationContractError(
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

    #: Optional per algorithm; algorithms that accept a ``policy_kwargs``
    #: constructor dict set this (via an algorithm's own
    #: ``_normalize_policy_kwargs``) before calling
    #: ``_policy_extractor_kwargs``.
    policy_kwargs: dict

    def _policy_extractor_kwargs(
        self, observation_space, *, augmentation_seed: Optional[int] = None
    ) -> dict[str, Any]:
        """Resolve ``{"actor_extractor", "critic_extractor",
        "encoder_sharing"}`` for building this algorithm's policy.

        Defaults to the schema-driven encoder(s) from
        ``_resolve_observation_encoders`` (lazily -- not resolved at all when
        both roles are overridden below, so building schema-driven encoders
        just to discard them never wastes compute or shifts RNG consumption).
        ``self.policy_kwargs`` may override either role directly with
        ``"actor_extractor_class"``/``"actor_extractor_kwargs"`` and
        ``"critic_extractor_class"``/``"critic_extractor_kwargs"`` -- an
        explicit ``*_class`` skips resolving the schema-driven encoder for
        that role entirely; a ``*_kwargs`` given without its ``*_class``
        raises (it would be silently ignored).
        """
        policy_kwargs = getattr(self, "policy_kwargs", None) or {}
        actor_class = policy_kwargs.get("actor_extractor_class")
        critic_class = policy_kwargs.get("critic_extractor_class")

        def _build(cls_: type, kwargs_key: str, class_key: str) -> BaseFeaturesExtractor:
            if not isinstance(cls_, type) or not issubclass(cls_, BaseFeaturesExtractor):
                raise TypeError(
                    f"policy_kwargs[{class_key!r}] must be a BaseFeaturesExtractor subclass."
                )
            extra = policy_kwargs.get(kwargs_key) or {}
            return cls_(observation_space=observation_space, **extra)

        if actor_class is None and policy_kwargs.get("actor_extractor_kwargs"):
            raise ValueError(
                "policy_kwargs['actor_extractor_kwargs'] was given without "
                "policy_kwargs['actor_extractor_class']; it would be silently "
                "ignored (the default extractor takes no such kwargs)."
            )
        if critic_class is None and policy_kwargs.get("critic_extractor_kwargs"):
            raise ValueError(
                "policy_kwargs['critic_extractor_kwargs'] was given without "
                "policy_kwargs['critic_extractor_class']; it would be silently "
                "ignored (the default extractor takes no such kwargs)."
            )

        need_resolved = actor_class is None or (
            self.encoder_sharing == "separate" and critic_class is None
        )
        if need_resolved:
            self._resolve_observation_encoders(
                observation_space, augmentation_seed=augmentation_seed
            )

        actor_extractor = (
            _build(actor_class, "actor_extractor_kwargs", "actor_extractor_class")
            if actor_class is not None
            else self.observation_encoders.actor
        )
        critic_extractor: Optional[BaseFeaturesExtractor] = None
        if critic_class is not None:
            critic_extractor = _build(
                critic_class, "critic_extractor_kwargs", "critic_extractor_class"
            )
        elif self.encoder_sharing == "separate":
            critic_extractor = self.observation_encoders.critic

        return {
            "actor_extractor": actor_extractor,
            "critic_extractor": critic_extractor,
            "encoder_sharing": self.encoder_sharing,
        }
