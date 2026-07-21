"""Shared helper for building an optional second (critic/value) features
extractor, used by both ``SAC`` and ``PPO``. Not a base-class hook -- only
these two algorithms call it, so a free function is enough (see
docs/superpowers/specs/2026-07-21-asymmetric-actor-critic-encoders-design.md).
"""
from __future__ import annotations

from typing import Any, Optional

from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import ImageEncoderFactory


def build_secondary_extractor(
    *,
    full_observation_space: spaces.Space,
    features_extractor_class: type[BaseFeaturesExtractor],
    primary_kwargs: dict[str, Any],
    extra_obs_keys: tuple[str, ...],
    override_image_encoder_factory: Optional[ImageEncoderFactory],
) -> Optional[BaseFeaturesExtractor]:
    """Build the critic/value-side extractor, or ``None`` if unconfigured.

    ``None`` is returned when neither ``extra_obs_keys`` nor
    ``override_image_encoder_factory`` is set -- the caller's policy should
    then fall back to the shared (actor's) extractor, exactly as before this
    feature existed.

    The secondary extractor is always built from ``full_observation_space``
    (critic/value always sees everything the env provides); it is the
    caller's job to trim ``extra_obs_keys`` out of the *actor's* space
    instead (see ``drop_dict_keys``).
    """
    if not extra_obs_keys and override_image_encoder_factory is None:
        return None
    kwargs = dict(primary_kwargs)
    if override_image_encoder_factory is not None:
        kwargs["image_encoder_factory"] = override_image_encoder_factory
    return features_extractor_class(
        observation_space=full_observation_space,
        **kwargs,
    )
