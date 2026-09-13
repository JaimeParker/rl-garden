"""One entry point turning any rl-garden observation space into a features
extractor: ``build_observation_encoder``.

See the observation-redesign plan's Layer B section. ``EncoderConfig`` says
*how* to encode; the observation space (normalized to a schema) says *what*
is being encoded.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import CombinedExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.observations import (
    ObservationContractError,
    ObservationSchema,
    normalize_observation_space,
)

# Every EncoderConfig field except this one only means something when the
# observation space actually has an image key; see
# _reject_image_config_without_images below.
_STATE_ONLY_ENCODER_FIELDS = "normalize_obs"


def _reject_image_config_without_images(encoder_config: EncoderConfig) -> None:
    """Raise unless every image-related ``EncoderConfig`` field is still at
    its default -- called when the FULL (unsubset) observation space has no
    image key at all. Keyed off the full space, not a per-consumer
    ``schema``/``keys`` subset: ``resolve_observation_encoders`` calls this
    function once per actor/critic consumer, and an asymmetric split (e.g.
    an image-observing actor with a state-only critic) legitimately passes a
    non-default, image-configured ``encoder_config`` for the critic's own
    (state-only) call too -- that must not raise.
    """
    default = EncoderConfig()
    mismatched = sorted(
        field.name
        for field in dataclasses.fields(EncoderConfig)
        if field.name != _STATE_ONLY_ENCODER_FIELDS
        and getattr(encoder_config, field.name) != getattr(default, field.name)
    )
    if mismatched:
        raise ObservationContractError(
            f"encoder_config sets image-related field(s) {mismatched!r}, but "
            "the observation space has no image ('rgb_<cam>'/'depth_<cam>') "
            "key; only 'normalize_obs' is meaningful for a state-only "
            "observation space."
        )


def build_observation_encoder(
    observation_space: "spaces.Box | spaces.Dict",
    encoder_config: Optional[EncoderConfig] = None,
    *,
    schema: Optional[ObservationSchema] = None,
    keys: Optional[Iterable[str]] = None,
    augmentation_seed: Optional[int] = None,
) -> BaseFeaturesExtractor:
    """Build the features extractor for ``observation_space``.

    ``observation_space`` may be a bare ``Box`` (treated as
    ``Dict({"state": Box})``) or a ``Dict`` following the strict
    ``rl_garden.observations`` vocabulary. ``schema`` defaults to
    ``ObservationSchema.from_space(normalized_space)``; ``keys`` (if given)
    subsets it. A state-only schema builds a ``FlattenExtractor`` over all of
    ``schema.state_keys`` (``"state"`` plus any ``state_<name>`` keys);
    otherwise a ``CombinedExtractor`` using ``encoder_config`` (defaults to
    ``EncoderConfig()``).

    Raises ``ObservationContractError`` if ``encoder_config`` sets any
    image-related field (everything except ``normalize_obs``) while
    ``observation_space`` has no image key at all.
    """
    normalized_space = normalize_observation_space(observation_space)
    full_schema = ObservationSchema.from_space(normalized_space)
    if encoder_config is not None and not full_schema.has_images:
        _reject_image_config_without_images(encoder_config)

    if schema is None:
        schema = full_schema
    if keys is not None:
        schema = schema.subset(tuple(keys))

    if not schema.has_images:
        # A bare Box caller gets raw tensors at runtime (no dict wrapping
        # happens to the actual env/dataset observations), so the extractor
        # must stay in Box mode for it -- only a genuine Dict input (already
        # ``{"state": Box}`` or normalized from a richer space via `keys`)
        # gets the dict-unwrapping FlattenExtractor.
        flatten_space = (
            observation_space
            if isinstance(observation_space, spaces.Box)
            else spaces.Dict({k: normalized_space.spaces[k] for k in schema.state_keys})
        )
        return FlattenExtractor(
            flatten_space,
            normalize_obs=(encoder_config.normalize_obs if encoder_config is not None else False),
            state_keys=(
                schema.state_keys if isinstance(flatten_space, spaces.Dict) else None
            ),
        )

    return CombinedExtractor(
        normalized_space,
        schema,
        encoder_config if encoder_config is not None else EncoderConfig(),
        augmentation_seed=augmentation_seed,
    )
