"""Observation composition layer: what an env/dataset observes, strictly typed.

See ``.agents/rules/training-development.md`` and the observation-redesign
plan for how this package fits into the encoder/algorithm layers.
"""
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.groups import ObsGroups, resolve_encoder_sharing, resolve_obs_groups
from rl_garden.observations.schema import (
    Modality,
    ObsEntry,
    ObservationContractError,
    ObservationSchema,
    key_modality,
    normalize_observation_space,
    validate_observation_space,
)

__all__ = [
    "ObservationConfig",
    "ObservationSchema",
    "ObsEntry",
    "Modality",
    "ObsGroups",
    "ObservationContractError",
    "normalize_observation_space",
    "validate_observation_space",
    "resolve_obs_groups",
    "resolve_encoder_sharing",
    "key_modality",
]
