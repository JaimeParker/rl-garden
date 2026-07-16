"""Reusable observation projections supplied by environment backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gymnasium import spaces

from rl_garden.common.types import Obs


@dataclass(frozen=True)
class ObservationView:
    """Project an environment observation into an algorithm-facing space.

    ``key_map`` contains ``(target_key, source_key)`` pairs. An empty mapping
    is the identity view and supports both tensor and dict observations.
    """

    observation_space: spaces.Space
    key_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.key_map:
            return
        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("Mapped observation views require a Dict space.")
        target_keys = tuple(target for target, _ in self.key_map)
        expected_keys = tuple(self.observation_space.spaces)
        if set(target_keys) != set(expected_keys) or len(target_keys) != len(
            expected_keys
        ):
            raise ValueError(
                "Observation view targets must match its observation space: "
                f"targets={target_keys}, expected={expected_keys}."
            )

    def transform(self, obs: Obs) -> Obs:
        if not self.key_map:
            return obs
        if not isinstance(obs, dict):
            raise TypeError("Mapped observation views require Dict observations.")
        projected = {}
        for target, source in self.key_map:
            if source in obs:
                projected[target] = obs[source]
            elif target in obs:
                projected[target] = obs[target]
            else:
                raise KeyError(
                    f"Observation view source {source!r} is missing for {target!r}."
                )
        return projected


def resolve_agent_observation_view(env) -> ObservationView:
    """Return the environment's agent view, or its normal observation view."""

    view: Optional[ObservationView] = getattr(env, "agent_observation_view", None)
    if view is not None:
        return view
    return ObservationView(env.single_observation_space)
