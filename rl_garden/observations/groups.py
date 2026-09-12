"""Asymmetric actor/critic observation groups (rsl_rl-style consumer groups)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.observations.schema import ObservationSchema


@dataclass(frozen=True)
class ObsGroups:
    """Which observation keys the actor / critic each consume.

    ``None`` for a field means "all keys in the schema".
    """

    actor: Optional[tuple[str, ...]] = None
    critic: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.actor is not None:
            object.__setattr__(self, "actor", tuple(self.actor))
        if self.critic is not None:
            object.__setattr__(self, "critic", tuple(self.critic))

    @property
    def is_symmetric(self) -> bool:
        return self.actor == self.critic


def resolve_obs_groups(
    schema: ObservationSchema, groups: Optional[ObsGroups]
) -> dict[str, ObservationSchema]:
    """Resolve ``groups`` against ``schema`` into concrete per-consumer schemas.

    An unknown key in ``groups.actor``/``groups.critic`` raises
    ``ObservationContractError`` (via ``ObservationSchema.subset``).
    """
    if groups is None:
        return {"actor": schema, "critic": schema}
    actor_keys = groups.actor if groups.actor is not None else schema.keys
    critic_keys = groups.critic if groups.critic is not None else schema.keys
    return {
        "actor": schema.subset(actor_keys),
        "critic": schema.subset(critic_keys),
    }
