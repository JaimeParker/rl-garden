"""Asymmetric actor/critic observation groups (rsl_rl-style consumer groups)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.observations.schema import ObservationContractError, ObservationSchema

# Duplicated (not imported) from rl_garden.algorithms._observation.EncoderSharing
# to keep this module's import graph light -- same rationale as
# rl_garden.common.cli_args's copy: rl_garden.algorithms is a heavy package
# and rl_garden.observations sits well below it.
EncoderSharing = Literal["shared_critic_grad", "shared", "separate"]


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


def resolve_encoder_sharing(
    *,
    requested: Optional[EncoderSharing],
    class_default: EncoderSharing,
    asymmetric: bool,
    has_critic_encoder: bool,
) -> tuple[EncoderSharing, str]:
    """Resolve an algorithm's actual ``encoder_sharing`` plus a human-readable
    origin string, from an optional user request and the observation-contract
    facts that force two independent encoders.

    Used both by ``ObservationEncoderMixin`` (at algorithm-construction time)
    and by ``algorithm_registry``'s static CLI preflight, so a config never
    needs to spell out ``encoder_sharing: separate`` when it is already
    implied by asymmetric ``obs_groups`` or a distinct ``critic_encoder``.

    - ``requested is None`` and (``asymmetric`` or ``has_critic_encoder``):
      inferred ``"separate"``.
    - ``requested is None`` otherwise: ``class_default``.
    - ``requested == "separate"``: ``"separate"``, explicit.
    - ``requested`` non-``"separate"`` with ``asymmetric``/``has_critic_encoder``:
      ``ObservationContractError`` -- an explicit contradiction.
    - ``requested`` non-``"separate"`` otherwise: ``requested``, explicit.
    """
    forces_separate = asymmetric or has_critic_encoder
    if requested is None:
        if forces_separate:
            reason = "asymmetric obs_groups" if asymmetric else "critic_encoder"
            return "separate", f"inferred ({reason})"
        return class_default, "default"
    if requested == "separate":
        return "separate", "explicit"
    if forces_separate:
        reason = (
            "obs_groups.actor != obs_groups.critic"
            if asymmetric
            else "a critic_encoder_config was given"
        )
        raise ObservationContractError(
            f"{reason}, which requires two independent encoders; encoder_sharing "
            f"was explicitly set to {requested!r} (expected 'separate')."
        )
    return requested, "explicit"
