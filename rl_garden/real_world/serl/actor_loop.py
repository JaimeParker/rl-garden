"""SERL's ActorLoop -- currently identical to the base; kept as its own
subclass for naming symmetry with HIL-SERL's own (future) override.
"""
from __future__ import annotations

from rl_garden.real_world.actor_loop import ActorLoop


class SerlActorLoop(ActorLoop):
    pass
