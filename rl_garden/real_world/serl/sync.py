"""SERL's sync classes -- currently identical to the base; kept as their own
subclasses for naming symmetry with HIL-SERL's own (future) override.
"""
from __future__ import annotations

from rl_garden.real_world.sync import ActorSyncClient, LearnerSyncServer


class SerlActorSyncClient(ActorSyncClient):
    pass


class SerlLearnerSyncServer(LearnerSyncServer):
    pass
