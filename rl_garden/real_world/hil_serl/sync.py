"""HIL-SERL's sync classes -- currently identical to the base; kept as their
own subclasses for naming symmetry with ``rl_garden.real_world.serl``.
"""
from __future__ import annotations

from rl_garden.real_world.sync import ActorSyncClient, LearnerSyncServer


class HilSerlActorSyncClient(ActorSyncClient):
    pass


class HilSerlLearnerSyncServer(LearnerSyncServer):
    pass
