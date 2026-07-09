"""SERL's LearnerLoop -- no offline-data refresh needed (SERL doesn't use a
growing on-disk demo dataset), so this leaves ``_refresh_offline_data()`` as
the base class's no-op default.
"""
from __future__ import annotations

from rl_garden.real_world.learner_loop import LearnerLoop


class SerlLearnerLoop(LearnerLoop):
    pass
