"""AWAC offline-to-online, built on the shared shell.

AWAC's own literature supports online fine-tuning after offline pretraining
(unlike TD3-BC, which is pure-offline only). ``Off2OnAWAC`` adds no behavior
on top of ``_AWACRolloutTrainingShell`` beyond a construction-time preset:
``Off2OnReplayMixin``'s two hooks stay at their no-op defaults, matching
``Off2OnIQL`` (AWAC needs no algorithm-specific change at the offline->online
switch).
"""
from __future__ import annotations

from rl_garden.algorithms.awac import _AWACRolloutTrainingShell


class Off2OnAWAC(_AWACRolloutTrainingShell):
    """AWAC offline pretraining + online fine-tuning."""

    _compatible_checkpoint_algorithms = ("Off2OnAWAC", "AWAC")
