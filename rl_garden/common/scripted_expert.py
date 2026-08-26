"""Expert-query interface for DAgger (``rl_garden.algorithms.dagger.DAgger``).

Deliberately minimal: a scripted/oracle expert is task- and env-specific
(e.g. a privileged-state IK controller), so providing one is the caller's
responsibility -- same as how expert data is supplied to ``BC``/``DiffusionBC``
today via a dataset file rather than a built-in oracle. This module only
defines the interface DAgger queries against.
"""
from __future__ import annotations

from typing import Protocol

import torch

from rl_garden.common.types import Obs


class ScriptedExpert(Protocol):
    """Callable expert: batched ``obs`` -> batched action, same convention as
    ``BasePolicy.predict()`` (leading batch dim matches ``env.num_envs``)."""

    def __call__(self, obs: Obs) -> torch.Tensor: ...
