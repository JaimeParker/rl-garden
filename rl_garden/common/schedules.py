"""Exploration-noise and hyper-parameter schedules.

Ported from DrQ-v2's ``utils.py:129-149``.  Supports constant floats and
linear / step-linear schedules expressed as strings.
"""
from __future__ import annotations

import re

import numpy as np


def schedule(schdl: str | float, step: int) -> float:
    """Evaluate a schedule string or constant at a given training step.

    Supported formats
    -----------------
    * ``float`` – returned as-is.
    * ``"linear(init, final, duration)"`` – linear interpolation from *init*
      to *final* over *duration* steps; clamped at boundaries.
    * ``"step_linear(init, final1, dur1, final2, dur2)"`` – two-stage linear:
      *init* → *final1* over *dur1*, then *final1* → *final2* over *dur2*.
    """
    try:
        return float(schdl)
    except ValueError:
        pass

    match = re.match(r"linear\((.+),(.+),(.+)\)", str(schdl))
    if match:
        init, final, duration = [float(g) for g in match.groups()]
        mix = np.clip(step / duration, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final

    match = re.match(
        r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", str(schdl)
    )
    if match:
        init, final1, duration1, final2, duration2 = [
            float(g) for g in match.groups()
        ]
        if step <= duration1:
            mix = np.clip(step / duration1, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final1
        else:
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return (1.0 - mix) * final1 + mix * final2

    raise NotImplementedError(f"Unsupported schedule: {schdl!r}")
