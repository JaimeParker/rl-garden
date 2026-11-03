"""ACT base-action providers and vendored ACT reference code.

The original ACT implementation is copied under this package from
``3rd_party/residual-rl/policies/act``.  The public API here wraps that code in
the base-action-provider contract used by ``ResidualSAC``.

The provider (and with it the vendored model code, which needs ``torchvision``)
is imported on first attribute access, so torch-only submodules such as
``rl_garden.models.act.trajectory_dataset`` import without it.
"""

from __future__ import annotations

import importlib
from typing import Any

_PROVIDER_EXPORTS = (
    "ACTBaseActionProvider",
    "BaseActionProvider",
    "resolve_act_checkpoint_path",
)

__all__ = list(_PROVIDER_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _PROVIDER_EXPORTS:
        return getattr(importlib.import_module("rl_garden.models.act.provider"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
