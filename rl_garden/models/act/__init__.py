"""ACT base-action providers and vendored ACT reference code.

The original ACT implementation is copied under this package from
``3rd_party/residual-rl/policies/act``.  The public API here wraps that code in
the base-action-provider contract used by ``ResidualSAC``.
"""

from rl_garden.models.act.provider import (
    ACTBaseActionProvider,
    BaseActionProvider,
    resolve_act_checkpoint_path,
)

__all__ = [
    "ACTBaseActionProvider",
    "BaseActionProvider",
    "resolve_act_checkpoint_path",
]
