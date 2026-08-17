"""rl-garden <-> RLinf integration.

RLinf (https://github.com/RLinf/RLinf, vendored read-only at
``3rd_party/RLinf``) is a Ray-based framework for distributed RL/VLA
training. This package hosts rl-garden-side adapter classes that let
rl-garden algorithms run as RLinf workers, per
``docs/design/rlinf-integration.md``.

RLinf is an optional dependency: importing this package must not require it.
Only constructing/launching an adapter class does. See
:func:`require_rlinf` for the import guard used throughout this package.
"""
from __future__ import annotations


def require_rlinf() -> None:
    """Raise a clear error if RLinf is not importable.

    Call this before importing any ``rlinf.*`` module from within this
    package, so a missing RLinf install fails with an actionable message
    instead of a bare ``ModuleNotFoundError`` deep in an adapter class body.
    """
    try:
        import rlinf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "rl_garden.integrations.rlinf requires RLinf to be installed. "
            "See https://github.com/RLinf/RLinf for install instructions "
            "(the vendored reference copy at 3rd_party/RLinf is read-only "
            "and not itself an installed package)."
        ) from exc
