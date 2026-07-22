"""ACT base-action providers and vendored ACT reference code.

The original ACT implementation is copied under this package from
``3rd_party/residual-rl/policies/act``.  The public API here wraps that code in
the base-action-provider contract used by ``ResidualSAC``.
"""

__all__ = [
    "ACTBaseActionProvider",
    "BaseActionProvider",
    "resolve_act_checkpoint_path",
]


def __getattr__(name):
    if name in __all__:
        from rl_garden.models.act import provider

        return getattr(provider, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
