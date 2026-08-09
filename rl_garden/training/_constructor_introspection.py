"""Shared constructor signature inspection for training configuration."""

from __future__ import annotations

import inspect


def inspect_constructor_parameters(
    target: type,
) -> dict[str, inspect.Parameter]:
    """Return named constructor parameters across ``target``'s MRO."""
    parameters: dict[str, inspect.Parameter] = {}
    for cls in target.__mro__:
        if cls is object or "__init__" not in cls.__dict__:
            continue
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        for name, parameter in signature.parameters.items():
            if (
                name in parameters
                or name in {"self", "env", "eval_env", "logger"}
                or parameter.kind
                in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            ):
                continue
            parameters[name] = parameter
    return parameters
