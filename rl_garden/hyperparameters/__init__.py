"""Parameter projection and keyword-compatible resolution."""
from dataclasses import fields
from typing import Any, Mapping, TypeVar

Params = TypeVar("Params")


def resolve_params(
    params_class: type[Params], params: Params | None, overrides: Mapping[str, Any]
) -> Params:
    """A new ``params_class`` instance: ``params`` (or the defaults) with keyword overrides.

    ``params`` may be an instance of a subclass, such as an Args object that inherits
    ``params_class``; only the declared parameter fields are taken from it. The result
    is a snapshot of the inputs and is not meant to be mutated afterwards.
    """
    unknown = overrides.keys() - {field.name for field in fields(params_class)}
    if unknown:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(unknown))}")
    if params is None:
        return params_class(**overrides)
    if not isinstance(params, params_class):
        raise TypeError(
            f"params must be a {params_class.__name__}, got {type(params).__name__}"
        )
    return params_class(**{**params_kwargs(params, params_class), **overrides})


def params_kwargs(args: Any, params_class: type) -> dict[str, Any]:
    """Project the declared parameters from an Args object or algorithm."""
    return {field.name: getattr(args, field.name) for field in fields(params_class)}
