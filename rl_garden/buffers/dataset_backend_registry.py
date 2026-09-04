"""Dataset backend registry: zero-if-else offline dataset loading.

Mirrors ``rl_garden.envs.backend_registry``'s ``EnvBackend``/registry-dict
pattern: adding a new dataset backend should only require registering it
once (at the bottom of the backend's own ``<name>_dataset.py`` module), not
editing every training entrypoint that consumes ``--dataset_backend``.

To add a new dataset backend::

    class MyDatasetBackend(DatasetBackend):
        @classmethod
        def infer_specs(cls, req: DatasetRequest): ...
        @classmethod
        def load(cls, buffer, req: DatasetRequest) -> int: ...

    register_dataset_backend("my_backend", MyDatasetBackend)

``DatasetRequest`` carries every field any backend might need (mirrors
``EnvRequest``) -- a backend that doesn't need ``obs_mode``/
``backend_config``/etc. just ignores them, exactly like ``EnvBackend
.resolve_config`` ignores ``EnvRequest`` fields it doesn't need. This is a
uniform, explicit contract on purpose (every backend implements the same
two methods with the same signature) rather than reflection-based partial
kwarg-forwarding -- the intended foundation for a later pass that further
normalizes what each backend returns, not just this registration problem.

No separate discovery step is needed: ``rl_garden.buffers.__init__`` already
eagerly imports every ``*_dataset.py`` module (each only lazily imports its
real backend package *inside function bodies*, so this stays cheap), which
is what actually triggers each backend's own ``register_dataset_backend(...)``
call the first time anything imports ``rl_garden.buffers``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import spaces


@dataclass
class DatasetRequest:
    """Backend-neutral dataset-loading spec, mirrors ``EnvRequest``."""

    path: str
    num_traj: int | None = None
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: str | None = None
    action_low: float = -1.0
    action_high: float = 1.0
    obs_mode: str | None = None
    # Per-backend CLI config, e.g. RLBenchConfig -- getattr(args, backend_name, None).
    backend_config: Any = None


class DatasetBackend:
    """Dataset backend protocol. See module docstring."""

    @classmethod
    def infer_specs(cls, req: DatasetRequest) -> tuple[spaces.Space, spaces.Box]:
        raise NotImplementedError

    @classmethod
    def load(cls, buffer: Any, req: DatasetRequest) -> int:
        raise NotImplementedError


_REGISTRY: dict[str, type[DatasetBackend]] = {}


def register_dataset_backend(name: str, cls: type[DatasetBackend]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Dataset backend {name!r} already registered")
    for method_name in ("infer_specs", "load"):
        if method_name not in cls.__dict__:
            raise TypeError(f"Dataset backend {name!r} must implement {method_name}().")
    _REGISTRY[name] = cls


def _get_backend(name: str) -> type[DatasetBackend]:
    import rl_garden.buffers  # noqa: F401 -- triggers every backend's own registration.

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset backend {name!r}. Available: {sorted(_REGISTRY)}. "
            "Add and register a loader module under rl_garden.buffers."
        )
    return _REGISTRY[name]


def infer_dataset_specs(
    req: DatasetRequest, *, backend_name: str
) -> tuple[spaces.Space, spaces.Box]:
    return _get_backend(backend_name).infer_specs(req)


def load_dataset(buffer: Any, req: DatasetRequest, *, backend_name: str) -> int:
    return _get_backend(backend_name).load(buffer, req)
