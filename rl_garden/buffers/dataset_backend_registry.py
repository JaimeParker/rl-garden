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
``EnvRequest``) -- a backend that doesn't need ``observation``/
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

from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


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
    # Per-backend CLI config, e.g. RLBenchConfig -- getattr(args, backend_name, None).
    backend_config: Any = None
    # Phase-3 observation surface. When set, ``infer_dataset_specs`` asserts
    # the loaded space's key set equals ``observation.expected_keys`` --
    # lets an offline run assert the same observation composition an online
    # env would produce. ``None`` means "the dataset's own keys are the
    # truth" (no assertion). Individual backends may also consult this
    # field directly (see e.g. ``rlbench_dataset.py``) to decide what to
    # load, not just to validate afterwards.
    observation: ObservationConfig | None = None


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
    obs_space, action_space = _get_backend(backend_name).infer_specs(req)
    if req.observation is not None:
        if not isinstance(obs_space, spaces.Dict):
            raise ObservationContractError(
                f"dataset backend {backend_name!r} returned a "
                f"{type(obs_space).__name__} observation space, but "
                "req.observation was given -- every dataset loader must "
                "return a Dict observation space to be checked against it."
            )
        actual_keys = set(obs_space.spaces.keys())
        expected_keys = set(req.observation.expected_keys)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ObservationContractError(
                f"dataset backend {backend_name!r} produced observation keys "
                f"{sorted(actual_keys)!r}, but req.observation expects "
                f"{sorted(expected_keys)!r} (missing={missing!r}, extra={extra!r})"
            )
    return obs_space, action_space


def load_dataset(buffer: Any, req: DatasetRequest, *, backend_name: str) -> int:
    return _get_backend(backend_name).load(buffer, req)
