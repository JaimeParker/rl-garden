"""Env backend registry: zero-if-else env creation via protocol + registry dict.

Each run function fills an :class:`EnvRequest` — a backend-neutral spec —
and calls :func:`make_training_envs`. The registry maps ``env_backend`` names to
:class:`EnvBackend` subclasses that translate the request into backend-specific
env configs and create the envs.

To add a new backend::

    # rl_garden/envs/backends/my_backend.py
    class MyBackend(EnvBackend):
        api_version = 2
        config_field = "my_backend"
        @classmethod
        def resolve_config(cls, req: EnvRequest, *, is_eval: bool): ...
        @classmethod
        def make_train_env(cls, req: EnvRequest): ...
        @classmethod
        def make_eval_env(cls, req: EnvRequest): ...

    register_env_backend("my_backend", MyBackend)

Backends are discovered automatically on first use: every module under
``rl_garden.envs.backends`` is imported (each is expected to call
``register_env_backend`` as an import-time side effect), plus every module
advertised via the ``rlgarden.env_backends`` entry point group. The entry
point form lets an externally pip-installed package (e.g. a real-robot
backend living in its own repo) register a backend without any file inside
``rl_garden`` itself — declare it in the external package's own
``pyproject.toml``::

    [project.entry-points."rlgarden.env_backends"]
    my_backend = "my_package.backend_module"

and have ``my_package.backend_module`` call ``register_env_backend(...)`` at
import time, same as an in-tree backend module does.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any

from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError, validate_observation_space

_ENV_BACKEND_ENTRY_POINT_GROUP = "rlgarden.env_backends"


@dataclass
class EnvRequest:
    """Backend-neutral env spec. Run functions fill this; backends translate it."""

    env_id: str
    num_envs: int
    control_mode: str
    render_mode: str
    seed: int
    # "What is observed" -- state / rgb / depth cameras, image size, frame
    # stacking. Replaces the old per-field observation-mode/camera-resolution/
    # frame-stacking knobs this request used to carry directly (deleted). A
    # backend must produce exactly ``observation.expected_keys`` or raise
    # ``ObservationContractError``; see ``rl_garden.observations``.
    observation: ObservationConfig = field(kw_only=True)
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Eval env:
    num_eval_envs: int = 8
    eval_record_dir: str | None = None
    capture_video: bool = True
    video_fps: int = 30
    num_eval_steps: int = 50
    # Set to False to skip eval env creation (e.g. DrQv2 when eval_freq == 0).
    create_eval_env: bool = True
    # Backend-specific extras, opaque to the registry:
    backend_config: Any = None  # e.g. ManiSkillConfig | RoboTwinConfig


class EnvBackend:
    """Version 2 environment backend protocol."""

    api_version: int
    config_field: str

    @classmethod
    def config_from_args(cls, args: Any) -> Any:
        try:
            return getattr(args, cls.config_field)
        except AttributeError as exc:
            raise ValueError(
                f"Backend {cls.__name__!r} requires args.{cls.config_field}"
            ) from exc

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        raise NotImplementedError

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        """Build the concrete backend config without creating an environment."""
        raise NotImplementedError

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        raise NotImplementedError


_REGISTRY: dict[str, type[EnvBackend]] = {}
_DISCOVERED = False
_DISCOVERY_LOCK = threading.Lock()


def register_env_backend(name: str, cls: type[EnvBackend]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Environment backend {name!r} already registered")
    if cls.__dict__.get("api_version") != 2:
        raise TypeError(f"Environment backend {name!r} must declare api_version = 2.")
    if not isinstance(cls.__dict__.get("config_field"), str):
        raise TypeError(f"Environment backend {name!r} must declare config_field.")
    if cls.config_field != name:
        raise TypeError(
            f"Environment backend {name!r} config_field must match its registry name."
        )
    for method_name in ("resolve_config", "make_train_env", "make_eval_env"):
        if method_name not in cls.__dict__:
            raise TypeError(
                f"Environment backend {name!r} API v2 must implement {method_name}()."
            )
    _REGISTRY[name] = cls


def discover_env_backends() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    with _DISCOVERY_LOCK:
        if _DISCOVERED:
            return
        package = importlib.import_module("rl_garden.envs.backends")
        for info in pkgutil.iter_modules(package.__path__):
            if not info.name.startswith("_"):
                importlib.import_module(f"rl_garden.envs.backends.{info.name}")
        for entry_point in importlib.metadata.entry_points(
            group=_ENV_BACKEND_ENTRY_POINT_GROUP
        ):
            entry_point.load()
        _DISCOVERED = True


def _get_backend(backend_name: str) -> type[EnvBackend]:
    """Return a registered backend class."""
    discover_env_backends()
    if backend_name not in _REGISTRY:
        raise KeyError(
            f"Unknown env backend {backend_name!r}. "
            f"Available: {sorted(_REGISTRY)}. "
            "Add and register a backend module under rl_garden.envs.backends."
        )
    return _REGISTRY[backend_name]


def resolve_backend_config(backend_name: str, args: Any) -> Any:
    """Resolve backend-specific CLI config without branching in training code."""
    return _get_backend(backend_name).config_from_args(args)


def should_create_eval_env(args: Any) -> bool:
    """Whether a training run should build a live eval env at all.

    Single source of truth for the "build an eval env only if periodic
    evaluation was actually requested" decision, shared by the online,
    offline, and off2on training entrypoints so they don't each re-derive it
    (and potentially disagree, as online/off2on previously did). Off2on's
    phase-specific ``offline_eval_freq``/``online_eval_freq`` (not present on
    the online/offline-only entrypoints, hence ``getattr``) can each
    independently trigger evaluation even when the general ``eval_freq`` is 0.
    """
    if args.eval_freq > 0:
        return True
    return bool(
        getattr(args, "offline_eval_freq", None)
        or getattr(args, "online_eval_freq", None)
    )


def _validate_env_observation_contract(env: Any, req: EnvRequest) -> None:
    """Enforce the observation contract on a freshly constructed env.

    Validates ``env.single_observation_space`` against the strict
    ``state``/``rgb_<cam>``/``depth_<cam>`` vocabulary and checks its key set
    matches exactly what ``req.observation`` asked for -- a backend that
    silently drops or adds a key fails here instead of surfacing as a
    confusing downstream shape mismatch.
    """
    space = env.single_observation_space
    validate_observation_space(space)
    actual_keys = set(space.spaces.keys()) if hasattr(space, "spaces") else {"state"}
    expected_keys = set(req.observation.expected_keys)
    if actual_keys != expected_keys:
        raise ObservationContractError(
            f"env backend produced observation keys {sorted(actual_keys)}, "
            f"expected {sorted(expected_keys)} from {req.observation!r}"
        )


def make_evaluation_env(backend_name: str, req: EnvRequest):
    """Create only an evaluation environment for offline training."""
    env = _get_backend(backend_name).make_eval_env(req)
    _validate_env_observation_contract(env, req)
    return env


def make_training_envs(backend_name: str, req: EnvRequest):
    """Create train and optional evaluation environments."""
    backend = _get_backend(backend_name)
    train_env = backend.make_train_env(req)
    _validate_env_observation_contract(train_env, req)
    eval_env = None
    if req.create_eval_env:
        eval_env = backend.make_eval_env(req)
        _validate_env_observation_contract(eval_env, req)
    return train_env, eval_env


def materialize_backend_configs(backend_name: str, req: EnvRequest) -> dict[str, Any]:
    """Resolve train/eval backend configs without constructing simulator resources."""
    backend = _get_backend(backend_name)
    configs = {"train": backend.resolve_config(req, is_eval=False)}
    if req.create_eval_env:
        configs["eval"] = backend.resolve_config(req, is_eval=True)
    return configs
