"""Env backend registry: zero-if-else env creation via protocol + registry dict.

Each run function fills an :class:`EnvRequest` — a backend-neutral spec —
and calls :func:`make_training_envs`. The registry maps ``env_backend`` names to
:class:`EnvBackend` subclasses that translate the request into backend-specific
env configs and create the envs.

To add a new backend::

    # rl_garden/envs/backends/my_backend.py
    class MyBackend(EnvBackend):
        @classmethod
        def make_train_env(cls, req: EnvRequest): ...
        @classmethod
        def make_eval_env(cls, req: EnvRequest): ...

    register_env_backend("my_backend", MyBackend)

Backends are discovered automatically on first use.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
from typing import Any, Optional


@dataclass
class EnvRequest:
    """Backend-neutral env spec. Run functions fill this; backends translate it."""

    env_id: str
    num_envs: int
    obs_mode: str           # "state" | "rgb" | "rgbd"
    control_mode: str
    render_mode: str
    seed: int
    # Visual fields (None when obs_mode == "state"):
    camera_width: Optional[int]
    camera_height: Optional[int]
    include_state: bool = True
    per_camera_rgbd: bool = False
    frame_stack: int = 1
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Eval env:
    num_eval_envs: int = 8
    eval_record_dir: Optional[str] = None
    capture_video: bool = True
    video_fps: int = 30
    num_eval_steps: int = 50
    # Set to False to skip eval env creation (e.g. DrQv2 when eval_freq == 0).
    create_eval_env: bool = True
    # Backend-specific extras, opaque to the registry:
    backend_config: Any = None  # e.g. ManiSkillConfig | RoboTwinConfig


class EnvBackend:
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
    def make_eval_env(cls, req: EnvRequest):
        raise NotImplementedError


_REGISTRY: dict[str, type[EnvBackend]] = {}
_DISCOVERED = False


def register_env_backend(name: str, cls: type[EnvBackend]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Environment backend {name!r} already registered")
    _REGISTRY[name] = cls


def discover_env_backends() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module("rl_garden.envs.backends")
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"rl_garden.envs.backends.{info.name}")
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


def make_evaluation_env(backend_name: str, req: EnvRequest):
    """Create only an evaluation environment for offline training."""
    return _get_backend(backend_name).make_eval_env(req)


def make_training_envs(backend_name: str, req: EnvRequest):
    """Create train and optional evaluation environments."""
    backend = _get_backend(backend_name)
    train_env = backend.make_train_env(req)
    eval_env = backend.make_eval_env(req) if req.create_eval_env else None
    return train_env, eval_env
