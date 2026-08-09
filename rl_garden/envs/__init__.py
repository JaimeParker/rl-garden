from __future__ import annotations

from typing import Any


def register_custom_envs() -> None:
    """Register rl-garden's vendored ManiSkill environments if deps are present."""
    try:
        import rl_garden.envs.custom  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"sapien", "mani_skill"}:
            raise


def __getattr__(name: str) -> Any:
    """Keep optional simulator implementations lazy until explicitly requested."""
    if name in {"ManiSkillEnvConfig", "make_maniskill_env"}:
        from rl_garden.envs import maniskill

        return getattr(maniskill, name)
    if name in {"RoboTwinEnvConfig", "make_robotwin_env"}:
        from rl_garden.envs import robotwin

        return getattr(robotwin, name)
    raise AttributeError(name)


__all__ = [
    "ManiSkillEnvConfig",
    "RoboTwinEnvConfig",
    "make_maniskill_env",
    "make_robotwin_env",
    "register_custom_envs",
]
