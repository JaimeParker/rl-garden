from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Keep optional simulator implementations lazy until explicitly requested."""
    if name in {"ManiSkillEnvConfig", "make_maniskill_env"}:
        from rl_garden.envs import maniskill

        return getattr(maniskill, name)
    if name in {"RoboTwinEnvConfig", "make_robotwin_env"}:
        from rl_garden.envs import robotwin

        return getattr(robotwin, name)
    if name in {"GymnasiumEnvConfig", "make_gymnasium_env"}:
        from rl_garden.envs import gymnasium

        return getattr(gymnasium, name)
    raise AttributeError(name)


__all__ = [
    "GymnasiumEnvConfig",
    "ManiSkillEnvConfig",
    "RoboTwinEnvConfig",
    "make_gymnasium_env",
    "make_maniskill_env",
    "make_robotwin_env",
]
