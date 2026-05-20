def register_custom_envs() -> None:
    """Register rl-garden's vendored ManiSkill environments if deps are present."""
    try:
        import rl_garden.envs.custom  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"sapien", "mani_skill"}:
            raise


register_custom_envs()

from rl_garden.envs.maniskill import ManiSkillEnvConfig, make_maniskill_env
from rl_garden.envs.robotwin import RoboTwinEnvConfig, make_robotwin_env

__all__ = [
    "ManiSkillEnvConfig",
    "RoboTwinEnvConfig",
    "make_maniskill_env",
    "make_robotwin_env",
    "register_custom_envs",
]
