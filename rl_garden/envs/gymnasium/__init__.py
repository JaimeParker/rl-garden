"""``gymnasium`` env backend: trains on any registered, single-instance
``gymnasium.Env`` without forking rl-garden or writing a backend.

``--gymnasium.import_modules`` names the module(s) whose import calls
``gymnasium.register`` for ``--env_id``; the env must emit rl-garden
observation keys itself (or a bare 1-D float ``Box``, auto-wrapped as
``state``). See ``docs/guides/gymnasium-custom-envs.md``.
"""
from rl_garden.envs.gymnasium.config import GymnasiumEnvConfig
from rl_garden.envs.gymnasium.env import make_gymnasium_env

__all__ = ["GymnasiumEnvConfig", "make_gymnasium_env"]
