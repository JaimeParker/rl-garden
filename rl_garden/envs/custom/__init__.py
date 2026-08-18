"""``custom`` env backend: a template for authoring a brand-new environment
that isn't wrapping an existing simulator (ManiSkill, MuJoCo, RoboTwin, ...).

Copy this whole directory to start a new backend: rename the package,
replace ``point_reach_env.py`` with your own ``gymnasium.Env`` subclass
(standard, single-instance, numpy-based -- see that file's docstring for
the gymnasium 1.3 authoring contract), keep ``env.py``'s
vectorize-then-adapt shape, update ``config.py``'s fields for your task,
and register a new key in ``rl_garden/envs/backends/<name>.py`` +
``EnvBackendArgs`` (``rl_garden/common/env_args.py``) following
``.agents/rules/adding-env-backend.md``.
"""
from rl_garden.envs.custom.config import CustomEnvConfig
from rl_garden.envs.custom.env import make_custom_env

__all__ = ["CustomEnvConfig", "make_custom_env"]
