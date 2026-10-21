"""rl-garden's own Isaac Lab 2.x tasks, built on ``v2.direct_env.RLGardenDirectRLEnv``.

Import time triggers each task module's ``gym.register(...)`` call. Must
only be imported *after* ``v2.app.get_or_launch_app()`` has booted Kit (see
``rl_garden.envs.isaaclab.v2.runtime.build``).
"""
from rl_garden.envs.isaaclab.v2.tasks.cartpole_direct import CartpoleDirectEnv
from rl_garden.envs.isaaclab.v2.tasks.cartpole_direct_camera import CartpoleDirectCameraEnv

__all__ = ["CartpoleDirectEnv", "CartpoleDirectCameraEnv"]
