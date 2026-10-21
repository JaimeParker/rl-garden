"""rl-garden's own Isaac Lab 3.0 tasks, built on ``v3.direct_env.RLGardenDirectRLEnv``.

Import time triggers each task module's ``gym.register(...)`` call. Isaac
Lab 3.0 allows importing these before the simulation runtime is launched;
``rl_garden.envs.isaaclab.v3.runtime.build`` imports them lazily so
``rl_garden.envs`` stays importable without Isaac Lab installed.
"""
from rl_garden.envs.isaaclab.v3.tasks.cartpole_direct import CartpoleDirectEnv
from rl_garden.envs.isaaclab.v3.tasks.cartpole_direct_camera import CartpoleDirectCameraEnv

__all__ = ["CartpoleDirectEnv", "CartpoleDirectCameraEnv"]
