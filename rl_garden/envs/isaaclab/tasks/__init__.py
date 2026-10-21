"""rl-garden's own IsaacLab tasks, built on ``RLGardenDirectRLEnv``.

Import time triggers each task module's ``gym.register(...)`` call. Isaac
Lab 3.0 allows importing these before the simulation runtime is launched;
``rl_garden.envs.isaaclab.env.make_isaaclab_env`` imports them lazily so
``rl_garden.envs`` stays importable without Isaac Lab installed.
"""
from rl_garden.envs.isaaclab.tasks.cartpole_direct import CartpoleDirectEnv
from rl_garden.envs.isaaclab.tasks.cartpole_direct_camera import CartpoleDirectCameraEnv

__all__ = ["CartpoleDirectEnv", "CartpoleDirectCameraEnv"]
