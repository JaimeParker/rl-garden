"""rl-garden's own IsaacLab tasks, built on ``RLGardenDirectRLEnv``.

Import time triggers each task module's ``gym.register(...)`` call. Must
only be imported *after* ``get_or_launch_app()`` has booted Kit (see
``rl_garden.envs.isaaclab.env.make_isaaclab_env``) -- IsaacLab requires the
app to exist before any ``isaaclab.*`` subclass can be imported.
"""
from rl_garden.envs.isaaclab.tasks.cartpole_direct import CartpoleDirectEnv
from rl_garden.envs.isaaclab.tasks.cartpole_direct_camera import CartpoleDirectCameraEnv
from rl_garden.envs.isaaclab.tasks.cartpole_direct_camera_plain import CartpoleDirectCameraPlainEnv

__all__ = ["CartpoleDirectEnv", "CartpoleDirectCameraEnv", "CartpoleDirectCameraPlainEnv"]
