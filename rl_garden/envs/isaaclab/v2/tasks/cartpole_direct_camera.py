"""Isaac Lab 2.x cfg for the camera variant of the Cartpole Direct task,
registered as ``RlGarden-Cartpole-Direct-Camera-v0``. Logic in
``tasks_common.cartpole.CartpoleCameraLogic``; rendering is always Kit RTX
through ``TiledCamera``.
"""
from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.tasks_common.cartpole import CartpoleCameraLogic
from rl_garden.envs.isaaclab.v2.direct_env import RLGardenDirectRLEnv
from rl_garden.envs.isaaclab.v2.tasks.cartpole_direct import CartpoleDirectEnvCfg, CartpoleDirectSceneCfg


@configclass
class CartpoleDirectCameraSceneCfg(CartpoleDirectSceneCfg):
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # Quaternion is (w, x, y, z) on 2.x: identity orientation in the world convention.
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        # PlainConv supports only 64x64 or 128x128 (see the v3 camera task).
        width=64,
        height=64,
    )


@configclass
class CartpoleDirectCameraEnvCfg(CartpoleDirectEnvCfg):
    # Placeholder; the adapter builds the Dict space from the runtime obs.
    observation_space = 1
    # clone_in_fabric=True (used by the state-only base cfg) mis-sizes the
    # camera sensor's per-env buffers -- IsaacLab's own camera example
    # (cartpole_camera_env.py) omits it too.
    scene: CartpoleDirectCameraSceneCfg = CartpoleDirectCameraSceneCfg(
        num_envs=32, env_spacing=4.0, replicate_physics=True
    )


class CartpoleDirectCameraEnv(CartpoleCameraLogic, RLGardenDirectRLEnv):
    cfg: CartpoleDirectCameraEnvCfg


gym.register(
    id="RlGarden-Cartpole-Direct-Camera-v0",
    entry_point=f"{__name__}:CartpoleDirectCameraEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:CartpoleDirectCameraEnvCfg"},
)
