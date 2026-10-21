"""Isaac Lab 3.0 cfg for the camera variant of the Cartpole Direct task,
registered as ``RlGarden-Cartpole-Direct-Camera-v0``.

Verifies the image-observation path: ``"rgb_front"`` is the raw camera
output (``uint8`` ``[0, 255]``, no mean subtraction) next to the 4-value
``"state"`` (logic in ``tasks_common.cartpole.CartpoleCameraLogic``).

The camera renderer is a ``PresetCfg`` (``--isaaclab.renderer``): the Kit-less
Newton Warp renderer by default, ``isaacsim_rtx`` (Kit RTX) etc. selectable.
Physics presets are inherited from ``CartpoleDirectEnvCfg``.
"""
from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from rl_garden.envs.isaaclab.tasks_common.cartpole import CartpoleCameraLogic
from rl_garden.envs.isaaclab.v3.direct_env import RLGardenDirectRLEnv
from rl_garden.envs.isaaclab.v3.tasks.cartpole_direct import CartpoleDirectEnvCfg, CartpoleDirectSceneCfg


@configclass
class CartpoleDirectCameraSceneCfg(CartpoleDirectSceneCfg):
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # Quaternion is (x, y, z, w): identity orientation in the world convention.
        offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        # rl-garden's default PlainConv image encoder only supports 64x64 or
        # 128x128 inputs (hardcoded conv/pool stride schedule); this task's
        # resolution is fixed here (ObservationConfig.image_size can't
        # override it -- see rl_garden.envs.isaaclab.env module docstring).
        width=64,
        height=64,
        # Defaults to newton_renderer; isaacsim_rtx/ovrtx/rtx are selectable.
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class CartpoleDirectCameraEnvCfg(CartpoleDirectEnvCfg):
    # observation_space can't express the {"rgb_front": ..., "state": ...}
    # dict shape RLGardenDirectRLEnv-scaffold tasks return -- placeholder
    # value only, see direct_env.py's module docstring. rl-garden's own adapter
    # builds its gym.spaces.Dict from the runtime obs dict instead.
    observation_space = 1
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
