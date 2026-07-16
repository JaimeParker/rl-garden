"""Camera variant of ``CartpoleDirectEnv``, registered as
``RlGarden-Cartpole-Direct-Camera-v0``.

Verifies the image-observation path: unlike IsaacLab's own
``cartpole_camera_env.py`` (which mean-subtracts the RGB tensor for
"better training results"), ``_get_observations()`` here returns the
``TiledCamera`` output untouched -- raw ``uint8`` ``[0, 255]`` -- per
``RLGardenDirectRLEnv``'s documented image convention. It also returns a
``"state"`` key alongside ``"rgb"`` (the same 4-value state
``CartpoleDirectEnv`` uses), demonstrating the state+image ``include_state``
case for backends other than ManiSkill.
"""
from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.tasks.cartpole_direct import CartpoleDirectEnv, CartpoleDirectEnvCfg


@configclass
class CartpoleDirectCameraEnvCfg(CartpoleDirectEnvCfg):
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        # rl-garden's default PlainConv image encoder only supports 64x64 or
        # 128x128 inputs (hardcoded conv/pool stride schedule) -- match the
        # CLI's --camera_width/--camera_height default of 64.
        width=64,
        height=64,
    )
    # observation_space can't express the {"rgb": ..., "state": ...} dict
    # shape RLGardenDirectRLEnv-scaffold tasks return -- placeholder value
    # only, see direct_env.py's module docstring. rl-garden's own adapter
    # builds its gym.spaces.Dict from the runtime obs dict instead.
    observation_space = 1
    # clone_in_fabric=True (used by the state-only base cfg) mis-sizes the
    # camera sensor's per-env buffers -- IsaacLab's own camera example
    # (cartpole_camera_env.py) omits it too.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True)


class CartpoleDirectCameraEnv(CartpoleDirectEnv):
    cfg: CartpoleDirectCameraEnvCfg

    def _get_observations(self) -> dict:
        state = super()._get_observations()["state"]
        rgb = self.camera.data.output["rgb"]
        return {"rgb": rgb, "state": state}


gym.register(
    id="RlGarden-Cartpole-Direct-Camera-v0",
    entry_point=f"{__name__}:CartpoleDirectCameraEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:CartpoleDirectCameraEnvCfg"},
)
