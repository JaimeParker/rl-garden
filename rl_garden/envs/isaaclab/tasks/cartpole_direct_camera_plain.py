"""Diagnostic variant of ``CartpoleDirectCameraEnv`` using IsaacLab's plain
``Camera`` sensor instead of ``TiledCamera``, registered as
``RlGarden-Cartpole-Direct-Camera-Plain-v0``.

``TiledCamera``-based training reliably stalls on 6017-nofwd right after env
setup completes (Replicator graph teardown/reload warnings with no further
progress, reproducible independent of ``num_envs``, resolution, and
``rendering_mode``). This variant isolates whether ``TiledCamera``'s
Replicator-based tiled-rendering path specifically is at fault, by spawning
a plain ``Camera`` directly in ``_setup_scene()`` instead of going through
``RLGardenDirectRLEnv``'s ``camera_cfg`` field (which is typed for
``TiledCameraCfg``).
"""
from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.tasks.cartpole_direct import CartpoleDirectEnv, CartpoleDirectEnvCfg

_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Camera",
    offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    ),
    width=64,
    height=64,
)


@configclass
class CartpoleDirectCameraPlainEnvCfg(CartpoleDirectEnvCfg):
    observation_space = 1
    # clone_in_fabric=True (used by the state-only base cfg) mis-sizes camera
    # sensor per-env buffers -- same fix as CartpoleDirectCameraEnvCfg.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True)


class CartpoleDirectCameraPlainEnv(CartpoleDirectEnv):
    cfg: CartpoleDirectCameraPlainEnvCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.camera = Camera(_CAMERA_CFG)
        self.scene.sensors["camera"] = self.camera

    def _get_observations(self) -> dict:
        state = super()._get_observations()["state"]
        rgb = self.camera.data.output["rgb"]
        return {"rgb": rgb, "state": state}


gym.register(
    id="RlGarden-Cartpole-Direct-Camera-Plain-v0",
    entry_point=f"{__name__}:CartpoleDirectCameraPlainEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:CartpoleDirectCameraPlainEnvCfg"},
)
