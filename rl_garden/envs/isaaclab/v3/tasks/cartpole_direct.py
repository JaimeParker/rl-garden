"""Isaac Lab 3.0 cfg for rl-garden's Cartpole Direct task, registered as
``RlGarden-Cartpole-Direct-v0``. Task logic lives in
``rl_garden.envs.isaaclab.tasks_common.cartpole``.

Physics is a ``PresetCfg`` (``--isaaclab.physics``): Isaac Sim PhysX by
default (the 2.x backend), ``newton_mjwarp`` etc. selectable, with the
official cartpole's per-backend solver tuning.
"""
from __future__ import annotations

import gymnasium as gym

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.cartpole.cartpole_common import CartpolePhysicsCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.tasks_common.cartpole import CartpoleLogic
from rl_garden.envs.isaaclab.v3.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv, RLGardenSceneCfg


@configclass
class CartpoleDirectPhysicsCfg(CartpolePhysicsCfg):
    """Official cartpole physics presets with Isaac Sim PhysX as the default."""

    default: PhysxCfg = PhysxCfg()


@configclass
class CartpoleDirectSceneCfg(RLGardenSceneCfg):
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class CartpoleDirectEnvCfg(RLGardenDirectEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, physics=CartpoleDirectPhysicsCfg()
    )

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    scene: CartpoleDirectSceneCfg = CartpoleDirectSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    max_cart_pos = 3.0
    initial_pole_angle_range = [-0.25, 0.25]

    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class CartpoleDirectEnv(CartpoleLogic, RLGardenDirectRLEnv):
    cfg: CartpoleDirectEnvCfg


gym.register(
    id="RlGarden-Cartpole-Direct-v0",
    entry_point=f"{__name__}:CartpoleDirectEnv",
    disable_env_checker=True,
    # make_isaaclab_env() resolves this via IsaacLab's own
    # parse_env_cfg()/load_cfg_from_registry(), matching the convention every
    # native isaaclab_tasks registration uses.
    kwargs={"env_cfg_entry_point": f"{__name__}:CartpoleDirectEnvCfg"},
)
