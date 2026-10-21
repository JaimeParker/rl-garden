"""Isaac Lab 2.x cfg for rl-garden's Cartpole Direct task, registered as
``RlGarden-Cartpole-Direct-v0``. Task logic lives in
``rl_garden.envs.isaaclab.tasks_common.cartpole``; physics is always Isaac
Sim PhysX.
"""
from __future__ import annotations

import gymnasium as gym

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.tasks_common.cartpole import CartpoleLogic
from rl_garden.envs.isaaclab.v2.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv, RLGardenSceneCfg


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

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    scene: CartpoleDirectSceneCfg = CartpoleDirectSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

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
    kwargs={"env_cfg_entry_point": f"{__name__}:CartpoleDirectEnvCfg"},
)
