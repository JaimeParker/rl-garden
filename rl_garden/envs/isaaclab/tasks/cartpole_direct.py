"""``RLGardenDirectRLEnv`` reimplementation of IsaacLab's Cartpole Direct
task, registered as ``RlGarden-Cartpole-Direct-v0``.

Logic (action scaling, reward terms, termination condition, reset pole-angle
jitter) is copied from Isaac Lab 2.x's ``isaaclab_tasks.direct.cartpole.
cartpole_env`` and kept unchanged by the 3.0 port (3.0's official task in
``isaaclab_tasks.core.cartpole`` changed observations, rewards and reset
randomization). This exists to verify parity: the scaffold's generic
``_setup_scene``/``_reset_idx`` should behave identically to the official
hand-written version, only the observation key (``"state"`` instead of
``"policy"``, per rl-garden's cross-backend convention) differs.

Physics is a ``PresetCfg`` (``--isaaclab.physics``): Isaac Sim PhysX by
default (the 2.x backend), ``newton_mjwarp`` etc. selectable, with the
official cartpole's per-backend solver tuning.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import gymnasium as gym
import torch

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.cartpole.cartpole_common import CartpolePhysicsCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from rl_garden.envs.isaaclab.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv, RLGardenSceneCfg


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


class CartpoleDirectEnv(RLGardenDirectRLEnv):
    cfg: CartpoleDirectEnvCfg

    def __init__(self, cfg: CartpoleDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(target=self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"state": obs}

    def _get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
        )
        return out_of_bounds, time_out

    def _sample_reset_state(
        self,
        env_ids: Sequence[int],
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        # _get_observations() reads the cached self.joint_pos/self.joint_vel
        # (last refreshed in _get_dones(), which runs before _reset_idx() in
        # DirectRLEnv.step()) -- sync the just-reset envs' slice now so the
        # observation returned for this step reflects the actual post-reset
        # state, matching the official cartpole_env.py's own _reset_idx patch.
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        return joint_pos, joint_vel


@torch.jit.script
def _compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    return rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel


gym.register(
    id="RlGarden-Cartpole-Direct-v0",
    entry_point=f"{__name__}:CartpoleDirectEnv",
    disable_env_checker=True,
    # make_isaaclab_env() resolves this via IsaacLab's own
    # parse_env_cfg()/load_cfg_from_registry(), matching the convention every
    # native isaaclab_tasks registration uses.
    kwargs={"env_cfg_entry_point": f"{__name__}:CartpoleDirectEnvCfg"},
)
