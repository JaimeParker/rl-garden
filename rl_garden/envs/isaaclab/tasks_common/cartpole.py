"""Cartpole Direct task logic shared by the ``v2`` and ``v3`` task packages.

Logic (action scaling, reward terms, termination condition, reset pole-angle
jitter) is copied from Isaac Lab 2.x's ``isaaclab_tasks.direct.cartpole.
cartpole_env`` (3.0's official ``isaaclab_tasks.core.cartpole`` changed
observations, rewards and reset randomization). Only the observation key
(``"state"`` instead of ``"policy"``, per rl-garden's cross-backend
convention) differs. Combine with a scaffold as
``class Env(CartpoleLogic, RLGardenDirectRLEnv)``; the cfg must define
``cart_dof_name``/``pole_dof_name``/``max_cart_pos``/
``initial_pole_angle_range``/``rew_scale_*``.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


def _sample_uniform(lower: float, upper: float, size: tuple[int, ...], device) -> torch.Tensor:
    # Same draw as isaaclab.utils.math.sample_uniform, without importing isaaclab.
    return torch.rand(*size, device=device) * (upper - lower) + lower


class CartpoleLogic:
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        self.joint_pos = self._joint_pos()
        self.joint_vel = self._joint_vel()

    def _apply_action(self) -> None:
        self._apply_joint_effort(self.actions, self._cart_dof_idx)

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
        self.joint_pos = self._joint_pos()
        self.joint_vel = self._joint_vel()

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
        joint_pos[:, self._pole_dof_idx] += _sample_uniform(
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


class CartpoleCameraLogic(CartpoleLogic):
    """Adds a raw-pixel ``"rgb_front"`` key (uint8 ``[0, 255]``, no mean
    subtraction) next to the 4-value ``"state"``."""

    def _get_observations(self) -> dict:
        state = super()._get_observations()["state"]
        return {"rgb_front": self._camera_rgb(), "state": state}


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
