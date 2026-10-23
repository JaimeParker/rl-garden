"""rl-garden's ``DirectRLEnv`` authoring scaffold for Isaac Lab 2.x.

Same shape and contract as the 3.0 scaffold (``rl_garden.envs.isaaclab.v3.
direct_env``, whose module docstring documents the observation-key and
raw-pixel conventions and the ``gym.spaces.Dict`` ``cfg.observation_space``
declaration the adapter validates and uses; depth keys are passed to the
network unscaled, in whatever unit the task emits, so emit finite float32
values and clip Isaac Lab's ``inf`` for no-hit to a max range):
``RLGardenSceneCfg`` declares one robot, an optional camera, a ground plane
and a light; ``InteractiveScene`` spawns and clones them from the cfg (with
``{ENV_REGEX_NS}`` prim paths) before ``_setup_scene()`` runs, which only binds
``self.robot``/``self.camera``.

2.x data conventions: asset/sensor ``.data.*`` fields and
``camera.data.output[...]`` are plain torch tensors; quaternions are
``(w, x, y, z)`` (identity ``(1.0, 0.0, 0.0, 0.0)``); tiled camera rendering
needs ``TiledCameraCfg`` (plain ``CameraCfg`` is the slow per-camera path).

Version-neutral accessors (same signatures as the 3.0 scaffold; the only
interface the ``tasks_common`` logic mixins use): ``_joint_pos()``/
``_joint_vel()`` return ``(num_envs, num_joints)`` torch tensors,
``_apply_joint_effort(target, joint_ids)`` sets joint effort targets,
``_camera_rgb()`` returns an owned contiguous ``uint8`` ``(num_envs, H, W, 3)``
copy of the camera output.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import Optional, Sequence, Tuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class RLGardenSceneCfg(InteractiveSceneCfg):
    """Scene shape ``RLGardenDirectRLEnv`` expects: one robot, optional camera."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = MISSING
    camera: Optional[TiledCameraCfg] = None
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class RLGardenDirectEnvCfg(DirectRLEnvCfg):
    """``DirectRLEnvCfg`` + the action scale ``RLGardenDirectRLEnv`` applies;
    ``scene`` must be an ``RLGardenSceneCfg`` subclass."""

    scene: RLGardenSceneCfg = MISSING
    action_scale: float = 1.0


class RLGardenDirectRLEnv(DirectRLEnv):
    """Task authors override ``_apply_action``/``_get_observations``/
    ``_get_rewards``/``_get_dones``; ``_setup_scene``/``_reset_idx``/
    ``_pre_physics_step`` already have generic implementations for the
    single-articulation (+ optional single-camera) case.
    """

    cfg: RLGardenDirectEnvCfg

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.camera = self.scene.sensors.get("camera")

    def _reset_idx(self, env_ids: Optional[Sequence[int]]) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        joint_pos, joint_vel = self._sample_reset_state(env_ids, joint_pos, joint_vel)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_reset_state(
        self,
        env_ids: Sequence[int],
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optional hook for reset-time randomization (e.g. initial pose jitter).

        Default: no randomization, joints reset to their default pos/vel.
        """
        return joint_pos, joint_vel

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.cfg.action_scale * actions.clone()

    def _joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    def _joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    def _apply_joint_effort(self, target: torch.Tensor, joint_ids: list[int]) -> None:
        self.robot.set_joint_effort_target(target, joint_ids=joint_ids)

    def _camera_rgb(self) -> torch.Tensor:
        # Non-contiguous view of the render buffer, which the next render
        # overwrites; clone to own a contiguous copy.
        return self.camera.data.output["rgb"].clone()

    # _apply_action / _get_observations / _get_rewards / _get_dones stay
    # abstract (inherited from DirectRLEnv) -- task authors implement these.
