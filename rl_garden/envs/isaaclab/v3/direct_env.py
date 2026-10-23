"""rl-garden's ``DirectRLEnv`` authoring scaffold.

IsaacLab's ``DirectRLEnv`` (direct-implementation style) skips the
``ManagerBasedRLEnv`` per-term dispatch and expects task authors to write
fully-tensorized batch computations directly. In practice, most Direct tasks
share the same scene shape -- one primary articulation (+ optionally one
camera), a ground plane and a light -- and the same reset: restore default
joint state with a scene-origin offset (see IsaacLab's own
``isaaclab_tasks/core/cartpole/cartpole_direct_env.py``).

In Isaac Lab 3.0 the scene is built entirely from the scene cfg:
``InteractiveScene`` spawns, replicates and collision-filters every asset
declared as an attribute of ``cfg.scene`` before ``_setup_scene()`` runs.
``RLGardenSceneCfg`` declares that common shape (``ground``/``robot``/
``camera``/``light``; per-env prim paths use the ``{ENV_REGEX_NS}`` macro),
``RLGardenDirectRLEnv._setup_scene`` only binds ``self.robot``/
``self.camera`` handles, and ``_reset_idx`` restores the defaults. A task
author only has to implement the four methods that are genuinely
task-specific: ``_apply_action``, ``_get_observations``, ``_get_rewards``,
``_get_dones``. Tasks with a different scene shape add assets to their own
``RLGardenSceneCfg`` subclass and override ``_setup_scene``/``_reset_idx``
as needed, same as any ``DirectRLEnv`` subclass.

3.0 data conventions: asset/sensor ``.data.*`` fields and
``camera.data.output[...]`` are ``ProxyArray`` objects -- read them through
``.torch`` (a zero-copy torch view). Quaternions are ``(x, y, z, w)``
everywhere (e.g. an identity camera offset is ``rot=(0.0, 0.0, 0.0, 1.0)``).

Version-neutral accessors (same signatures as the 2.x scaffold in
``rl_garden.envs.isaaclab.v2.direct_env``; the only interface the
``tasks_common`` logic mixins use): ``_joint_pos()``/``_joint_vel()`` return
``(num_envs, num_joints)`` torch views, ``_apply_joint_effort(target,
joint_ids)`` sets joint effort targets, ``_camera_rgb()`` returns an owned
contiguous ``uint8`` ``(num_envs, H, W, 3)`` tensor.

Observation key convention: unlike IsaacLab's native tasks (which return
``{"policy": obs}``), ``_get_observations()`` here should return rl-garden's
own cross-backend keys -- ``"state"`` for the flat proprioceptive tensor,
``"rgb_<camera>"``/``"depth_<camera>"`` for images, always named even for a
single camera (the observation contract has no bare rgb/depth keys) -- so
the resulting env plugs directly into the same ``ImageFrameStackWrapper``
used for other backends. Image tensors must be **raw pixel values** (e.g.
IsaacLab's own ``camera.data.output["rgb"].torch``, passed through
untouched, uint8 ``[0, 255]``): do NOT apply the mean-subtraction
IsaacLab's own camera examples do for RGB. rl-garden's
image encoders decide whether to apply ``/255`` normalization from the
declared space's dtype/range (see
``rl_garden.encoders.base.image_needs_normalization``), so pre-normalized
input is silently handled wrong rather than raising an error. That rule is
RGB-only: ``depth_<camera>`` keys are passed to the network unscaled, in
whatever unit the task emits, so the task must emit finite float32 values
(clip Isaac Lab's ``inf`` for no-hit to a max range).

Declare ``cfg.observation_space`` as a ``gym.spaces.Dict`` keyed by exactly
the keys ``_get_observations()`` returns, one per-env ``Box`` each (e.g.
``Box(0, 255, (H, W, 3), np.uint8)`` for ``"rgb_<camera>"``,
``Box(0.0, 1.0, (H, W, 1), np.float32)`` for a ``[0, 1]`` depth range).
IsaacLab never checks the runtime obs dict against it; rl-garden's adapter
validates it against the first observation (key set, shape, dtype; raises
``ObservationContractError``) and uses its bounds as the env's observation
space. An int placeholder (e.g. ``1``) still satisfies
``DirectRLEnvCfg.validate()`` and falls back to inferring the spaces from the
runtime tensors. ``cfg.action_space`` is the action dimension (an int, not a
placeholder: IsaacLab sizes its action buffers from it); the adapter slices
the per-env ``Box`` off IsaacLab's batched action space.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import Optional, Sequence, Tuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


@configclass
class RLGardenSceneCfg(InteractiveSceneCfg):
    """Scene shape ``RLGardenDirectRLEnv`` expects: one robot, optional camera."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = MISSING
    camera: Optional[CameraCfg] = None
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class RLGardenDirectEnvCfg(DirectRLEnvCfg):
    """``DirectRLEnvCfg`` + the action scale ``RLGardenDirectRLEnv`` applies.

    ``decimation``/``episode_length_s``/``sim``/``scene``/
    ``observation_space``/``action_space`` are still owned by the concrete
    per-task cfg, exactly as in a plain ``DirectRLEnvCfg`` subclass; ``scene``
    must be an ``RLGardenSceneCfg`` subclass.
    """

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
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        joint_pos, joint_vel = self._sample_reset_state(env_ids, joint_pos, joint_vel)

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

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
        return self.robot.data.joint_pos.torch

    def _joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch

    def _apply_joint_effort(self, target: torch.Tensor, joint_ids: list[int]) -> None:
        self.robot.set_joint_effort_target_index(target=target, joint_ids=joint_ids)

    def _camera_rgb(self) -> torch.Tensor:
        # .torch is a non-contiguous view of the camera's RGBA buffer, which
        # the next render overwrites; clone to own a contiguous copy.
        return self.camera.data.output["rgb"].torch.clone()

    # _apply_action / _get_observations / _get_rewards / _get_dones stay
    # abstract (inherited from DirectRLEnv) -- task authors implement these.
