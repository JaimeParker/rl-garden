"""rl-garden's ``DirectRLEnv`` authoring scaffold.

IsaacLab's ``DirectRLEnv`` (direct-implementation style) skips the
``ManagerBasedRLEnv`` per-term dispatch and expects task authors to write
fully-tensorized batch computations directly. In practice, every real Direct
task also overrides the same two boilerplate methods
(``_setup_scene``/``_reset_idx``) to do the same thing: spawn one primary
articulation (+ optionally one camera), clone/replicate the scene, and on
reset restore default joint state with a scene-origin offset (see IsaacLab's
own ``cartpole_env.py``/``cartpole_camera_env.py``).

``RLGardenDirectRLEnv`` covers exactly that common shape declaratively via
``RLGardenDirectEnvCfg.robot_cfg``/``camera_cfg``, so a task author only has
to implement the four methods that are genuinely task-specific:
``_apply_action``, ``_get_observations``, ``_get_rewards``, ``_get_dones``.
Tasks with a different scene shape (multiple objects, non-default reset
randomization beyond ``_sample_reset_state``, etc.) can still override
``_setup_scene``/``_reset_idx`` wholesale, same as any ``DirectRLEnv``
subclass.

Observation key convention: unlike IsaacLab's native tasks (which return
``{"policy": obs}``), ``_get_observations()`` here should return rl-garden's
own cross-backend keys -- ``"state"`` for the flat proprioceptive tensor,
``"rgb"``/``"depth"`` (or ``"rgb_<camera>"``/``"depth_<camera>"`` for
multi-camera tasks) for images -- so the resulting env plugs directly into
the same ``ImageFrameStackWrapper`` used for other backends. Image tensors
must be **raw pixel values** (e.g. ``camera.data.output["rgb"]`` passed
through untouched, uint8 ``[0, 255]``): do NOT apply the mean-subtraction
IsaacLab's own ``cartpole_camera_env.py`` example does for RGB. rl-garden's
image encoders decide whether to apply ``/255`` normalization from the
declared space's dtype/range (see
``rl_garden.encoders.base.image_needs_normalization``), so pre-normalized
input is silently handled wrong rather than raising an error.

``cfg.observation_space``/``cfg.action_space`` still need placeholder values
(any int is fine, e.g. ``1``) to satisfy ``DirectRLEnvCfg.validate()`` --
IsaacLab never checks the runtime obs dict against them, and rl-garden's own
adapter builds its ``gym.spaces.Dict``/``Box`` from the runtime obs/action
tensors instead.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import Optional, Sequence, Tuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass


@configclass
class RLGardenDirectEnvCfg(DirectRLEnvCfg):
    """``DirectRLEnvCfg`` + declarative scene config for ``RLGardenDirectRLEnv``.

    ``decimation``/``episode_length_s``/``scene``/``observation_space``/
    ``action_space`` are still owned by the concrete per-task cfg, exactly as
    in a plain ``DirectRLEnvCfg`` subclass -- this only adds the scene fields
    that ``RLGardenDirectRLEnv``'s generic ``_setup_scene``/``_reset_idx``
    consume.
    """

    robot_cfg: ArticulationCfg = MISSING
    camera_cfg: Optional[TiledCameraCfg] = None
    action_scale: float = 1.0


class RLGardenDirectRLEnv(DirectRLEnv):
    """Task authors override ``_apply_action``/``_get_observations``/
    ``_get_rewards``/``_get_dones``; ``_setup_scene``/``_reset_idx``/
    ``_pre_physics_step`` already have generic implementations for the
    single-articulation (+ optional single-camera) case.
    """

    cfg: RLGardenDirectEnvCfg

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        if self.cfg.camera_cfg is not None:
            self.camera = TiledCamera(self.cfg.camera_cfg)
            self.scene.sensors["camera"] = self.camera

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # CPU simulation needs collisions filtered explicitly.
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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

    # _apply_action / _get_observations / _get_rewards / _get_dones stay
    # abstract (inherited from DirectRLEnv) -- task authors implement these.
