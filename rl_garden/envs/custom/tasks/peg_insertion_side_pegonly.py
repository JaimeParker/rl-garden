from typing import Dict, Optional

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from rl_garden.envs.custom.tasks.peg_insertion_side import (
    PegInsertionSideEnv,
    _build_box_with_hole,
    _build_pose_axes,
)
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.utils import common
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from rl_garden.envs.custom.agents.robots.panda.panda_gripper_closed import (
    PandaGripperClosed,
)


def _compute_reach_pose_and_qpos(env, env_id: int) -> Optional[np.ndarray]:
    """Compute reach_pose (same as peg_insertion_side solution) and solve IK for qpos.
    Returns qpos with gripper closed (0), or None if IK fails.
    """
    try:
        import mplib
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )
    except ImportError:
        return None

    base_env = env.unwrapped
    FINGER_LENGTH = 0.025

    # Same as peg_insertion_side solution: get grasp_pose then reach_pose.
    # Use the per-env peg actor (instead of merged actor[0]) so each env has its own IK target.
    peg_entity = base_env.peg._objs[env_id]
    peg_component = peg_entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if peg_component is None:
        return None
    mesh = get_component_mesh(peg_component, to_world_frame=True)
    if mesh is None:
        return None
    obb = mesh.bounding_box_oriented
    approaching = np.array([0, 0, -1])
    tcp_link = sapien_utils.get_obj_by_name(
        base_env.agent.robot.get_links(), "panda_hand_tcp"
    )
    target_closing = (
        tcp_link.pose.to_transformation_matrix()[env_id, :3, 1].cpu().numpy()
    )

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = base_env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose(
        [-max(0.05, base_env.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0]
    )
    grasp_pose = grasp_pose * offset

    # Reach pose is the grasp pose itself (no extra offset).
    reach_pose = grasp_pose * sapien.Pose([0, 0, 0])

    # Solve IK via mplib (base pose same as PegInsertionSide: [-0.615, 0, 0])
    agent = base_env.agent
    base_pose = np.array([-0.615, 0, 0, 1, 0, 0, 0], dtype=np.float64)

    link_names = [link.get_name() for link in agent.robot.get_links()]
    joint_names = [joint.get_name() for joint in agent.robot.get_active_joints()]
    planner = mplib.Planner(
        urdf=agent.urdf_path,
        srdf=agent.urdf_path.replace(".urdf", ".srdf"),
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="panda_hand_tcp",
    )
    planner.set_base_pose(base_pose)

    target_pose = np.concatenate([reach_pose.p, reach_pose.q])
    qpos_now = base_env.agent.robot.get_qpos().cpu().numpy()[env_id]
    result = planner.plan_screw(
        target_pose,
        qpos_now,
        time_step=base_env.control_timestep,
        use_point_cloud=False,
    )

    if result["status"] != "Success":
        return None

    planned_qpos = np.array(result["position"][-1], dtype=np.float32)
    qpos_now_full = base_env.agent.robot.get_qpos().cpu().numpy()[env_id]
    n_dof = len(qpos_now_full)
    if len(planned_qpos) < n_dof:
        target_qpos = qpos_now_full.astype(np.float32).copy()
        target_qpos[: len(planned_qpos)] = planned_qpos
    else:
        target_qpos = planned_qpos[:n_dof].copy()
    target_qpos[-2:] = 0.01 # gripper closed
    return target_qpos


@register_env("PegInsertionSidePegOnly-v1", max_episode_steps=100)
class PegInsertionSidePegOnlyEnv(PegInsertionSideEnv):
    """Variant of PegInsertionSideEnv where the robot starts at reach pose above the peg.

    - Hole is fixed at center of box (in _load_scene).
    - Peg pose comes from parent env logic and can optionally be fixed with fix_peg_pose=True.
    - Box pose comes from parent env logic, or is fixed when fix_box=True.
    - Robot is placed at reach_pose (computed like peg_insertion_side solution) via IK.
    - Gripper is closed and fixed (no gripper in action space).
    """

    SUPPORTED_ROBOTS = [
        "panda_wristcam_gripper_closed",
        "panda_wristcam_gripper_closed_wo_norm",
    ]

    # Fixed box pose when fix_box=True (center of parent's random range)
    _fixed_box_xy = (0.0, 0.3)
    _fixed_box_z_rot = np.pi / 2
    _fixed_peg_xy = (-0.05, -0.15)
    _fixed_peg_z_rot = np.pi / 2 - np.pi / 8
    # _fixed_peg_xy = (0, -0.15)
    # _fixed_peg_z_rot = np.pi / 2

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam_gripper_closed",
        fix_box: bool = True,
        fix_peg_pose: bool = False,
        debug_pose_vis: bool = False,
        peg_density: float = 1000.0, ## default 1000
        **kwargs,
    ):
        self.fix_box = fix_box  # set before super() since parent's __init__ calls reset()
        self.fix_peg_pose = fix_peg_pose
        self.peg_density = peg_density
        super().__init__(
            *args,
            robot_uids=robot_uids,
            debug_pose_vis=debug_pose_vis,
            **kwargs,
        )

    def _set_fixed_peg_pose(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            peg_xy = torch.tensor(
                [self._fixed_peg_xy[0], self._fixed_peg_xy[1]],
                device=self.device,
                dtype=torch.float32,
            ).expand(b, 2)
            peg_pos = torch.zeros((b, 3), device=self.device)
            peg_pos[:, :2] = peg_xy
            peg_pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            angle = self._fixed_peg_z_rot
            peg_quat = torch.zeros((b, 4), device=self.device)
            peg_quat[:, 0] = np.cos(angle / 2)
            peg_quat[:, 3] = np.sin(angle / 2)
            self.peg.set_pose(Pose.create_from_pq(peg_pos, peg_quat))

    def _set_fixed_box_pose(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            box_xy = torch.tensor(
                [self._fixed_box_xy[0], self._fixed_box_xy[1]],
                device=self.device,
                dtype=torch.float32,
            ).expand(b, 2)
            box_pos = torch.zeros((b, 3), device=self.device)
            box_pos[:, :2] = box_xy
            box_pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            # Quaternion (w,x,y,z) for z-axis rotation: w=cos(θ/2), z=sin(θ/2)
            angle = self._fixed_box_z_rot
            box_quat = torch.zeros((b, 4), device=self.device)
            box_quat[:, 0] = np.cos(angle / 2)
            box_quat[:, 3] = np.sin(angle / 2)
            self.box.set_pose(Pose.create_from_pq(box_pos, box_quat))

    def _apply_fixed_scene_poses(self, env_idx: torch.Tensor):
        if self.fix_peg_pose:
            self._set_fixed_peg_pose(env_idx)
        if self.fix_box:
            self._set_fixed_box_pose(env_idx)

    # Camera config for visualization (modify eye/target to change viewing angle)
    _human_render_camera_eye = [0., -0.8, 0.1]
    _human_render_camera_target = [0, 0, 0.1]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            self._human_render_camera_eye,
            self._human_render_camera_target,
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        """Override to fix hole at center of box (centers=0)."""
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = np.full(self.num_envs, 0.1, dtype=np.float32)
            radii = np.full(self.num_envs, 0.02, dtype=np.float32)
            # Hole fixed at center of box (parent randomizes centers)
            centers = np.zeros((len(lengths), 2))

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            pegs = []
            boxes = []
            if self.debug_pose_vis:
                peg_head_pose_visuals = []
                box_hole_pose_visuals = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius], density=self.peg_density)
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)
                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)

                if self.debug_pose_vis:
                    builder = _build_pose_axes(self.scene)
                    builder.set_scene_idxs(scene_idxs)
                    peg_head_pose_vis = builder.build_kinematic(f"peg_head_pose_vis_{i}")
                    self.remove_from_state_dict_registry(peg_head_pose_vis)
                    peg_head_pose_visuals.append(peg_head_pose_vis)

                    builder = _build_pose_axes(self.scene, axis_half_length=0.08)
                    builder.set_scene_idxs(scene_idxs)
                    box_hole_pose_vis = builder.build_kinematic(f"box_hole_pose_vis_{i}")
                    self.remove_from_state_dict_registry(box_hole_pose_vis)
                    box_hole_pose_visuals.append(box_hole_pose_vis)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")
            if self.debug_pose_vis:
                self.peg_head_pose_vis = Actor.merge(
                    peg_head_pose_visuals, "peg_head_pose_vis"
                )
                self.box_hole_pose_vis = Actor.merge(
                    box_hole_pose_visuals, "box_hole_pose_vis"
                )
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        # Peg and box from parent (random), only overwrite robot
        super()._initialize_episode(env_idx, options)
        self._apply_fixed_scene_poses(env_idx)

        # Overwrite robot: per-env reach_pose via IK, retry by reinitializing failed env only.
        max_ik_retries = 10
        env_ids = [int(i) for i in env_idx.detach().cpu().tolist()]
        qpos_all = self.agent.robot.get_qpos().cpu().numpy().copy()

        fallback_qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                -np.pi / 4,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        failed_envs = []
        for env_id in env_ids:
            target_qpos = None
            for attempt in range(1, max_ik_retries + 1):
                target_qpos = _compute_reach_pose_and_qpos(self, env_id)
                if target_qpos is not None:
                    break

                if attempt < max_ik_retries:
                    single_env_idx = torch.tensor(
                        [env_id], device=env_idx.device, dtype=env_idx.dtype
                    )
                    super()._initialize_episode(single_env_idx, options)
                    self._apply_fixed_scene_poses(single_env_idx)

            if target_qpos is None:
                failed_envs.append(env_id)
                qpos_all[env_id] = fallback_qpos
            else:
                qpos_all[env_id] = target_qpos

        self.agent.robot.set_qpos(qpos_all)
        if failed_envs:
            print(
                "Fallback: IK failed for env indices "
                f"{failed_envs} after {max_ik_retries} attempts"
            )

        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
