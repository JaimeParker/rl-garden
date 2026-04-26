"""Panda wrist-camera agent with fixed closed gripper and unnormalized arm actions."""

from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent

from rl_garden.envs.custom.agents.controllers.pd_ee_twist import (
    PDEETwistControllerConfig,
)
from rl_garden.envs.custom.agents.robots.panda.panda_gripper_closed import (
    PandaGripperClosed,
)


@register_agent()
class PandaGripperClosedWoNorm(PandaGripperClosed):
    """Panda with wrist camera, fixed closed gripper, and raw arm control ranges."""

    uid = "panda_wristcam_gripper_closed_wo_norm"

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        arm_pd_ee_delta_pose_raw = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_raw = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            normalize_action=False,
        )
        arm_pd_ee_twist_raw = PDEETwistControllerConfig(
            joint_names=self.arm_joint_names,
            twist_lower=-0.1,
            twist_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        configs["pd_ee_delta_pose"]["arm"] = arm_pd_ee_delta_pose_raw
        configs["pd_joint_delta_pos"]["arm"] = arm_pd_joint_delta_pos_raw
        configs["pd_ee_twist"]["arm"] = arm_pd_ee_twist_raw
        return deepcopy_dict(configs)
