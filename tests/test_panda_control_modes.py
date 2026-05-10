from __future__ import annotations

from rl_garden.envs.custom.agents.controllers.pd_ee_twist import PDEETwistControllerConfig
from rl_garden.envs.custom.agents.robots.panda.panda import Panda


class _PandaConfigStub:
    arm_joint_names = Panda.arm_joint_names
    gripper_joint_names = Panda.gripper_joint_names
    ee_link_name = Panda.ee_link_name
    arm_stiffness = Panda.arm_stiffness
    arm_damping = Panda.arm_damping
    arm_force_limit = Panda.arm_force_limit
    gripper_stiffness = Panda.gripper_stiffness
    gripper_damping = Panda.gripper_damping
    gripper_force_limit = Panda.gripper_force_limit
    urdf_path = Panda.urdf_path


def test_panda_exposes_twist_control_instead_of_pose_control():
    configs = Panda._controller_configs.fget(_PandaConfigStub())

    assert "pd_ee_pose" not in configs
    assert "pd_ee_twist" in configs
    assert isinstance(configs["pd_ee_twist"]["arm"], PDEETwistControllerConfig)