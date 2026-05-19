from __future__ import annotations

from rl_garden.envs.custom.agents.controllers.pd_ee_twist import PDEETwistControllerConfig
from rl_garden.envs.custom.agents.robots.panda.panda import Panda
from rl_garden.envs.custom.agents.robots.panda.panda_gripper_closed_wo_norm import (
    PandaGripperClosedWoNorm,
)
from robot_infra.controller.simulator.maniskill import (
    ImpedanceEEDeltaPoseControllerConfig,
    ImpedanceEETwistControllerConfig,
)


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


def test_panda_exposes_impedance_control_modes():
    configs = Panda._controller_configs.fget(_PandaConfigStub())

    assert "impedance_ee_delta_pose" in configs
    assert "impedance_ee_twist" in configs
    assert isinstance(
        configs["impedance_ee_delta_pose"]["arm"],
        ImpedanceEEDeltaPoseControllerConfig,
    )
    assert isinstance(
        configs["impedance_ee_twist"]["arm"],
        ImpedanceEETwistControllerConfig,
    )


def test_wo_norm_variant_keeps_impedance_arm_actions_unnormalized():
    panda = object.__new__(PandaGripperClosedWoNorm)
    configs = PandaGripperClosedWoNorm._controller_configs.fget(panda)

    assert configs["impedance_ee_delta_pose"]["arm"].normalize_action is False
    assert configs["impedance_ee_twist"]["arm"].normalize_action is False
