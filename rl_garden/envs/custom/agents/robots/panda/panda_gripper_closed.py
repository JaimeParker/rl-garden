"""Panda agent variant with gripper fixed closed (0-dim gripper action space)."""

from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.registration import register_agent

from rl_garden.envs.custom.agents.controllers.fixed_gripper import (
    FixedGripperControllerConfig,
)
from rl_garden.envs.custom.agents.robots.panda.panda_wristcam import PandaWristCam


@register_agent()
class PandaGripperClosed(PandaWristCam):
    """Panda with wrist camera; gripper is always closed (no gripper in action space)."""

    uid = "panda_wristcam_gripper_closed"
    fixed_gripper_target = 0.0

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        gripper_fixed = FixedGripperControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
            fixed_target=self.fixed_gripper_target,
        )
        for config in configs.values():
            if isinstance(config, dict) and "gripper" in config:
                config["gripper"] = gripper_fixed
        return deepcopy_dict(configs)
