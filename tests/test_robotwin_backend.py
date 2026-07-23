from __future__ import annotations

from rl_garden.common.env_args import RoboTwinConfig
from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.robotwin import RoboTwinBackend


def _request(control_mode: str) -> EnvRequest:
    return EnvRequest(
        env_id="place_empty_cup",
        num_envs=1,
        num_eval_envs=1,
        obs_mode="rgb",
        control_mode=control_mode,
        render_mode="rgb_array",
        seed=1,
        camera_width=640,
        camera_height=480,
        backend_config=RoboTwinConfig(),
    )


def test_robotwin_backend_derives_absolute_ee_pose_action_dim() -> None:
    cfg = RoboTwinBackend._make_cfg(_request("ee_pose"), is_eval=True)

    assert cfg.control_mode == "ee_pose"
    assert cfg.action_dim == 14


def test_robotwin_backend_keeps_joint_action_dim() -> None:
    cfg = RoboTwinBackend._make_cfg(_request("joint_pos"), is_eval=True)

    assert cfg.control_mode == "joint_pos"
    assert cfg.action_dim == 14
