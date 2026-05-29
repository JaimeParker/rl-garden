"""Tests for shared training example CLI argument defaults."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from rl_garden.common.cli_args import (
    LoggingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
)


def _args(script_name: str):
    module = _load_example_module(script_name)
    return module.Args()


def _load_example_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / script_name
    module_name = f"_rl_garden_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_state_sac_defaults_match_existing_cli() -> None:
    args = _args("train_sac_state.py")

    assert args.env_id == "PickCube-v1"
    assert args.control_mode == "pd_joint_delta_pos"
    assert args.total_timesteps == 1_000_000
    assert args.buffer_size == 1_000_000
    assert args.batch_size == 1024
    assert args.utd == 0.5
    assert args.gamma == 0.8


def test_rgbd_sac_defaults_match_existing_cli() -> None:
    args = _args("train_sac_rgbd.py")

    assert args.env_id == "PickCube-v1"
    assert args.obs_mode == "rgb"
    assert args.include_state is True
    assert args.camera_width == 64
    assert args.camera_height == 64
    assert args.buffer_size == 200_000
    assert args.batch_size == 512
    assert args.utd == 0.25
    assert args.encoder == "plain_conv"


def test_robotwin_sac_defaults_are_memory_safe() -> None:
    args = _args("train_sac_robotwin_rgbd.py")

    assert args.env_id == "place_shoe"
    assert args.camera_width == 64
    assert args.camera_height == 64
    assert args.buffer_device == "cuda"
    assert args.buffer_size == 100_000
    assert args.batch_size == 512
    assert args.utd == 0.25
    assert args.include_wrist_cameras is True
    assert args.reward_mode == "dense"
    assert args.control_mode == "delta_joint_pos"


def test_robotwin_sac_make_env_uses_64px_images(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module("train_sac_robotwin_rgbd.py")
    captured = {}

    def fake_make_robotwin_env(cfg):
        captured["cfg"] = cfg
        return cfg

    monkeypatch.setattr(module, "make_robotwin_env", fake_make_robotwin_env)
    args = module.Args()

    cfg = module._make_env(args, num_envs=3, is_eval=True)

    assert cfg is captured["cfg"]
    assert cfg.num_envs == 3
    assert cfg.image_size == (64, 64)
    assert cfg.include_wrist_cameras is True
    assert cfg.render_every_control_step is False
    assert cfg.control_step_cap is None
    assert cfg.random_light is False
    assert cfg.crazy_random_light_rate == 0.0
    assert cfg.head_camera_type == "D435"
    assert cfg.task_config["camera"]["head_camera_type"] == "D435"
    assert cfg.task_config["camera"]["collect_wrist_camera"] is True
    assert cfg.task_config["eval_video_log"] is True


def test_robotwin_ppo_make_env_uses_auto_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module("train_ppo_robotwin_rgbd.py")
    captured = {}

    def fake_make_robotwin_env(cfg):
        captured["cfg"] = cfg
        return cfg

    monkeypatch.setattr(module, "make_robotwin_env", fake_make_robotwin_env)
    args = module.Args()

    cfg = module._make_env(args, num_envs=2)

    assert cfg is captured["cfg"]
    assert cfg.num_envs == 2
    assert cfg.device == "auto"
    assert cfg.image_size == (64, 64)
    assert cfg.include_wrist_cameras is True
    assert cfg.render_every_control_step is False
    assert cfg.control_step_cap is None
    assert cfg.random_light is False
    assert cfg.crazy_random_light_rate == 0.0
    assert cfg.head_camera_type == "D435"
    assert cfg.task_config["camera"]["head_camera_type"] == "D435"
    assert cfg.task_config["domain_randomization"]["random_light"] is False
    assert cfg.task_config["camera"]["collect_wrist_camera"] is True


def test_robotwin_ppo_make_env_can_disable_wrist_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module("train_ppo_robotwin_rgbd.py")
    captured = {}

    def fake_make_robotwin_env(cfg):
        captured["cfg"] = cfg
        return cfg

    monkeypatch.setattr(module, "make_robotwin_env", fake_make_robotwin_env)
    args = module.Args()
    args.collect_wrist_camera = False

    cfg = module._make_env(args, num_envs=2)

    assert cfg is captured["cfg"]
    assert cfg.include_wrist_cameras is False
    assert cfg.task_config["camera"]["collect_wrist_camera"] is False


def test_robotwin_ppo_make_env_forwards_profile_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module("train_ppo_robotwin_rgbd.py")
    captured = {}

    def fake_make_robotwin_env(cfg):
        captured["cfg"] = cfg
        return cfg

    monkeypatch.setattr(module, "make_robotwin_env", fake_make_robotwin_env)
    args = module.Args()
    args.profile_timing = True
    args.profile_interval = 7
    args.render_every_control_step = True
    args.control_step_cap = 16
    args.random_light = True
    args.crazy_random_light_rate = 0.1
    args.head_camera_type = "Train_D435_128x96"

    cfg = module._make_env(args, num_envs=2)

    assert cfg is captured["cfg"]
    assert cfg.profile_timing is True
    assert cfg.profile_interval == 7
    assert cfg.render_every_control_step is True
    assert cfg.control_step_cap == 16
    assert cfg.random_light is True
    assert cfg.crazy_random_light_rate == 0.1
    assert cfg.head_camera_type == "Train_D435_128x96"
    assert cfg.task_config["render_every_control_step"] is True
    assert cfg.task_config["control_step_cap"] == 16
    assert cfg.task_config["camera"]["head_camera_type"] == "Train_D435_128x96"


def test_robotwin_ppo_make_env_can_disable_topp(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module("train_ppo_robotwin_rgbd.py")
    captured = {}

    def fake_make_robotwin_env(cfg):
        captured["cfg"] = cfg
        return cfg

    monkeypatch.setattr(module, "make_robotwin_env", fake_make_robotwin_env)
    args = module.Args()
    args.disable_topp = True

    cfg = module._make_env(args, num_envs=2)

    assert cfg is captured["cfg"]
    assert cfg.task_config["need_topp"] is False


def test_peg_sac_defaults_keep_peg_specific_overrides() -> None:
    args = _args("train_sac_rgbd_peg.py")

    assert args.env_id == "PegInsertionSidePegOnly-v1"
    assert args.control_mode == "pd_ee_delta_pose"
    assert args.sim_backend == "gpu"
    assert args.render_backend == "gpu"
    assert args.reward_mode == "normalized_dense"
    assert args.robot_uids == "panda_wristcam_gripper_closed_wo_norm"
    assert args.fix_peg_pose is False
    assert args.fix_box is True
    assert args.fixed_peg_xy == (-0.05, -0.15)
    assert args.fixed_peg_z_rot_deg == 67.5


def test_state_wsrl_defaults_match_existing_cli() -> None:
    args = _args("train_wsrl.py")

    assert args.env_id == "PickCube-v1"
    assert args.num_offline_steps == 0
    assert args.num_online_steps == 1_000_000
    assert args.buffer_size == 1_000_000
    assert args.batch_size == 256
    assert args.utd == 4.0
    assert args.gamma == 0.99
    assert args.use_cql_loss is True
    assert args.use_calql is True


def test_rgbd_wsrl_defaults_match_existing_cli() -> None:
    args = _args("train_wsrl_rgbd.py")

    assert args.obs_mode == "rgb"
    assert args.camera_width == 128
    assert args.camera_height == 128
    assert args.buffer_size == 200_000
    assert args.batch_size == 512
    assert args.utd == 0.25
    assert args.gamma == 0.99


@dataclass
class _BadPlainConvArgs:
    encoder: str = "plain_conv"
    encoder_features_dim: int = 256
    pretrained_weights: str | None = "resnet10_pretrained"
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False


def test_plain_conv_rejects_resnet_only_options() -> None:
    with pytest.raises(ValueError, match="only supported for resnet encoders"):
        image_encoder_factory_from_args(_BadPlainConvArgs())


def test_image_keys_from_obs_mode() -> None:
    assert image_keys_from_obs_mode("rgb") == ("rgb",)
    assert image_keys_from_obs_mode("rgbd") == ("rgb", "depth")


def test_log_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    args = LoggingArgs()
    monkeypatch.setenv("RLG_STD_LOG", "0")
    monkeypatch.setenv("RLG_LOG_TYPE", "none")
    monkeypatch.setenv("RLG_LOG_KEYWORDS", "train,fps")
    monkeypatch.setenv("RLG_WANDB_PROJECT", "custom-project")
    monkeypatch.setenv("RLG_WANDB_ENTITY", "custom-entity")
    monkeypatch.setenv("RLG_WANDB_GROUP", "custom-group")

    apply_log_env_overrides(args)

    assert args.std_log is False
    assert args.log_type == "none"
    assert args.log_keywords == "train,fps"
    assert args.wandb_project == "custom-project"
    assert args.wandb_entity == "custom-entity"
    assert args.wandb_group == "custom-group"
