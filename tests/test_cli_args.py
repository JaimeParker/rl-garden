"""Tests for shared training example CLI argument defaults."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rl_garden.common.cli_args import (
    ENCODER_REGISTRY,
    LoggingArgs,
    VisionArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_env,
    image_keys_from_obs_mode,
    vit_sac_kwargs_from_args,
)
from rl_garden.training.offline._args import OfflineVisionArgs, TDMPC2MultitaskTrainingArgs
from rl_garden.training.offline.awac import AWACArgs
from rl_garden.training.offline.td3_bc import TD3BCArgs
from rl_garden.training.off2on._args import (
    WSRLTrainingArgs,
    initial_training_phase_from_args,
    warn_if_off2on_warmup_uses_uninitialized_policy,
)
from rl_garden.training.off2on.awac import AWACOff2OnArgs
from rl_garden.training.online._args import (
    DrQv2TrainingArgs,
    FlashSACTrainingArgs,
    SACTrainingArgs,
    TDMPC2TrainingArgs,
    VisionTDMPC2TrainingArgs,
    sac_initial_training_phase_from_args,
)


def test_tdmpc2_defaults_require_single_env() -> None:
    args = TDMPC2TrainingArgs()

    assert args.num_envs == 1
    assert args.num_eval_envs == 1
    assert args.episode_length == 100
    assert args.buffer_size == 1_000_000
    assert args.horizon == 3
    assert args.num_samples == 512
    assert args.num_elites == 64
    assert args.num_pi_trajs == 24
    assert args.iterations == 6
    assert args.latent_dim == 512
    assert args.num_q == 5
    assert args.num_bins == 101
    assert args.vmin == -10.0
    assert args.vmax == 10.0
    assert args.episodic is False
    assert args.discount_denom == 5.0
    assert args.use_planner is True
    assert args.seed_steps is None


def test_vision_tdmpc2_defaults_shrink_buffer() -> None:
    args = VisionTDMPC2TrainingArgs()

    assert args.num_envs == 1
    assert args.buffer_size == 200_000
    assert args.obs_mode == "rgb"


def test_td3_bc_defaults_match_corl_trainconfig() -> None:
    args = TD3BCArgs()

    assert args.tau == 0.005
    assert args.actor_lr == 3e-4
    assert args.critic_lr == 3e-4
    assert args.policy_noise == 0.2
    assert args.noise_clip == 0.5
    assert args.policy_freq == 2
    assert args.alpha == 2.5
    assert args.n_critics == 2
    assert args.actor_use_layer_norm is False
    assert args.critic_use_layer_norm is False


def test_awac_defaults_match_corl_trainconfig() -> None:
    args = AWACArgs()

    assert args.tau == 5e-3
    assert args.actor_lr == 3e-4
    assert args.critic_lr == 3e-4
    assert args.awac_lambda == 1.0
    assert args.exp_adv_max == 100.0
    assert args.n_critics == 2


def test_awac_off2on_defaults_require_state_obs_box_scope() -> None:
    args = AWACOff2OnArgs()

    assert args.warmup_steps == 0
    assert args.online_replay_mode == "mixed"
    assert args.offline_data_ratio == "auto"
    assert args.awac_lambda == 1.0
    assert args.exp_adv_max == 100.0
    assert args.actor_lr == 3e-4
    assert args.critic_lr == 3e-4


def test_tdmpc2_multitask_defaults_require_no_env() -> None:
    args = TDMPC2MultitaskTrainingArgs()

    assert args.dataset_dir == ""
    assert args.mmap_dir == ""
    assert args.num_offline_steps == 10_000_000
    assert args.horizon == 3
    assert args.task_dim == 96
    assert args.latent_dim == 512
    assert args.num_q == 5
    assert args.num_bins == 101
    assert not hasattr(args, "env_id")
    assert not hasattr(args, "num_envs")


def test_state_sac_defaults_match_existing_cli() -> None:
    args = SACTrainingArgs()

    assert args.env_id == "PickCube-v1"
    assert args.control_mode == "pd_joint_delta_pos"
    assert args.total_timesteps == 1_000_000
    assert args.buffer_size == 1_000_000
    assert args.batch_size == 1024
    assert args.utd == 0.5
    assert args.gamma == 0.8


def test_rgbd_sac_defaults_match_existing_cli() -> None:
    from rl_garden.training.online.sac import SACArgs
    args = SACArgs()

    assert args.env_id == "PickCube-v1"
    assert args.obs_mode == "rgb"
    assert args.include_state is True
    assert args.frame_stack == 1
    assert args.camera_width == 64
    assert args.camera_height == 64
    assert args.buffer_size == 200_000
    assert args.batch_size == 512
    assert args.utd == 0.25
    assert args.nstep == 1
    assert args.encoder == "plain_conv"
    assert args.n_critics == 2
    assert args.critic_subsample_size is None
    assert args.actor_use_layer_norm is False
    assert args.critic_use_layer_norm is False
    assert args.hidden_dim == 256
    assert args.actor_hidden_layers == 3
    assert args.critic_hidden_layers == 3
    assert args.actor_log_std_min == -5.0
    assert args.actor_log_std_mode == "clamp"
    assert args.critic_only_steps == 0
    assert args.critic_only_freeze_encoder is True
    assert args.critic_only_random_action_prob == 0.0
    assert args.load_actor_checkpoint is None
    assert args.plain_conv_weight_init == "kaiming_uniform"
    assert args.plain_conv_last_act is True
    assert args.plain_conv_pooling == "flatten"
    assert args.image_augmentation == "none"
    assert args.image_random_shift_pad == 4
    assert args.q_landscape_diagnostics is False


def test_residual_sac_defaults_match_existing_cli() -> None:
    from rl_garden.training.online.residual_sac import ResidualSACArgs

    args = ResidualSACArgs()

    assert args.env_id == "PickCube-v1"
    assert args.obs_mode == "rgb"
    assert args.residual_action_scale == 0.1
    assert args.debug is False
    assert args.base_policy == "act"
    assert args.base_ckpt_path == "act-peg-only"
    assert args.base_act_temporal_agg is True
    assert args.base_act_temporal_agg_k == 0.01
    assert args.base_sac_encoder == "plain_conv"
    assert args.base_sac_encoder_features_dim == 256
    assert args.base_sac_image_fusion_mode is None
    assert args.base_sac_deterministic is True
    assert args.offline_dataset_path is None
    assert args.offline_num_traj is None
    assert args.offline_buffer_size is None
    assert args.offline_data_ratio == 0.5


def test_online_specialized_args_keep_existing_defaults() -> None:
    drq = DrQv2TrainingArgs()
    flash = FlashSACTrainingArgs()

    assert drq.obs_mode == "rgbd"
    assert drq.batch_size == 256
    assert drq.hidden_dim == 1024
    assert drq.load_replay_buffer is False
    assert flash.num_envs == 512
    assert flash.batch_size == 2048
    assert flash.capture_video is False
    assert flash.log_type == "wandb"
    assert flash.log_dir == "runs"


def test_algorithm_args_are_not_exported_from_common_cli_args() -> None:
    import rl_garden.common.cli_args as cli_args

    for name in (
        "SACTrainingArgs",
        "VisionSACTrainingArgs",
        "PPOTrainingArgs",
        "VisionPPOTrainingArgs",
        "WSRLTrainingArgs",
        "VisionWSRLTrainingArgs",
    ):
        assert not hasattr(cli_args, name)


def test_flash_sac_logging_cli_is_flat() -> None:
    from rl_garden.training.online import registry

    args = registry.parse_args(["flash_sac", "--log-type", "none"])
    assert args.log_type == "none"
    assert not hasattr(args, "logging")

    with pytest.raises(SystemExit):
        registry.parse_args(["flash_sac", "--logging.log-type", "none"])


def test_sac_critic_only_args_map_to_initial_training_phase() -> None:
    args = SACTrainingArgs(
        critic_only_steps=123,
        critic_only_freeze_encoder=True,
        critic_only_random_action_prob=0.25,
    )

    phase = sac_initial_training_phase_from_args(args)

    assert phase is not None
    assert phase.duration_steps == 123
    assert phase.update_actor is False
    assert phase.update_critic is True
    assert phase.update_encoder is False
    assert phase.random_action_prob == 0.25


def test_wsrl_warmup_args_map_to_collect_only_phase() -> None:
    phase = initial_training_phase_from_args(WSRLTrainingArgs(warmup_steps=321))

    assert phase is not None
    assert phase.duration_steps == 321
    assert phase.update_actor is False
    assert phase.update_critic is False
    assert phase.update_encoder is False


def test_wsrl_uninitialized_warmup_warning_describes_actual_behavior() -> None:
    args = WSRLTrainingArgs(warmup_steps=100, load_checkpoint=None)

    with pytest.warns(
        UserWarning,
        match="randomly initialized policy.*updates are paused",
    ):
        warn_if_off2on_warmup_uses_uninitialized_policy(args)


def test_rgbd_wsrl_pure_online_path_switches_mode() -> None:
    from rl_garden.training.off2on._runner import _switch_to_online_mode
    from rl_garden.training.off2on.wsrl import WSRLOff2OnArgs

    args = WSRLOff2OnArgs(
        num_offline_steps=0,
        warmup_steps=0,
        online_replay_mode="mixed",
        offline_data_ratio=0.25,
    )
    agent = type(
        "Agent",
        (),
        {
            "switch_to_online_mode": lambda self, **kwargs: setattr(
                self, "switch_kwargs", kwargs
            )
        },
    )()

    _switch_to_online_mode(agent, args, logger=None)

    assert agent.switch_kwargs == {
        "online_replay_mode": "mixed",
        "offline_data_ratio": 0.25,
    }


def _make_rt_req(
    rt,
    *,
    num_envs: int = 2,
    num_eval_envs: int = 2,
    capture_video: bool = False,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    from rl_garden.envs.backend_registry import EnvRequest

    return EnvRequest(
        env_id="place_shoe",
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        obs_mode="rgb",
        control_mode="delta_joint_pos",
        render_mode="rgb_array",
        seed=1,
        camera_width=64,
        camera_height=64,
        capture_video=capture_video,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        backend_config=rt,
    )


def _make_ms_req(ms):
    from rl_garden.envs.backend_registry import EnvRequest

    return EnvRequest(
        env_id="PegInsertionSidePegOnly-v1",
        num_envs=2,
        num_eval_envs=3,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        seed=1,
        camera_width=64,
        camera_height=64,
        capture_video=True,
        backend_config=ms,
    )


def test_maniskill_config_has_no_task_specific_named_fields() -> None:
    from rl_garden.common.env_args import ManiSkillConfig

    field_names = set(ManiSkillConfig.__dataclass_fields__.keys())

    assert field_names == {"sim_backend", "render_backend", "reward_mode", "env_kwargs_json"}


def test_maniskill_backend_forwards_env_kwargs_json() -> None:
    import json

    from rl_garden.common.env_args import ManiSkillConfig
    from rl_garden.envs.backends.maniskill import ManiSkillBackend

    ms = ManiSkillConfig(
        reward_mode="normalized_dense",
        env_kwargs_json=json.dumps({"robot_uids": "panda_wristcam_gripper_closed", "fix_box": True}),
    )
    req = _make_ms_req(ms)

    cfg = ManiSkillBackend._make_cfg(req, is_eval=True)

    assert cfg.reward_mode == "normalized_dense"
    assert cfg.env_kwargs == {"robot_uids": "panda_wristcam_gripper_closed", "fix_box": True}
    assert cfg.num_envs == 3
    assert cfg.save_video is True


def test_maniskill_backend_env_kwargs_json_empty_string_is_no_op() -> None:
    from rl_garden.common.env_args import ManiSkillConfig
    from rl_garden.envs.backends.maniskill import ManiSkillBackend

    ms = ManiSkillConfig(env_kwargs_json="")
    req = _make_ms_req(ms)

    cfg = ManiSkillBackend._make_cfg(req, is_eval=True)

    assert cfg.env_kwargs == {}


def _make_mujoco_req(mj, *, num_envs: int = 2, num_eval_envs: int = 3):
    from rl_garden.envs.backend_registry import EnvRequest

    return EnvRequest(
        env_id="HalfCheetah-v4",
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        obs_mode="state",
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        camera_width=None,
        camera_height=None,
        backend_config=mj,
    )


def test_mujoco_config_defaults() -> None:
    from rl_garden.common.env_args import MujocoConfig

    mj = MujocoConfig()

    assert mj.device == "cpu"
    assert mj.env_kwargs_json == "{}"
    assert mj.vectorization == "sync"


def test_mujoco_config_has_no_task_specific_named_fields() -> None:
    from rl_garden.common.env_args import MujocoConfig

    field_names = set(MujocoConfig.__dataclass_fields__.keys())

    assert field_names == {"device", "env_kwargs_json", "vectorization"}


def test_mujoco_backend_forwards_vectorization() -> None:
    from rl_garden.common.env_args import MujocoConfig
    from rl_garden.envs.backends.mujoco import MujocoBackend

    req = _make_mujoco_req(MujocoConfig(vectorization="async"))

    cfg = MujocoBackend._make_cfg(req, is_eval=False)

    assert cfg.vectorization == "async"


def test_mujoco_backend_forwards_env_kwargs_json() -> None:
    import json

    from rl_garden.common.env_args import MujocoConfig
    from rl_garden.envs.backends.mujoco import MujocoBackend

    mj = MujocoConfig(
        device="cuda:0",
        env_kwargs_json=json.dumps({"forward_reward_weight": 2.0, "reset_noise_scale": 0.05}),
    )
    req = _make_mujoco_req(mj)

    cfg = MujocoBackend._make_cfg(req, is_eval=False)

    assert cfg.device == "cuda:0"
    assert cfg.env_kwargs == {"forward_reward_weight": 2.0, "reset_noise_scale": 0.05}
    assert cfg.num_envs == 2


def test_mujoco_backend_env_kwargs_json_empty_string_is_no_op() -> None:
    from rl_garden.common.env_args import MujocoConfig
    from rl_garden.envs.backends.mujoco import MujocoBackend

    mj = MujocoConfig(env_kwargs_json="")
    req = _make_mujoco_req(mj)

    cfg = MujocoBackend._make_cfg(req, is_eval=False)

    assert cfg.env_kwargs == {}


def test_mujoco_backend_make_cfg_is_eval_swaps_num_envs() -> None:
    from rl_garden.common.env_args import MujocoConfig
    from rl_garden.envs.backends.mujoco import MujocoBackend

    req = _make_mujoco_req(MujocoConfig(), num_envs=2, num_eval_envs=3)

    train_cfg = MujocoBackend._make_cfg(req, is_eval=False)
    eval_cfg = MujocoBackend._make_cfg(req, is_eval=True)

    assert train_cfg.num_envs == 2
    assert eval_cfg.num_envs == 3


def _make_mujoco_warp_req(mjw, *, num_envs: int = 2, num_eval_envs: int = 3, obs_mode: str = "state"):
    from rl_garden.envs.backend_registry import EnvRequest

    return EnvRequest(
        env_id="RlGarden-InvertedPendulum-Warp-v0",
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        obs_mode=obs_mode,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        camera_width=64,
        camera_height=64,
        backend_config=mjw,
    )


def test_mujoco_warp_config_defaults() -> None:
    from rl_garden.common.env_args import MujocoWarpConfig

    mjw = MujocoWarpConfig()

    assert mjw.device == "cuda:0"
    assert mjw.env_kwargs_json == "{}"


def test_mujoco_warp_config_has_no_task_specific_named_fields() -> None:
    from rl_garden.common.env_args import MujocoWarpConfig

    field_names = set(MujocoWarpConfig.__dataclass_fields__.keys())

    assert field_names == {"device", "env_kwargs_json"}


def test_mujoco_warp_backend_forwards_env_kwargs_json() -> None:
    import json

    from rl_garden.common.env_args import MujocoWarpConfig
    from rl_garden.envs.backends.mujoco_warp import MujocoWarpBackend

    mjw = MujocoWarpConfig(
        device="cuda:1", env_kwargs_json=json.dumps({"frame_skip": 4})
    )
    req = _make_mujoco_warp_req(mjw)

    cfg = MujocoWarpBackend._make_cfg(req, is_eval=False)

    assert cfg.device == "cuda:1"
    assert cfg.env_kwargs == {"frame_skip": 4}
    assert cfg.num_envs == 2


def test_mujoco_warp_backend_env_kwargs_json_empty_string_is_no_op() -> None:
    from rl_garden.common.env_args import MujocoWarpConfig
    from rl_garden.envs.backends.mujoco_warp import MujocoWarpBackend

    mjw = MujocoWarpConfig(env_kwargs_json="")
    req = _make_mujoco_warp_req(mjw)

    cfg = MujocoWarpBackend._make_cfg(req, is_eval=False)

    assert cfg.env_kwargs == {}


def test_mujoco_warp_backend_make_cfg_is_eval_swaps_num_envs() -> None:
    from rl_garden.common.env_args import MujocoWarpConfig
    from rl_garden.envs.backends.mujoco_warp import MujocoWarpBackend

    req = _make_mujoco_warp_req(MujocoWarpConfig(), num_envs=2, num_eval_envs=3)

    train_cfg = MujocoWarpBackend._make_cfg(req, is_eval=False)
    eval_cfg = MujocoWarpBackend._make_cfg(req, is_eval=True)

    assert train_cfg.num_envs == 2
    assert eval_cfg.num_envs == 3


def test_mujoco_warp_backend_render_flags_follow_obs_mode() -> None:
    from rl_garden.common.env_args import MujocoWarpConfig
    from rl_garden.envs.backends.mujoco_warp import MujocoWarpBackend

    state_req = _make_mujoco_warp_req(MujocoWarpConfig(), obs_mode="state")
    rgb_req = _make_mujoco_warp_req(MujocoWarpConfig(), obs_mode="rgb")
    rgbd_req = _make_mujoco_warp_req(MujocoWarpConfig(), obs_mode="rgbd")

    state_cfg = MujocoWarpBackend._make_cfg(state_req, is_eval=False)
    rgb_cfg = MujocoWarpBackend._make_cfg(rgb_req, is_eval=False)
    rgbd_cfg = MujocoWarpBackend._make_cfg(rgbd_req, is_eval=False)

    assert state_cfg.render_rgb is False and state_cfg.render_depth is False
    assert rgb_cfg.render_rgb is True and rgb_cfg.render_depth is False
    assert rgbd_cfg.render_rgb is True and rgbd_cfg.render_depth is True


def test_robotwin_config_defaults() -> None:
    from rl_garden.common.env_args import RoboTwinConfig

    rt = RoboTwinConfig()

    assert rt.include_wrist_cameras is True
    assert rt.reward_mode == "dense"
    assert rt.step_lim == 400
    assert rt.planner_backend == "mplib"
    assert rt.embodiment == ["aloha-agilex"]
    assert rt.wrist_camera_type == "D435"
    assert rt.head_camera_type == "D435"
    assert rt.device == "auto"
    assert rt.joint_delta_scale == 0.05
    assert rt.gripper_delta_scale == 0.2
    assert rt.disable_topp is False
    assert rt.random_light is False


def test_env_backend_args_resolves_registered_config_without_algorithm_branching() -> None:
    from rl_garden.common.env_args import EnvBackendArgs

    args = EnvBackendArgs(env_backend="robotwin")

    assert args.resolve_backend_config() is args.robotwin


def test_env_backend_args_rejects_unknown_backend() -> None:
    from rl_garden.common.env_args import EnvBackendArgs

    args = EnvBackendArgs(env_backend="missing")

    with pytest.raises(KeyError, match="Available.*maniskill.*robotwin"):
        args.resolve_backend_config()


def test_robotwin_backend_make_cfg_64px() -> None:
    from rl_garden.common.env_args import RoboTwinConfig
    from rl_garden.envs.backends.robotwin import RoboTwinBackend

    rt = RoboTwinConfig()
    req = _make_rt_req(rt, num_envs=3, num_eval_envs=3, capture_video=True)

    cfg = RoboTwinBackend._make_cfg(req, is_eval=True)

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


def test_robotwin_backend_make_cfg_device() -> None:
    from rl_garden.common.env_args import RoboTwinConfig
    from rl_garden.envs.backends.robotwin import RoboTwinBackend

    rt = RoboTwinConfig()
    req = _make_rt_req(rt, num_envs=2)

    cfg = RoboTwinBackend._make_cfg(req, is_eval=False)

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


def test_robotwin_backend_disable_wrist_cameras() -> None:
    from rl_garden.common.env_args import RoboTwinConfig
    from rl_garden.envs.backends.robotwin import RoboTwinBackend

    rt = RoboTwinConfig(include_wrist_cameras=False)
    req = _make_rt_req(rt, num_envs=2)

    cfg = RoboTwinBackend._make_cfg(req, is_eval=False)

    assert cfg.include_wrist_cameras is False
    assert cfg.task_config["camera"]["collect_wrist_camera"] is False


def test_robotwin_backend_forwards_all_options() -> None:
    from rl_garden.common.env_args import RoboTwinConfig
    from rl_garden.envs.backends.robotwin import RoboTwinBackend

    rt = RoboTwinConfig(
        profile_timing=True,
        profile_interval=7,
        render_every_control_step=True,
        control_step_cap=16,
        random_light=True,
        crazy_random_light_rate=0.1,
        head_camera_type="Train_D435_128x96",
    )
    req = _make_rt_req(rt, num_envs=2, reward_scale=2.0, reward_bias=-1.0)

    cfg = RoboTwinBackend._make_cfg(req, is_eval=False)

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
    assert cfg.reward_scale == 2.0
    assert cfg.reward_bias == -1.0


def test_robotwin_backend_disable_topp() -> None:
    from rl_garden.common.env_args import RoboTwinConfig
    from rl_garden.envs.backends.robotwin import RoboTwinBackend

    rt = RoboTwinConfig(disable_topp=True)
    req = _make_rt_req(rt, num_envs=2)

    cfg = RoboTwinBackend._make_cfg(req, is_eval=False)

    assert cfg.task_config["need_topp"] is False


def test_peg_defaults_require_no_backend_config_overrides() -> None:
    """Peg's known-good config matches the vendored task class's own defaults,
    so a zero-override ManiSkillConfig must already produce them."""
    from rl_garden.common.env_args import ManiSkillConfig
    from rl_garden.envs.backends.maniskill import ManiSkillBackend

    ms = ManiSkillConfig(sim_backend="gpu", render_backend="gpu", reward_mode="normalized_dense")
    req = _make_ms_req(ms)

    cfg = ManiSkillBackend._make_cfg(req, is_eval=False)

    assert cfg.env_id == "PegInsertionSidePegOnly-v1"
    assert cfg.control_mode == "pd_ee_delta_pose"
    assert cfg.sim_backend == "gpu"
    assert cfg.render_backend == "gpu"
    assert cfg.reward_mode == "normalized_dense"
    assert cfg.env_kwargs == {}
    # robot_uids/fix_peg_pose/fix_box/fixed_peg_xy/fixed_peg_z_rot_deg are left
    # None here on purpose: PegInsertionSidePegOnly-v1's own constructor defaults
    # already reproduce the deleted residual peg scripts' behavior.
    assert cfg.robot_uids is None
    assert cfg.fix_peg_pose is None
    assert cfg.fix_box is None
    assert cfg.fixed_peg_xy is None
    assert cfg.fixed_peg_z_rot_deg is None


def test_state_wsrl_defaults_match_existing_cli() -> None:
    args = WSRLTrainingArgs()

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
    from rl_garden.training.off2on.wsrl import WSRLOff2OnArgs

    args = WSRLOff2OnArgs()

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


@dataclass
class _VitArgs:
    encoder: str = "vit"
    encoder_features_dim: int = 256
    include_state: bool = True
    image_fusion_mode: str = "stack_channels"
    vit_fusion_mode: str = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: str = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: str | None = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    plain_conv_weight_init: str = "kaiming_uniform"
    plain_conv_last_act: bool = True
    plain_conv_pooling: str = "flatten"
    image_augmentation: str = "none"
    image_random_shift_pad: int = 4


def test_plain_conv_rejects_resnet_only_options() -> None:
    with pytest.raises(ValueError, match="only supported for resnet encoders"):
        image_encoder_factory_from_args(_BadPlainConvArgs())


def test_vit_rejects_resnet_only_options() -> None:
    args = _VitArgs(pretrained_weights="resnet10_pretrained")
    with pytest.raises(ValueError, match="only supported for resnet encoders"):
        image_encoder_factory_from_args(args)


def test_non_plain_conv_rejects_plain_conv_only_options() -> None:
    args = _VitArgs(plain_conv_weight_init="orthogonal")
    with pytest.raises(ValueError, match="only supported for the plain_conv encoder"):
        image_encoder_factory_from_args(args)

    args = _VitArgs(plain_conv_last_act=False)
    with pytest.raises(ValueError, match="only supported for the plain_conv encoder"):
        image_encoder_factory_from_args(args)

    args = _VitArgs(plain_conv_pooling="gap")
    with pytest.raises(ValueError, match="only supported for the plain_conv encoder"):
        image_encoder_factory_from_args(args)

def test_vit_sac_kwargs_defaults_to_per_key() -> None:
    kwargs = vit_sac_kwargs_from_args(_VitArgs(), ("rgb_base", "rgb_wrist"))
    extractor_kwargs = kwargs["policy_kwargs"]["features_extractor_kwargs"]
    assert extractor_kwargs["fusion_mode"] == "per_key"
    assert extractor_kwargs["image_keys"] == ("rgb_base", "rgb_wrist")
    # head hyperparams live at the bundle top level, not in the extractor kwargs
    assert kwargs["actor_feature_dim"] == 128
    assert kwargs["critic_spatial_emb_dim"] == 1024
    assert "actor_feature_dim" not in extractor_kwargs
    assert "critic_spatial_emb_dim" not in extractor_kwargs


def test_encoder_registry_matches_literal() -> None:
    """Every VisionArgs.encoder choice must have exactly one registry entry."""
    from typing import get_args, get_type_hints

    literal_choices = set(get_args(get_type_hints(VisionArgs)["encoder"]))
    assert literal_choices == set(ENCODER_REGISTRY)


@pytest.mark.parametrize(
    "encoder", ["plain_conv", "resnet10", "resnet18", "drqv2_conv", "cnn3d"]
)
def test_vit_sac_kwargs_empty_for_non_vit(encoder: str) -> None:
    # Regression guard: non-vit encoders must NOT inject actor_feature_dim, which
    # crashed every non-vit RGBD run when forwarded to SACPolicy. Empty bundle ->
    # the algorithm constructor falls back to actor_feature_dim=None.
    kwargs = vit_sac_kwargs_from_args(VisionArgs(encoder=encoder), ("rgb",))
    assert kwargs == {}
    assert "actor_feature_dim" not in kwargs


def test_image_encoder_factory_returns_callable_for_each_encoder() -> None:
    for encoder in ENCODER_REGISTRY:
        factory = image_encoder_factory_from_args(VisionArgs(encoder=encoder))
        assert callable(factory)


def test_offline_plain_conv_factory_uses_shared_defaults() -> None:
    factory = image_encoder_factory_from_args(OfflineVisionArgs(encoder="plain_conv"))
    assert callable(factory)


def test_image_keys_from_obs_mode() -> None:
    assert image_keys_from_obs_mode("rgb") == ("rgb",)
    assert image_keys_from_obs_mode("rgbd") == ("rgb", "depth")


def test_image_keys_from_env_can_filter_explicit_per_camera_keys() -> None:
    from gymnasium import spaces

    class _Env:
        single_observation_space = spaces.Dict(
            {
                "rgb_base_camera": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype="uint8"
                ),
                "rgb_hand_camera": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype="uint8"
                ),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype="float32"),
            }
        )

    args = VisionArgs(per_camera_rgbd=True, image_keys="rgb_base_camera")

    assert image_keys_from_env(_Env(), args) == ("rgb_base_camera",)


def test_image_keys_from_env_rejects_missing_explicit_key() -> None:
    from gymnasium import spaces

    class _Env:
        single_observation_space = spaces.Dict(
            {
                "rgb_base_camera": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype="uint8"
                ),
            }
        )

    args = VisionArgs(per_camera_rgbd=True, image_keys="rgb_missing")

    with pytest.raises(ValueError, match="rgb_missing"):
        image_keys_from_env(_Env(), args)


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
