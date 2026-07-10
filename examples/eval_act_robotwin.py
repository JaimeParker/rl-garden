"""Evaluate a frozen ACT base policy directly on RoboTwin.

This diagnostic entrypoint checks whether the ACT base policy can solve a task
under the exact RoboTwin environment configuration used by residual SAC.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch

from rl_garden.common import Logger
from rl_garden.common.env_args import RoboTwinConfig
from rl_garden.envs.backend_registry import EnvRequest, make_evaluation_env
from rl_garden.policies.base_policies import make_base_policy


@dataclass
class EvalACTRoboTwinArgs:
    env_id: str = "place_empty_cup"
    obs_mode: Literal["rgb"] = "rgb"
    control_mode: Literal["delta_joint_pos", "ee_delta_pose", "joint_pos"] = "ee_delta_pose"
    base_ckpt_path: str = "pretrained_models/place_empty_cup.ckpt"
    base_act_stats_path: Optional[str] = None
    base_act_temporal_agg: bool = True
    base_act_temporal_agg_k: float = 0.01
    base_act_image_width: Optional[int] = None
    base_act_image_height: Optional[int] = None
    num_eval_envs: int = 1
    num_eval_episodes: int = 10
    seed: int = 1
    camera_width: int = 320
    camera_height: int = 240
    include_state: bool = True
    capture_video: bool = True
    video_fps: int = 30
    diagnostic_video: bool = True
    action_diagnostics: bool = True
    diagnostic_steps: int = 20
    diagnostic_output_path: Optional[str] = None
    eval_output_dir: Optional[str] = None
    device: str = "auto"
    log_type: Literal["wandb", "none"] = "none"
    exp_name: Optional[str] = None
    wandb_project: str = "rl-garden"
    wandb_entity: Optional[str] = None
    robotwin: RoboTwinConfig = field(default_factory=RoboTwinConfig)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args() -> EvalACTRoboTwinArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="place_empty_cup")
    parser.add_argument("--obs-mode", default="rgb", choices=["rgb"])
    parser.add_argument(
        "--control-mode",
        default="ee_delta_pose",
        choices=["delta_joint_pos", "ee_delta_pose", "joint_pos"],
    )
    parser.add_argument("--base-ckpt-path", default="pretrained_models/place_empty_cup.ckpt")
    parser.add_argument("--base-act-stats-path", default=None)
    parser.add_argument(
        "--base-act-temporal-agg",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
    )
    parser.add_argument("--base-act-temporal-agg-k", type=float, default=0.01)
    parser.add_argument("--base-act-image-width", type=int, default=None)
    parser.add_argument("--base-act-image-height", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--include-state", type=_str_to_bool, default=True)
    parser.add_argument("--capture-video", type=_str_to_bool, default=True)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--diagnostic-video", type=_str_to_bool, default=True)
    parser.add_argument("--action-diagnostics", type=_str_to_bool, default=True)
    parser.add_argument("--diagnostic-steps", type=int, default=20)
    parser.add_argument("--diagnostic-output-path", default=None)
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-type", choices=["wandb", "none"], default="none")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--wandb-project", default="rl-garden")
    parser.add_argument("--wandb-entity", default=None)

    parser.add_argument("--robotwin.robotwin-root", dest="robotwin_root", default=None)
    parser.add_argument("--robotwin.assets-path", dest="assets_path", default=None)
    parser.add_argument("--robotwin.seeds-path", dest="seeds_path", default=None)
    parser.add_argument("--robotwin.step-lim", dest="step_lim", type=int, default=200)
    parser.add_argument("--robotwin.planner-backend", dest="planner_backend", default="mplib")
    parser.add_argument("--robotwin.embodiment", dest="embodiment", nargs="+", default=["aloha-agilex"])
    parser.add_argument("--robotwin.reward-mode", dest="reward_mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--robotwin.head-camera-type", dest="head_camera_type", default="D435")
    parser.add_argument("--robotwin.wrist-camera-type", dest="wrist_camera_type", default="D435")
    parser.add_argument("--robotwin.control-step-cap", dest="control_step_cap", type=int, default=None)
    parser.add_argument("--robotwin.random-background", dest="random_background", type=_str_to_bool, default=True)
    parser.add_argument("--robotwin.cluttered-table", dest="cluttered_table", type=_str_to_bool, default=True)
    parser.add_argument("--robotwin.clean-background-rate", dest="clean_background_rate", type=float, default=0.02)
    parser.add_argument("--robotwin.random-head-camera-dis", dest="random_head_camera_dis", type=float, default=0.0)
    parser.add_argument("--robotwin.random-table-height", dest="random_table_height", type=float, default=0.03)
    parser.add_argument("--robotwin.joint-delta-scale", dest="joint_delta_scale", type=float, default=0.05)
    parser.add_argument("--robotwin.gripper-delta-scale", dest="gripper_delta_scale", type=float, default=0.2)
    parser.add_argument("--robotwin.ee-delta-pos-scale", dest="ee_delta_pos_scale", type=float, default=0.03)
    parser.add_argument("--robotwin.ee-delta-rot-scale", dest="ee_delta_rot_scale", type=float, default=0.15)
    parser.add_argument("--robotwin.profile-timing", dest="profile_timing", action="store_true")
    parser.add_argument("--robotwin.profile-interval", dest="profile_interval", type=int, default=100)
    parser.add_argument("--robotwin.no-include-wrist-cameras", dest="include_wrist_cameras", action="store_false")
    parser.add_argument("--robotwin.include-wrist-cameras", dest="include_wrist_cameras", action="store_true")
    parser.set_defaults(include_wrist_cameras=True)

    ns = parser.parse_args()
    if (ns.base_act_image_width is None) != (ns.base_act_image_height is None):
        parser.error(
            "--base-act-image-width and --base-act-image-height must be provided together."
        )
    robotwin = RoboTwinConfig(
        include_wrist_cameras=ns.include_wrist_cameras,
        head_camera_type=ns.head_camera_type,
        wrist_camera_type=ns.wrist_camera_type,
        control_step_cap=ns.control_step_cap,
        random_background=ns.random_background,
        cluttered_table=ns.cluttered_table,
        clean_background_rate=ns.clean_background_rate,
        random_head_camera_dis=ns.random_head_camera_dis,
        random_table_height=ns.random_table_height,
        profile_timing=ns.profile_timing,
        profile_interval=ns.profile_interval,
        robotwin_root=ns.robotwin_root,
        assets_path=ns.assets_path,
        seeds_path=ns.seeds_path,
        step_lim=ns.step_lim,
        planner_backend=ns.planner_backend,
        embodiment=ns.embodiment,
        reward_mode=ns.reward_mode,
        joint_delta_scale=ns.joint_delta_scale,
        gripper_delta_scale=ns.gripper_delta_scale,
        ee_delta_pos_scale=ns.ee_delta_pos_scale,
        ee_delta_rot_scale=ns.ee_delta_rot_scale,
    )
    return EvalACTRoboTwinArgs(
        env_id=ns.env_id,
        obs_mode=ns.obs_mode,
        control_mode=ns.control_mode,
        base_ckpt_path=ns.base_ckpt_path,
        base_act_stats_path=ns.base_act_stats_path,
        base_act_temporal_agg=ns.base_act_temporal_agg,
        base_act_temporal_agg_k=ns.base_act_temporal_agg_k,
        base_act_image_width=ns.base_act_image_width,
        base_act_image_height=ns.base_act_image_height,
        num_eval_envs=ns.num_eval_envs,
        num_eval_episodes=ns.num_eval_episodes,
        seed=ns.seed,
        camera_width=ns.camera_width,
        camera_height=ns.camera_height,
        include_state=ns.include_state,
        capture_video=ns.capture_video,
        video_fps=ns.video_fps,
        diagnostic_video=ns.diagnostic_video,
        action_diagnostics=ns.action_diagnostics,
        diagnostic_steps=ns.diagnostic_steps,
        diagnostic_output_path=ns.diagnostic_output_path,
        eval_output_dir=ns.eval_output_dir,
        device=ns.device,
        log_type=ns.log_type,
        exp_name=ns.exp_name,
        wandb_project=ns.wandb_project,
        wandb_entity=ns.wandb_entity,
        robotwin=robotwin,
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _base_act_image_size(args: EvalACTRoboTwinArgs) -> tuple[int, int]:
    if args.base_act_image_width is not None and args.base_act_image_height is not None:
        return (args.base_act_image_height, args.base_act_image_width)
    return (args.camera_height, args.camera_width)


def _as_bool_tensor(value: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(num_envs)


def _record_dir(args: EvalACTRoboTwinArgs) -> Optional[str]:
    if not args.capture_video:
        return None
    if args.eval_output_dir is not None:
        return args.eval_output_dir
    ckpt = Path(args.base_ckpt_path)
    return str(ckpt.parent / "act_robotwin_eval_videos")


def _diagnostic_video_path(args: EvalACTRoboTwinArgs) -> Optional[Path]:
    record_dir = _record_dir(args)
    if not args.capture_video or not args.diagnostic_video or record_dir is None:
        return None
    return Path(record_dir) / "rl_garden_diagnostic_multicam.mp4"


def _diagnostic_output_path(args: EvalACTRoboTwinArgs) -> Optional[Path]:
    if not args.action_diagnostics:
        return None
    if args.diagnostic_output_path is not None:
        return Path(args.diagnostic_output_path)
    output_dir = args.eval_output_dir
    if output_dir is None:
        output_dir = str(Path(args.base_ckpt_path).parent / "act_robotwin_eval_videos")
    return Path(output_dir) / "rl_garden_action_diagnostics.jsonl"


def _to_uint8_image(value: torch.Tensor) -> np.ndarray:
    image = value.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)


def _obs_diagnostic_frame(obs: dict[str, torch.Tensor]) -> Optional[np.ndarray]:
    frames = []
    for key in ("rgb", "rgb_left_wrist", "rgb_right_wrist"):
        value = obs.get(key)
        if value is not None:
            frames.append(_to_uint8_image(value))
    if not frames:
        return None
    height = min(frame.shape[0] for frame in frames)
    resized = []
    for frame in frames:
        if frame.shape[0] != height:
            from PIL import Image

            width = int(round(frame.shape[1] * height / frame.shape[0]))
            frame = np.asarray(Image.fromarray(frame).resize((width, height)))
        resized.append(frame)
    return np.concatenate(resized, axis=1)


def _write_diagnostic_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Writing diagnostic videos requires imageio. Install imageio or run "
            "with --diagnostic-video false."
        ) from exc
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _tensor_first_row(value: Optional[torch.Tensor]) -> Optional[list[float]]:
    if value is None:
        return None
    row = value.detach().float().cpu()
    if row.ndim == 0:
        return [float(row.item())]
    if row.ndim > 1:
        row = row.reshape(row.shape[0], -1)[0]
    return [float(x) for x in row.tolist()]


def _split_action_summary(action: torch.Tensor, control_mode: str) -> dict[str, float]:
    action = action.detach().float().cpu().reshape(-1)
    if action.numel() != 14:
        return {}
    if control_mode in {"joint_pos", "delta_joint_pos"}:
        parts = {
            "left_arm": action[:6],
            "left_gripper": action[6:7],
            "right_arm": action[7:13],
            "right_gripper": action[13:14],
        }
    else:
        parts = {
            "left_xyz": action[:3],
            "left_rotvec": action[3:6],
            "left_gripper": action[6:7],
            "right_xyz": action[7:10],
            "right_rotvec": action[10:13],
            "right_gripper": action[13:14],
        }
    summary: dict[str, float] = {}
    for name, values in parts.items():
        summary[f"{name}_mean"] = float(values.mean().item())
        summary[f"{name}_abs_max"] = float(values.abs().max().item())
    return summary


def _diagnostic_record(
    *,
    global_step: int,
    control_mode: str,
    action: torch.Tensor,
    state_before: Optional[torch.Tensor],
    state_after: Optional[torch.Tensor],
    reward: torch.Tensor,
    success: torch.Tensor,
    done: torch.Tensor,
) -> dict[str, Any]:
    action_cpu = action.detach().float().cpu()
    record: dict[str, Any] = {
        "step": int(global_step),
        "control_mode": control_mode,
        "action_shape": list(action_cpu.shape),
        "action_min": float(action_cpu.min().item()),
        "action_max": float(action_cpu.max().item()),
        "action_mean": float(action_cpu.mean().item()),
        "action_std": float(action_cpu.std(unbiased=False).item()),
        "action_abs_mean": float(action_cpu.abs().mean().item()),
        "action_abs_max": float(action_cpu.abs().max().item()),
        "action_first": _tensor_first_row(action_cpu),
        "reward_first": float(reward.detach().float().cpu().reshape(-1)[0].item()),
        "success_first": bool(success.detach().cpu().reshape(-1)[0].item()),
        "done_first": bool(done.detach().cpu().reshape(-1)[0].item()),
    }
    record.update(_split_action_summary(action_cpu.reshape(-1, action_cpu.shape[-1])[0], control_mode))
    if state_before is not None and state_after is not None:
        before = state_before.detach().float().cpu()
        after = state_after.detach().float().cpu()
        delta = after.reshape(after.shape[0], -1) - before.reshape(before.shape[0], -1)
        record["state_before_first"] = _tensor_first_row(before)
        record["state_after_first"] = _tensor_first_row(after)
        record["state_delta_norm_first"] = float(delta[0].norm().item())
        record["state_delta_abs_max_first"] = float(delta[0].abs().max().item())
    return record


def _write_action_diagnostics(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _env_request(args: EvalACTRoboTwinArgs) -> EnvRequest:
    robotwin = args.robotwin
    robotwin.device = args.device
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        include_state=args.include_state,
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=_record_dir(args),
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=robotwin.step_lim,
        backend_config=robotwin,
    )


def _scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().item())


def evaluate(args: EvalACTRoboTwinArgs, logger: Logger) -> dict[str, float]:
    if args.num_eval_envs <= 0:
        raise ValueError(f"num_eval_envs must be positive, got {args.num_eval_envs}.")
    if args.num_eval_episodes <= 0:
        raise ValueError(
            f"num_eval_episodes must be positive, got {args.num_eval_episodes}."
        )

    device = _device(args.device)
    env = make_evaluation_env("robotwin", _env_request(args))
    provider = make_base_policy(
        base_policy="act",
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env=env,
        base_ckpt_path=args.base_ckpt_path,
        base_act_stats_path=args.base_act_stats_path,
        device=device,
        base_act_temporal_agg=args.base_act_temporal_agg,
        base_act_temporal_agg_k=args.base_act_temporal_agg_k,
        base_act_image_size=_base_act_image_size(args),
    )
    provider.eval()

    completed = 0
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    reward_values: list[float] = []
    action_abs_means: list[float] = []
    action_abs_maxes: list[float] = []
    diagnostic_frames: list[np.ndarray] = []
    diagnostic_path = _diagnostic_video_path(args)
    action_diagnostic_path = _diagnostic_output_path(args)
    action_diagnostic_records: list[dict[str, Any]] = []
    global_step = 0

    try:
        obs, _ = env.reset(seed=args.seed)
        frame = _obs_diagnostic_frame(obs)
        if frame is not None:
            diagnostic_frames.append(frame)
        provider.reset()
        episode_returns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        while completed < args.num_eval_episodes:
            state_before = obs.get("state") if isinstance(obs, dict) else None
            with torch.no_grad():
                actions = provider.select_action(obs).actions.detach()
            action_abs_means.append(float(actions.abs().mean().detach().cpu().item()))
            action_abs_maxes.append(float(actions.abs().max().detach().cpu().item()))
            obs, rewards, terms, truncs, infos = env.step(actions)
            state_after = obs.get("state") if isinstance(obs, dict) else None
            frame = _obs_diagnostic_frame(obs)
            if frame is not None:
                diagnostic_frames.append(frame)

            rewards = rewards.reshape(env.num_envs)
            reward_values.extend(float(x) for x in rewards.detach().cpu().tolist())
            episode_returns += rewards
            episode_lengths += 1
            done = (terms | truncs).reshape(env.num_envs)
            success = _as_bool_tensor(infos.get("success"), env.num_envs, env.device)
            if (
                args.action_diagnostics
                and len(action_diagnostic_records) < args.diagnostic_steps
            ):
                record = _diagnostic_record(
                    global_step=global_step,
                    control_mode=args.control_mode,
                    action=actions,
                    state_before=state_before,
                    state_after=state_after,
                    reward=rewards,
                    success=success,
                    done=done,
                )
                action_diagnostic_records.append(record)
                print(
                    "[act-diagnostic] "
                    f"step={record['step']} control={record['control_mode']} "
                    f"action_abs_mean={record['action_abs_mean']:.4g} "
                    f"action_abs_max={record['action_abs_max']:.4g} "
                    f"state_delta_norm={record.get('state_delta_norm_first', float('nan')):.4g} "
                    f"reward={record['reward_first']:.4g} "
                    f"success={int(record['success_first'])}",
                    flush=True,
                )
            global_step += 1

            logger.add_scalar(
                "eval/step_reward",
                float(rewards.float().mean().cpu().item()),
                global_step,
            )
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for key, value in final_info["episode"].items():
                    logger.add_scalar(
                        f"eval/{key}",
                        float(value[done_mask].float().mean().cpu().item()),
                        global_step,
                    )
            if done.any():
                done_ids = torch.where(done)[0]
                for env_id in done_ids.detach().cpu().tolist():
                    returns.append(_scalar(episode_returns[env_id]))
                    lengths.append(int(episode_lengths[env_id].detach().cpu().item()))
                    successes.append(bool(success[env_id].detach().cpu().item()))
                    completed += 1
                    if completed >= args.num_eval_episodes:
                        break
                provider.reset(env_ids=done_ids)
                episode_returns[done] = 0
                episode_lengths[done] = 0
    finally:
        env.close()
        if diagnostic_path is not None:
            _write_diagnostic_video(diagnostic_path, diagnostic_frames, args.video_fps)
        if action_diagnostic_path is not None:
            _write_action_diagnostics(action_diagnostic_path, action_diagnostic_records)

    reward_t = torch.as_tensor(reward_values, dtype=torch.float32)
    return_t = torch.as_tensor(returns, dtype=torch.float32)
    length_t = torch.as_tensor(lengths, dtype=torch.float32)
    return {
        "episodes": float(len(successes)),
        "success_rate": float(sum(successes) / len(successes)),
        "return_mean": float(return_t.mean().item()),
        "return_std": float(return_t.std(unbiased=False).item()),
        "length_mean": float(length_t.mean().item()),
        "reward_mean": float(reward_t.mean().item()),
        "reward_std": float(reward_t.std(unbiased=False).item()),
        "reward_min": float(reward_t.min().item()),
        "reward_max": float(reward_t.max().item()),
        "reward_positive_fraction": float((reward_t > 0).float().mean().item()),
        "action_abs_mean": float(torch.as_tensor(action_abs_means).mean().item()),
        "action_abs_max": float(torch.as_tensor(action_abs_maxes).max().item()),
    }


def main() -> None:
    args = parse_args()
    run_name = args.exp_name or f"act_only_{args.env_id}_s{args.seed}_{args.num_eval_episodes}ep"
    logger = Logger.create(
        log_type=args.log_type,
        log_dir="runs",
        run_name=run_name,
        config=asdict(args),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.env_id,
    )
    try:
        metrics = evaluate(args, logger)
        for key, value in metrics.items():
            print(f"{key}: {value:.6g}", flush=True)
            logger.add_summary(f"eval/{key}", value)
        if logger.wandb_run is not None:
            print(f"wandb_url: {logger.wandb_run.url}", flush=True)
        if args.capture_video:
            print(f"video_dir: {_record_dir(args)}", flush=True)
            if args.diagnostic_video:
                print(f"diagnostic_video: {_diagnostic_video_path(args)}", flush=True)
        if args.action_diagnostics:
            print(f"action_diagnostics: {_diagnostic_output_path(args)}", flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
