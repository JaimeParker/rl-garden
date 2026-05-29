"""Evaluate a ResidualSAC RGBD policy on PegInsertionSidePegOnly-v1.

The script records a custom side-by-side video from the per-camera RGB
observations used by the policy:

    [base_camera | hand_camera]

Example:
    CUDA_VISIBLE_DEVICES=0 python examples/eval_residual_sac_rgbd_peg.py \
      --checkpoint_path runs/<run>/checkpoints/final.pt \
      --base_policy act \
      --base_ckpt_path act-peg-only \
      --num_eval_envs 16 \
      --num_eval_steps 100
"""
from __future__ import annotations

import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro

from rl_garden.algorithms import ResidualSAC
from rl_garden.common import seed_everything
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.cli_args import image_encoder_factory_from_args, image_keys_from_env
from rl_garden.common.utils import get_device
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env
from rl_garden.policies.base_policies import make_base_policy


@dataclass
class EvalResidualRGBDPegArgs:
    checkpoint_path: str
    env_id: str = "PegInsertionSidePegOnly-v1"
    num_eval_envs: int = 16
    num_eval_steps: int = 100
    seed: int = 1
    device: str = "auto"
    buffer_device: str = "cpu"

    obs_mode: str = "rgb"
    include_state: bool = True
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "per_key"
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    per_camera_rgbd: bool = True

    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: str = "normalized_dense"
    robot_uids: str = "panda_wristcam_gripper_closed"
    fix_peg_pose: bool = False
    fix_box: bool = True
    fixed_peg_xy: tuple[float, float] = (-0.05, -0.15)
    fixed_peg_z_rot_deg: float = 67.5

    residual_action_scale: float = 0.1
    base_policy: Literal["act", "sac", "zero"] = "act"
    base_ckpt_path: Optional[str] = "act-peg-only"
    base_act_temporal_agg: bool = True
    base_act_temporal_agg_k: float = 0.01
    base_sac_encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    base_sac_encoder_features_dim: int = 256
    base_sac_image_fusion_mode: Optional[Literal["stack_channels", "per_key"]] = None
    base_sac_deterministic: bool = True

    output_dir: Optional[str] = None
    video_name: Optional[str] = None
    video_fps: int = 30
    video_codec: str = "auto"
    video_scale: int = 4
    save_video: bool = True
    video_env_index: int = 0
    base_camera_key: str = "rgb_base_camera"
    wrist_camera_key: str = "rgb_hand_camera"

    strict: bool = True


class FFMpegVideoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        fps: int,
        frame_shape: tuple[int, int, int],
        codec: str = "auto",
    ) -> None:
        height, width, channels = frame_shape
        if channels != 3:
            raise ValueError(f"Expected RGB frames with 3 channels, got shape {frame_shape}.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg, codec = _ffmpeg_executable_and_codec(codec)
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed.")
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        return_code = self._proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {self.path}")


def _ffmpeg_candidates() -> list[str]:
    candidates: list[str] = []
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        candidates.append(ffmpeg)
    try:
        import imageio_ffmpeg
    except ImportError:
        pass
    else:
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    return candidates or ["ffmpeg"]


def _supports_encoder(ffmpeg: str, codec: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return codec in result.stdout


def _ffmpeg_executable_and_codec(requested_codec: str) -> tuple[str, str]:
    candidates = _ffmpeg_candidates()
    if requested_codec != "auto":
        for ffmpeg in candidates:
            if _supports_encoder(ffmpeg, requested_codec):
                return ffmpeg, requested_codec
        return candidates[0], requested_codec

    for codec in ("libx264", "libopenh264", "h264_nvenc", "mpeg4"):
        for ffmpeg in candidates:
            if _supports_encoder(ffmpeg, codec):
                return ffmpeg, codec
    return candidates[0], "mpeg4"


def _to_numpy_rgb(value, env_index: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4:
        array = array[env_index]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape HxWxC or NxHxWxC, got {array.shape}.")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    pad = height - frame.shape[0]
    if pad <= 0:
        return frame
    before = pad // 2
    after = pad - before
    return np.pad(frame, ((before, after), (0, 0), (0, 0)), mode="constant")


def _scale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"video_scale must be positive, got {scale}.")
    if scale == 1:
        return frame
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def combined_camera_frame(
    obs: dict,
    *,
    base_key: str,
    wrist_key: str,
    env_index: int,
    scale: int = 1,
) -> np.ndarray:
    missing = [key for key in (base_key, wrist_key) if key not in obs]
    if missing:
        raise KeyError(
            f"Observation is missing camera key(s) {missing}. Available keys: {sorted(obs)}"
        )
    base = _to_numpy_rgb(obs[base_key], env_index)
    wrist = _to_numpy_rgb(obs[wrist_key], env_index)
    height = max(base.shape[0], wrist.shape[0])
    frame = np.concatenate(
        [_pad_to_height(base, height), _pad_to_height(wrist, height)],
        axis=1,
    )
    return _scale_frame(frame, scale)


def _make_eval_env(args: EvalResidualRGBDPegArgs):
    cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        fix_peg_pose=args.fix_peg_pose,
        fix_box=args.fix_box,
        fixed_peg_xy=args.fixed_peg_xy,
        fixed_peg_z_rot_deg=args.fixed_peg_z_rot_deg,
        reconfiguration_freq=1,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        render_mode="rgb_array",
        per_camera_rgbd=args.per_camera_rgbd,
    )
    return make_maniskill_env(cfg)


def _make_base_action_provider(args: EvalResidualRGBDPegArgs, env):
    provider = make_base_policy(
        base_policy=args.base_policy,
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env=env,
        base_ckpt_path=args.base_ckpt_path,
        base_act_temporal_agg=args.base_act_temporal_agg,
        base_act_temporal_agg_k=args.base_act_temporal_agg_k,
        base_sac_encoder=args.base_sac_encoder,
        base_sac_encoder_features_dim=args.base_sac_encoder_features_dim,
        base_sac_image_fusion_mode=args.base_sac_image_fusion_mode,
        base_sac_deterministic=args.base_sac_deterministic,
    )
    if args.base_policy == "act":
        print(
            "[eval_residual] base_policy=act "
            f"ckpt={provider.checkpoint_path} "
            f"state_dim={provider.spec.state_dim} "
            f"action_dim={provider.spec.action_dim} "
            f"num_queries={provider.config.num_queries}",
            flush=True,
        )
    elif args.base_policy == "sac":
        print(
            "[eval_residual] base_policy=sac "
            f"ckpt={args.base_ckpt_path} "
            f"deterministic={args.base_sac_deterministic}",
            flush=True,
        )
    else:
        print("[eval_residual] base_policy=zero", flush=True)
    return provider


def _make_agent(args: EvalResidualRGBDPegArgs, env, device: torch.device) -> ResidualSAC:
    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_env(env, args)
    base_action_provider = _make_base_action_provider(args, env)
    return ResidualSAC(
        env=env,
        eval_env=env,
        base_action_provider=base_action_provider,
        residual_action_scale=args.residual_action_scale,
        buffer_size=max(args.num_eval_envs, 1),
        buffer_device=args.buffer_device,
        learning_starts=0,
        batch_size=1,
        training_freq=1,
        seed=args.seed,
        device=device,
        logger=None,
        std_log=False,
        eval_freq=0,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_final_checkpoint=False,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
    )


def _policy_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state", {})
    policy_state = state.get("policy", {})
    if not isinstance(policy_state, dict):
        raise TypeError("Checkpoint state does not contain a policy state_dict.")
    return policy_state


def _infer_resnet_name(policy_state: dict[str, torch.Tensor]) -> Optional[str]:
    block_ids: set[int] = set()
    for key in policy_state:
        marker = ".blocks."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        block = suffix.split(".", 1)[0]
        if block.isdigit():
            block_ids.add(int(block))
    if not block_ids:
        return None
    block_count = max(block_ids) + 1
    if block_count == 4:
        return "resnet10"
    if block_count == 8:
        return "resnet18"
    if block_count == 16:
        raise ValueError(
            "Checkpoint uses ResNet-34 (16 residual blocks), which is not a supported "
            "base SAC encoder. Supported: resnet10 (4 blocks), resnet18 (8 blocks)."
        )
    raise ValueError(f"Cannot infer ResNet architecture from {block_count} residual blocks.")


def _infer_encoder_features_dim(policy_state: dict[str, torch.Tensor]) -> Optional[int]:
    suffix = ".bottleneck.0.bias"
    for key, tensor in policy_state.items():
        if key.endswith(suffix):
            return int(tensor.shape[0])
    suffix = ".fc.0.0.bias"
    for key, tensor in policy_state.items():
        if key.endswith(suffix):
            return int(tensor.shape[0])
    return None


def _apply_checkpoint_config(args: EvalResidualRGBDPegArgs, checkpoint: dict) -> None:
    hparams = checkpoint.get("metadata", {}).get("hyperparameters", {})
    policy_state = _policy_state_dict(checkpoint)

    inferred_encoder = _infer_resnet_name(policy_state)
    if inferred_encoder is not None and args.encoder == "plain_conv":
        args.encoder = inferred_encoder  # type: ignore[assignment]
    if inferred_encoder is None:
        has_plain_conv = any(".cnn." in key for key in policy_state)
        if has_plain_conv:
            args.encoder = "plain_conv"

    inferred_features_dim = _infer_encoder_features_dim(policy_state)
    if inferred_features_dim is not None:
        args.encoder_features_dim = inferred_features_dim

    if "image_fusion_mode" in hparams:
        args.image_fusion_mode = hparams["image_fusion_mode"]
    if "residual_action_scale" in hparams:
        args.residual_action_scale = float(hparams["residual_action_scale"])

    print(
        "[eval_residual] policy_config "
        f"encoder={args.encoder} "
        f"encoder_features_dim={args.encoder_features_dim} "
        f"image_fusion_mode={args.image_fusion_mode} "
        f"residual_action_scale={args.residual_action_scale}",
        flush=True,
    )


def _default_output_dir(args: EvalResidualRGBDPegArgs) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    checkpoint = Path(args.checkpoint_path)
    return checkpoint.parent / "eval_residual_videos"


def _video_path(args: EvalResidualRGBDPegArgs) -> Path:
    name = args.video_name
    if name is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        name = f"{args.env_id}_residual_eval_{stamp}.mp4"
    return _default_output_dir(args) / name


def _append_final_metrics(metrics: dict[str, list[torch.Tensor]], infos) -> None:
    if "final_info" not in infos:
        return
    episode = infos["final_info"].get("episode", {})
    for key, value in episode.items():
        metrics[key].append(value)


def _summarize_metrics(metrics: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, values in metrics.items():
        out[key] = float(torch.stack(values).float().mean().item())
    return out


def evaluate(args: EvalResidualRGBDPegArgs) -> dict[str, float]:
    seed_everything(args.seed)
    device = get_device(args.device)
    if not 0 <= args.video_env_index < args.num_eval_envs:
        raise ValueError(
            f"video_env_index must be in [0, {args.num_eval_envs}), "
            f"got {args.video_env_index}."
        )
    checkpoint = load_checkpoint_file(args.checkpoint_path, map_location="cpu")
    _apply_checkpoint_config(args, checkpoint)
    eval_env = _make_eval_env(args)
    writer: Optional[FFMpegVideoWriter] = None
    video_path: Optional[Path] = None

    try:
        agent = _make_agent(args, eval_env, device)
        agent.load(
            args.checkpoint_path,
            strict=args.strict,
            load_replay_buffer=False,
            load_optimizers=False,
        )
        agent.policy.eval()
        agent.base_action_provider.bind_env(eval_env)
        agent.base_action_provider.reset()

        obs, _ = eval_env.reset(seed=args.seed)
        if args.save_video:
            frame = combined_camera_frame(
                obs,
                base_key=args.base_camera_key,
                wrist_key=args.wrist_camera_key,
                env_index=args.video_env_index,
                scale=args.video_scale,
            )
            video_path = _video_path(args)
            writer = FFMpegVideoWriter(
                video_path,
                fps=args.video_fps,
                frame_shape=frame.shape,
                codec=args.video_codec,
            )
            writer.write(frame)

        metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True, return_info=False)
                obs, _, _, _, infos = eval_env.step(action)
            _append_final_metrics(metrics, infos)
            if writer is not None:
                writer.write(
                    combined_camera_frame(
                        obs,
                        base_key=args.base_camera_key,
                        wrist_key=args.wrist_camera_key,
                        env_index=args.video_env_index,
                        scale=args.video_scale,
                    )
                )

        result = _summarize_metrics(metrics)
        print("\n=== ResidualSAC Evaluation ===", flush=True)
        for key in ("success_at_end", "success_once", "return"):
            if key in result:
                print(f"{key}: {result[key]:.4f}", flush=True)
        for key, value in sorted(result.items()):
            if key not in {"success_at_end", "success_once", "return"}:
                print(f"{key}: {value:.4f}", flush=True)
        if video_path is not None:
            print(f"video: {video_path}", flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        eval_env.close()


def main() -> None:
    args = tyro.cli(EvalResidualRGBDPegArgs)
    evaluate(args)


if __name__ == "__main__":
    main()
