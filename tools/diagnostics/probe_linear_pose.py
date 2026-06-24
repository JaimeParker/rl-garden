"""Diagnostic probe: linear-probe R^2 for object pose from a frozen visual
SAC encoder's features.

Rolls out a trained ``final.pt`` vision SAC checkpoint, freezes the
``CombinedExtractor``, and fits ridge regressions from per-camera image
features (and the proprio branch, as a positive control) to ground-truth
cubeA/cubeB positions and TCP-relative offsets (read from sim state).
Reports held-out R^2 per (feature set, target).

cubeA/cubeB positions are expressed relative to the robot base pose to
remove the per-env GPU-parallelization scene offset; tcp_minus_cubeA/B are
frame-invariant by construction.

Example:
    python tools/diagnostics/probe_linear_pose.py \
      --checkpoint_path runs/<run>/checkpoints/final.pt \
      --num_envs 16 --num_steps 1000
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from rl_garden.algorithms import SAC
from rl_garden.common import seed_everything
from rl_garden.common.utils import get_device
from rl_garden.encoders.combined import default_image_encoder_factory, discover_image_keys
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class ProbeArgs:
    checkpoint_path: str
    env_id: str = "StackCube-v1"
    control_mode: str = "pd_joint_delta_pos"
    num_envs: int = 16
    num_steps: int = 1000
    seed: int = 1
    device: str = "auto"
    deterministic: bool = False
    encoder_features_dim: int = 256
    train_frac: float = 0.8
    ridge_alpha: float = 1.0


def _make_env(args: ProbeArgs):
    cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb",
        include_state=True,
        control_mode=args.control_mode,
        reward_mode="normalized_dense",
        sim_backend="gpu",
        render_backend="gpu",
        reconfiguration_freq=1,
        camera_width=64,
        camera_height=64,
        per_camera_rgbd=True,
    )
    return make_maniskill_env(cfg)


def _make_agent(args: ProbeArgs, env, device: torch.device) -> SAC:
    image_keys = discover_image_keys(env.single_observation_space)
    factory = default_image_encoder_factory(features_dim=args.encoder_features_dim)
    return SAC(
        env=env,
        eval_env=env,
        buffer_size=max(args.num_envs, 1),
        buffer_device="cpu",
        learning_starts=0,
        batch_size=1,
        training_freq=1,
        seed=args.seed,
        device=device,
        logger=None,
        std_log=False,
        eval_freq=0,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_final_checkpoint=False,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode="per_key",
    )


def _ridge_r2(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    x_mean, x_std = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True) + 1e-8
    y_mean, y_std = y_train.mean(0, keepdim=True), y_train.std(0, keepdim=True) + 1e-8

    xtr = (x_train - x_mean) / x_std
    xte = (x_test - x_mean) / x_std
    ytr = (y_train - y_mean) / y_std
    yte = (y_test - y_mean) / y_std

    d = xtr.shape[1]
    a = xtr.T @ xtr + alpha * torch.eye(d, dtype=xtr.dtype)
    b = xtr.T @ ytr
    w = torch.linalg.solve(a, b)

    pred = xte @ w
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def main() -> None:
    args = tyro.cli(ProbeArgs)
    seed_everything(args.seed)
    device = get_device(args.device)

    env = _make_env(args)
    try:
        agent = _make_agent(args, env, device)
        agent.load(
            args.checkpoint_path, strict=True, load_replay_buffer=False, load_optimizers=False
        )
        agent.policy.eval()

        extractor = agent.policy.features_extractor
        unwrapped = env.unwrapped

        obs, _ = env.reset(seed=args.seed)

        base_feats, hand_feats, proprio_feats = [], [], []
        rel_cubeA, rel_cubeB, rel_tcp, tcp_minus_cubeA, tcp_minus_cubeB = [], [], [], [], []

        with torch.no_grad():
            for _ in range(args.num_steps):
                cubeA_p = unwrapped.cubeA.pose.p
                cubeB_p = unwrapped.cubeB.pose.p
                tcp_p = unwrapped.agent.tcp.pose.p
                base_p = unwrapped.agent.robot.pose.p

                img_feats = extractor._encode_images(obs, stop_gradient=True)
                proprio = extractor._encode_proprio(obs[extractor.state_key])

                base_feats.append(img_feats[0].cpu())
                hand_feats.append(img_feats[1].cpu())
                proprio_feats.append(proprio.cpu())
                rel_cubeA.append((cubeA_p - base_p).cpu())
                rel_cubeB.append((cubeB_p - base_p).cpu())
                rel_tcp.append((tcp_p - base_p).cpu())
                tcp_minus_cubeA.append((tcp_p - cubeA_p).cpu())
                tcp_minus_cubeB.append((tcp_p - cubeB_p).cpu())

                action = agent.policy.predict(obs, deterministic=args.deterministic)
                obs, _, _, _, _ = env.step(action)

        base_x = torch.cat(base_feats).double()
        hand_x = torch.cat(hand_feats).double()
        combined_x = torch.cat([base_x, hand_x], dim=-1)
        proprio_x = torch.cat(proprio_feats).double()

        feature_sets = {
            "base_camera": base_x,
            "hand_camera": hand_x,
            "combined_image": combined_x,
            "proprio": proprio_x,
        }
        targets = {
            "rel_cubeA": torch.cat(rel_cubeA).double(),
            "rel_cubeB": torch.cat(rel_cubeB).double(),
            "tcp_minus_cubeA": torch.cat(tcp_minus_cubeA).double(),
            "tcp_minus_cubeB": torch.cat(tcp_minus_cubeB).double(),
            "rel_tcp": torch.cat(rel_tcp).double(),
        }

        n = base_x.shape[0]
        n_train = int(n * args.train_frac)
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=g)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        print(f"checkpoint: {args.checkpoint_path}")
        print(
            f"samples: {n} (train={len(train_idx)}, test={len(test_idx)}), "
            f"deterministic={args.deterministic}"
        )
        header = f"{'feature_set':<16}" + "".join(f"{name:>18}" for name in targets)
        print(header)
        for fname, x in feature_sets.items():
            row = f"{fname:<16}"
            for tname, y in targets.items():
                r2 = _ridge_r2(
                    x[train_idx], y[train_idx], x[test_idx], y[test_idx], args.ridge_alpha
                )
                row += f"{float(r2.mean()):>18.3f}"
            print(row)
    finally:
        env.close()


if __name__ == "__main__":
    main()
