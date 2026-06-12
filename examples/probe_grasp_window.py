"""Diagnostic probe: does a StackCube SAC policy ever enter the cubeA grasp
window, and if so does it hold or progress toward placing?

Rolls out a ``final.pt`` checkpoint (stochastic actions, matching the
training rollout policy by default) and records, for each 50-step episode,
the per-step TCP-cubeA distance, gripper finger opening, and the env's
``evaluate()`` flags (``is_cubeA_grasped``, ``is_cubeA_on_cubeB``,
``success``).

Example:
    python examples/probe_grasp_window.py \
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
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_delta_pos"
    num_envs: int = 16
    num_steps: int = 1000
    seed: int = 1
    device: str = "auto"
    deterministic: bool = False
    encoder_features_dim: int = 256


def _make_env(args: ProbeArgs):
    if args.obs_mode == "state":
        cfg = ManiSkillEnvConfig(
            env_id=args.env_id,
            num_envs=args.num_envs,
            obs_mode="state",
            control_mode=args.control_mode,
            sim_backend="gpu",
            render_backend="gpu",
            reconfiguration_freq=1,
        )
    else:
        cfg = ManiSkillEnvConfig(
            env_id=args.env_id,
            num_envs=args.num_envs,
            obs_mode=args.obs_mode,
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
    kwargs: dict = dict(
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
    )
    if args.obs_mode != "state":
        image_keys = discover_image_keys(env.single_observation_space)
        factory = default_image_encoder_factory(features_dim=args.encoder_features_dim)
        kwargs.update(
            image_keys=image_keys,
            image_encoder_factory=factory,
            image_fusion_mode="per_key",
        )
    return SAC(**kwargs)


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

        unwrapped = env.unwrapped
        gripper_width = float(unwrapped.agent.robot.get_qlimits()[0, -1, 1] * 2)

        obs, _ = env.reset(seed=args.seed)

        dists, grips, graspeds, on_bs, succs = [], [], [], [], []
        for _ in range(args.num_steps):
            dist = torch.linalg.norm(
                unwrapped.agent.tcp.pose.p - unwrapped.cubeA.pose.p, dim=-1
            )
            grip = unwrapped.agent.robot.get_qpos()[:, -2:].sum(dim=-1) / gripper_width

            with torch.no_grad():
                action = agent.policy.predict(obs, deterministic=args.deterministic)
                obs, _, term, trunc, infos = env.step(action)

            done = torch.logical_or(term, trunc)
            src = infos["final_info"] if done.any() else infos

            dists.append(dist.cpu())
            grips.append(grip.cpu())
            graspeds.append(src["is_cubeA_grasped"].cpu())
            on_bs.append(src["is_cubeA_on_cubeB"].cpu())
            succs.append(src["success"].cpu())

        dist_t = torch.stack(dists)
        grip_t = torch.stack(grips)
        grasped_t = torch.stack(graspeds)
        on_b_t = torch.stack(on_bs)
        succ_t = torch.stack(succs)

        steps, num_envs = dist_t.shape
        episode_len = 50
        assert steps % episode_len == 0, f"num_steps must be a multiple of {episode_len}"
        n_ep = steps // episode_len

        dist_e = dist_t.view(n_ep, episode_len, num_envs)
        grip_e = grip_t.view(n_ep, episode_len, num_envs)
        grasped_e = grasped_t.view(n_ep, episode_len, num_envs)
        on_b_e = on_b_t.view(n_ep, episode_len, num_envs)
        succ_e = succ_t.view(n_ep, episode_len, num_envs)

        ever_grasped = grasped_e.any(dim=1)
        grasp_steps = grasped_e.sum(dim=1).float()
        ever_on_b = on_b_e.any(dim=1)
        ever_succ = succ_e.any(dim=1)
        min_dist = dist_e.min(dim=1).values
        closest_idx = dist_e.argmin(dim=1)
        grip_at_closest = torch.gather(grip_e, 1, closest_idx.unsqueeze(1)).squeeze(1)

        n_total = n_ep * num_envs
        n_grasped = int(ever_grasped.sum())

        print(f"checkpoint: {args.checkpoint_path}")
        print(
            f"episodes: {n_total} (envs={num_envs}, episodes/env={n_ep}), "
            f"deterministic={args.deterministic}"
        )
        print(
            f"min_dist_to_cubeA[m]: mean={min_dist.mean():.4f} "
            f"median={min_dist.median():.4f} p10={min_dist.quantile(0.1):.4f} "
            f"min={min_dist.min():.4f}"
        )
        print(f"ever_grasped: {n_grasped}/{n_total} ({100 * n_grasped / n_total:.1f}%)")
        if n_grasped > 0:
            gs = grasp_steps[ever_grasped]
            print(f"  grasp duration [steps]: mean={gs.mean():.2f} max={gs.max():.0f}")
            on_b_given_grasp = ever_on_b[ever_grasped]
            print(
                f"  on_cubeB given grasped: {int(on_b_given_grasp.sum())}/{n_grasped} "
                f"({100 * float(on_b_given_grasp.float().mean()):.1f}%)"
            )
        print(f"ever_on_cubeB (any episode): {int(ever_on_b.sum())}/{n_total}")
        print(f"ever_success: {int(ever_succ.sum())}/{n_total}")
        print(
            f"gripper_opening_at_min_dist [1=open,0=closed]: "
            f"mean={grip_at_closest.mean():.3f}"
        )
        if n_grasped > 0:
            print(f"  grasped episodes:     mean={grip_at_closest[ever_grasped].mean():.3f}")
        if n_grasped < n_total:
            print(f"  non-grasped episodes: mean={grip_at_closest[~ever_grasped].mean():.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
