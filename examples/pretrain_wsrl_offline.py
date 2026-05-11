"""Pure offline WSRL pretraining — no sim env, no eval.

Use this when you have a static offline dataset (e.g., real-robot teleop H5)
and want to produce a pretrained actor + critic checkpoint that you'll later
fine-tune online in a separate process (load checkpoint into a fresh WSRL
agent, then ``switch_to_online_mode(...)`` + ``agent.learn(...)``).

What this script does NOT do:
- Construct or step a ManiSkill env (no sim dependency).
- Run any evaluation episodes.
- Switch to online mode.

The observation / action spaces are inferred from the H5 dataset's first
trajectory; action bounds default to ±1 unless overridden. WSRL training
hyperparameters are inherited from ``WSRLTrainingArgs``.

Usage:
    python examples/pretrain_wsrl_offline.py \\
        --offline_dataset_path demos/real_robot.h5 \\
        --num_offline_steps 100000 \\
        --checkpoint_dir runs/real_robot_pretrain
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gymnasium import spaces
from tqdm import trange

from rl_garden.algorithms import WSRL
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    WSRLTrainingArgs,
    apply_log_env_overrides,
    resolve_checkpoint_dir,
)


@dataclass
class Args(WSRLTrainingArgs):
    # Override defaults that don't apply offline-only.
    num_offline_steps: int = 100_000
    num_online_steps: int = 0  # ignored; kept for arg-compat
    offline_dataset_path: Optional[str] = None
    buffer_device: str = "cuda"
    learning_starts: int = 0  # offline-only: training begins immediately
    capture_video: bool = False  # no env to capture

    # Spec hints — used only when H5 doesn't disambiguate.
    action_low: float = -1.0
    action_high: float = 1.0
    spec_num_envs: int = 1

    # Where to save the final pretrained checkpoint.
    save_filename: str = "offline_pretrained.pt"


class _EnvSpec:
    """Minimal stand-in for an env that only exposes spaces + num_envs.

    Mirrors the attributes ``OffPolicyAlgorithm`` actually reads (``num_envs``,
    ``single_observation_space``, ``single_action_space``). No ``reset`` /
    ``step`` — this stub must never be passed to ``agent.learn()``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        num_envs: int = 1,
    ) -> None:
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs


def _infer_specs_from_h5(
    path: str | Path,
    action_low: float,
    action_high: float,
) -> tuple[spaces.Space, spaces.Box]:
    """Peek at the H5 to derive observation + action specs.

    For state-only datasets, ``obs`` is a flat 2D array. We do not try to
    auto-detect dict observations here; vision data uses a different script.
    """
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Loading H5 datasets requires h5py. Install with `pip install h5py`."
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {path}.")
        traj = f[traj_keys[0]]
        if "obs" in traj:
            obs_node = traj["obs"]
        elif "observations" in traj:
            obs_node = traj["observations"]
        else:
            raise ValueError(f"No obs/observations field in {traj_keys[0]}.")
        if isinstance(obs_node, h5py.Group):
            raise NotImplementedError(
                "Dict observations detected. This script supports flat Box "
                "observations only; for RGBD pretraining, write a vision-specific "
                "variant that constructs a Dict observation space."
            )
        obs_shape = tuple(obs_node.shape[1:])  # strip time dim

        if "actions" in traj:
            act_node = traj["actions"]
        elif "action" in traj:
            act_node = traj["action"]
        else:
            raise ValueError(f"No actions field in {traj_keys[0]}.")
        action_shape = tuple(act_node.shape[1:])

    obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
    )
    action_space = spaces.Box(
        low=action_low, high=action_high, shape=action_shape, dtype=np.float32
    )
    return obs_space, action_space


def _build_agent(args: Args, env_spec: _EnvSpec, logger: Logger) -> WSRL:
    """Construct WSRL with the same algorithm knobs as train_wsrl.py."""
    return WSRL(
        env=env_spec,
        eval_env=None,  # no eval in offline-only pretraining
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        cql_alpha_lr=args.cql_alpha_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        use_cql_loss=args.use_cql_loss,
        cql_n_actions=args.cql_n_actions,
        cql_alpha=args.cql_alpha,
        cql_autotune_alpha=args.cql_autotune_alpha,
        cql_alpha_lagrange_init=args.cql_alpha_lagrange_init,
        cql_target_action_gap=args.cql_target_action_gap,
        cql_importance_sample=args.cql_importance_sample,
        cql_max_target_backup=args.cql_max_target_backup,
        cql_temp=args.cql_temp,
        cql_clip_diff_min=args.cql_clip_diff_min,
        cql_clip_diff_max=args.cql_clip_diff_max,
        cql_action_sample_method=args.cql_action_sample_method,
        backup_entropy=args.backup_entropy,
        use_calql=args.use_calql,
        calql_bound_random_actions=args.calql_bound_random_actions,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        critic_use_group_norm=args.critic_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        std_parameterization=args.std_parameterization,
        offline_sampling=args.offline_sampling,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=0,  # disable any eval-cadence checks
        num_eval_steps=0,
        checkpoint_dir=None,  # we save manually at the end
        checkpoint_freq=0,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=False,
    )


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    if not args.offline_dataset_path:
        raise SystemExit("--offline_dataset_path is required for offline pretraining.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")
    if args.buffer_device == "cuda" and not torch.cuda.is_available():
        print("[pretrain] CUDA not available; falling back to CPU buffer.")
        args.buffer_device = "cpu"

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"offline_pretrain__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=vars(args),
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group or "offline_pretrain",
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # 1) Infer spaces from H5 and build an env stub.
    obs_space, action_space = _infer_specs_from_h5(
        args.offline_dataset_path, args.action_low, args.action_high
    )
    if args.std_log:
        print(
            f"[pretrain] inferred obs={obs_space.shape} action={action_space.shape} "
            f"action_range=[{args.action_low}, {args.action_high}]",
            flush=True,
        )
    env_spec = _EnvSpec(obs_space, action_space, num_envs=args.spec_num_envs)

    # 2) Build agent (no env interaction; eval_env=None).
    agent = _build_agent(args, env_spec, logger)

    # 3) Load the H5 dataset into the replay buffer.
    loaded = load_maniskill_h5_to_replay_buffer(
        agent.replay_buffer,
        args.offline_dataset_path,
        num_traj=args.offline_num_traj,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
    )
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(f"[pretrain] loaded_transitions={loaded}", flush=True)

    # 4) Optionally resume from a previous offline checkpoint.
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
        if args.std_log:
            print(f"[pretrain] resumed_from={args.load_checkpoint}", flush=True)

    # 5) Pure offline gradient loop.
    gradient_steps = (
        int(agent.utd) if float(agent.utd).is_integer() and agent.utd > 1 else 1
    )
    for step in trange(args.num_offline_steps, desc="offline"):
        losses = agent.train(gradient_steps)
        global_step = step + 1
        if args.log_freq > 0 and global_step % args.log_freq == 0:
            agent._log_update_metrics(losses, global_step)
            if args.std_log:
                progress = 100.0 * global_step / args.num_offline_steps
                loss_summary = " ".join(
                    f"{k}={v:.4f}"
                    for k, v in losses.items()
                    if isinstance(v, (int, float))
                )
                print(
                    f"[offline] step={global_step}/{args.num_offline_steps} "
                    f"({progress:.2f}%) {loss_summary}",
                    flush=True,
                )
        if (
            checkpoint_dir is not None
            and args.checkpoint_freq > 0
            and global_step % args.checkpoint_freq == 0
        ):
            ckpt = agent.save(
                Path(checkpoint_dir) / f"checkpoint_{global_step}.pt",
                include_replay_buffer=args.save_replay_buffer,
            )
            if args.std_log:
                print(f"[offline] intermediate_checkpoint={ckpt}", flush=True)

    # 6) Final save.
    if checkpoint_dir is not None:
        final_path = agent.save(
            Path(checkpoint_dir) / args.save_filename,
            include_replay_buffer=args.save_replay_buffer,
        )
        logger.add_summary("offline/final_checkpoint", str(final_path))
        if args.std_log:
            print(f"[pretrain] final_checkpoint={final_path}", flush=True)
    else:
        if args.std_log:
            print(
                "[pretrain] no checkpoint_dir resolved; pass --checkpoint_dir "
                "or --save_final_checkpoint=True to keep the pretrained weights.",
                flush=True,
            )

    logger.close()


if __name__ == "__main__":
    main()
