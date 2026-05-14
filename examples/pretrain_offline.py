"""Generic offline pretraining entrypoint for flat ManiSkill H5 datasets.

Use ``--algorithm`` to choose the offline algorithm:

    python examples/pretrain_offline.py --algorithm calql --offline_dataset_path demos/demo.h5
    python examples/pretrain_offline.py --algorithm cql --offline_dataset_path demos/demo.h5
    python examples/pretrain_offline.py --algorithm wsrl --offline_dataset_path demos/demo.h5

``wsrl`` produces a WSRL agent (Cal-QL-based by design) whose checkpoint is
intended to resume through WSRL's offline→online flow on a deployment machine.
Standalone offline training should prefer ``cql`` or ``calql``.

``wsrl-calql`` is a deprecated alias for ``wsrl`` kept for backward
compatibility; selecting it prints a one-time warning and otherwise behaves
identically.
"""
from __future__ import annotations

import time
from typing import Optional, TypeAlias

import torch
import tyro

from rl_garden.algorithms import (
    CalQL,
    CQL,
    OfflineEnvSpec,
    WSRL,
    infer_box_specs_from_h5,
    run_offline_pretraining,
)
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    OfflinePretrainArgs,
    apply_log_env_overrides,
    resolve_checkpoint_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env

OfflinePretrainAgent: TypeAlias = CQL | CalQL | WSRL


def _algorithm(args: OfflinePretrainArgs) -> str:
    return args.agent if args.agent is not None else args.algorithm


def _save_filename(args: OfflinePretrainArgs, algorithm: str) -> str:
    if args.save_filename is not None:
        return args.save_filename
    return f"{algorithm.replace('-', '_')}_offline_pretrained.pt"


def _cql_kwargs(
    args: OfflinePretrainArgs, env_spec: OfflineEnvSpec, logger: Logger
) -> dict:
    return dict(
        env=env_spec,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        offline_sampling=args.offline_sampling,
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
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        ent_coef="auto",
        target_entropy="auto",
        backup_entropy=args.backup_entropy,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        use_cql_loss=args.use_cql_loss,
        use_td_loss=args.use_td_loss,
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
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=False,
    )


def _wsrl_kwargs(
    args: OfflinePretrainArgs, env_spec: OfflineEnvSpec, logger: Logger
) -> dict:
    kwargs = _cql_kwargs(args, env_spec, logger)
    kwargs.update(
        eval_env=None,
        learning_starts=0,
        training_freq=args.training_freq,
        eval_freq=0,
        num_eval_steps=0,
    )
    return kwargs


def build_offline_agent(
    args: OfflinePretrainArgs,
    env_spec: OfflineEnvSpec,
    logger: Logger,
    algorithm: str,
) -> OfflinePretrainAgent:
    common_kwargs = _cql_kwargs(args, env_spec, logger)
    if algorithm == "cql":
        return CQL(**common_kwargs)
    if algorithm == "calql":
        return CalQL(
            **common_kwargs,
            use_calql=args.use_calql,
            calql_bound_random_actions=args.calql_bound_random_actions,
            sparse_reward_mc=args.sparse_reward_mc,
            sparse_negative_reward=args.sparse_negative_reward,
            success_threshold=args.success_threshold,
        )
    if algorithm in ("wsrl", "wsrl-calql"):
        return WSRL(
            **_wsrl_kwargs(args, env_spec, logger),
            use_calql=args.use_calql,
            calql_bound_random_actions=args.calql_bound_random_actions,
            sparse_reward_mc=args.sparse_reward_mc,
            sparse_negative_reward=args.sparse_negative_reward,
            success_threshold=args.success_threshold,
        )
    raise ValueError(f"Unknown offline pretrain algorithm: {algorithm!r}")


def main(
    args_cls: type[OfflinePretrainArgs] = OfflinePretrainArgs,
    *,
    allowed_algorithms: Optional[set[str]] = None,
) -> None:
    args = tyro.cli(args_cls)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    algorithm = _algorithm(args)
    if allowed_algorithms is not None and algorithm not in allowed_algorithms:
        allowed = ", ".join(sorted(allowed_algorithms))
        raise SystemExit(f"--algorithm/--agent must be one of: {allowed}.")
    if not args.offline_dataset_path:
        raise SystemExit("--offline_dataset_path is required for offline pretraining.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")
    if algorithm == "wsrl-calql":
        import warnings as _warnings
        _warnings.warn(
            "--algorithm wsrl-calql is deprecated; use --algorithm wsrl. "
            "Both produce identical WSRL agents (Cal-QL-based by definition). "
            "The legacy alias still writes ``wsrl_calql_offline_pretrained.pt`` "
            "for backward-compatible output paths.",
            DeprecationWarning,
            stacklevel=2,
        )
    if algorithm == "cql" and args.sparse_reward_mc:
        raise SystemExit("--sparse_reward_mc requires --algorithm calql or wsrl.")
    if args.buffer_device == "cuda" and not torch.cuda.is_available():
        print("[pretrain] CUDA not available; falling back to CPU buffer.")
        args.buffer_device = "cpu"

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"{algorithm}_offline_pretrain__{args.seed}__{int(time.time())}"
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=vars(args) | {"resolved_algorithm": algorithm},
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group or f"{algorithm}_offline_pretrain",
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items())
        + f"\n|resolved_algorithm|{algorithm}|",
    )

    obs_space, action_space = infer_box_specs_from_h5(
        args.offline_dataset_path,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=args.spec_num_envs)
    if args.std_log:
        print(
            f"[pretrain] algorithm={algorithm} obs={obs_space.shape} "
            f"action={action_space.shape}",
            flush=True,
        )

    agent = build_offline_agent(args, env_spec, logger, algorithm)
    loaded = load_maniskill_h5_to_replay_buffer(
        agent.replay_buffer,
        args.offline_dataset_path,
        num_traj=args.offline_num_traj,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        success_key=args.success_key,
    )
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(f"[pretrain] loaded_transitions={loaded}", flush=True)

    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
        if args.std_log:
            print(f"[pretrain] resumed_from={args.load_checkpoint}", flush=True)

    # --- setup optional eval env ---
    eval_env = None
    if args.env_id is not None:
        eval_env = make_maniskill_env(
            ManiSkillEnvConfig(
                env_id=args.env_id,
                num_envs=args.num_eval_envs,
                control_mode=args.control_mode,
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
                reconfiguration_freq=1,
            )
        )
        agent.eval_env = eval_env
        agent.eval_freq = args.eval_freq
        agent.num_eval_steps = args.num_eval_steps

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename=_save_filename(args, algorithm),
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=agent.eval_freq if eval_env is not None else 0,
        desc=f"{algorithm}-offline",
    )

    if eval_env is not None:
        eval_env.close()
    logger.close()


if __name__ == "__main__":
    main()
