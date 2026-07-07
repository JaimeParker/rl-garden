"""Cal-QL offline-to-online training registration.

Builds ``Off2OnCalQL`` (a thin subclass of the shared off2on Cal-QL shell that
also backs ``WSRL``) instead of ``WSRL`` itself, and reuses the same
``_runner.run_off2on`` orchestration. This entrypoint's default preset has
no warmup, retains offline data mixed throughout online fine-tuning, and
keeps the CQL/Cal-QL regularizer online — matching Nakamoto et al. 2023
(Cal-QL) instead of the WSRL paper's warmup-then-discard preset used by the
`wsrl` entrypoint.
"""
from dataclasses import dataclass
from typing import Literal

from rl_garden.common.cli_args import (
    image_encoder_factory_from_args,
    image_keys_from_env,
    vit_sac_kwargs_from_args,
)
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.off2on._args import (
    VisionWSRLTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on._registry import registry


@dataclass
class CalQLOff2OnArgs(VisionWSRLTrainingArgs, EnvBackendArgs):
    """Cal-QL off2on args: no warmup, mixed replay, CQL retained online.

    For state obs pass --obs_mode state.
    """

    warmup_steps: int = 0
    online_replay_mode: Literal["empty", "append", "mixed"] = "mixed"
    offline_data_ratio: float | str = "auto"
    online_use_cql_loss: bool = True
    online_cql_alpha: float = 5.0  # same as cql_alpha default: "unchanged" going online


def build_calql(args: CalQLOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnCalQL

    is_visual = args.obs_mode != "state"
    image_kwargs: dict = {}
    if is_visual:
        factory = image_encoder_factory_from_args(args)
        image_keys = image_keys_from_env(env, args)
        image_kwargs = dict(
            image_keys=image_keys,
            image_encoder_factory=factory,
            image_fusion_mode=args.image_fusion_mode,
            **vit_sac_kwargs_from_args(args, image_keys),
        )

    agent = Off2OnCalQL(
        env=env,
        eval_env=eval_env,
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
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
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
        online_cql_alpha=args.online_cql_alpha,
        online_use_cql_loss=args.online_use_cql_loss,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        **image_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_calql(args: CalQLOff2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_calql, algorithm="calql")


registry.register("calql", CalQLOff2OnArgs, run_calql)
