from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineActorArgs,
    OfflineCommonArgs,
    OfflineCompileArgs,
    OfflineCQLArgs,
    OfflineCriticArgs,
    OfflineDiscountArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class CQLArgs(
    OfflineCommonArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
    OfflineCriticArgs,
    OfflineCompileArgs,
    OfflineCQLArgs,
):
    """Conservative Q-learning offline pretraining."""


def _cql_kwargs(args: Any, env_spec: OfflineEnvSpec, logger: Logger) -> dict:
    from gymnasium import spaces

    from rl_garden.common.cli_args import (
        image_encoder_factory_from_args,
        vit_sac_kwargs_from_args,
    )
    from rl_garden.encoders.combined import discover_image_keys

    kwargs = dict(
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
    if isinstance(env_spec.single_observation_space, spaces.Dict):
        image_keys = discover_image_keys(env_spec.single_observation_space)
        kwargs.update(
            image_encoder_factory=image_encoder_factory_from_args(args),
            image_keys=image_keys,
            state_key="state",
            use_proprio=args.include_state,
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=False,
            **vit_sac_kwargs_from_args(args, image_keys),
        )
    return kwargs


def build_cql(args, env_spec, logger):
    from rl_garden.algorithms import CQL

    return CQL(**_cql_kwargs(args, env_spec, logger))


def run_cql(args: CQLArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_cql)


registry.register("cql", CQLArgs, run_cql)
