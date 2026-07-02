from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineActorArgs,
    OfflineCommonArgs,
    OfflineCriticArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineIQLArgs,
    OfflineQNetworkArgs,
    OfflineValueArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class IQLArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
    OfflineQNetworkArgs,
    OfflineCriticArgs,
    OfflineValueArgs,
    OfflineIQLArgs,
):
    """Implicit Q-learning offline pretraining."""


def _iql_kwargs(args: Any, env_spec: OfflineEnvSpec, logger: Logger) -> dict:
    from gymnasium import spaces

    from rl_garden.common.cli_args import image_encoder_factory_from_args
    from rl_garden.encoders import discover_image_keys

    obs_space = env_spec.single_observation_space
    kwargs = dict(
        env=env_spec,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        offline_sampling=args.offline_sampling,
        utd=args.utd,
        actor_lr=args.actor_lr,
        critic_value_lr=args.critic_value_lr,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        expectile=args.expectile,
        temperature=args.temperature,
        adv_clip_max=args.adv_clip_max,
        net_arch=args.net_arch,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        value_use_layer_norm=args.value_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        critic_use_group_norm=args.critic_use_group_norm,
        value_use_group_norm=args.value_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        value_dropout_rate=args.value_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        std_parameterization=args.std_parameterization,
        seed=args.seed,
        device=args.device,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=False,
    )
    if isinstance(obs_space, spaces.Dict):
        image_keys = discover_image_keys(obs_space)
        kwargs.update(
            image_encoder_factory=image_encoder_factory_from_args(args),
            image_keys=image_keys,
            state_key="state",
            use_proprio=args.include_state,
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=False,
        )
    return kwargs


def build_iql(args, env_spec, logger):
    from rl_garden.algorithms import IQL

    return IQL(**_iql_kwargs(args, env_spec, logger))


def run_iql(args: IQLArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_iql)


registry.register("iql", IQLArgs, run_iql)
