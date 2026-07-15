"""TD3-BC offline pretraining registration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineTD3BCArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class TD3BCArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineTD3BCArgs,
):
    """TD3-BC offline pretraining. Box observations only."""


def _td3_bc_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    return dict(
        env=env_spec,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        batch_size=args.batch_size,
        gamma=args.gamma,
        offline_sampling=args.offline_sampling,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        n_critics=args.n_critics,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        critic_use_group_norm=args.critic_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        seed=args.seed,
        device=args.device,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_env=eval_env,
        eval_freq=args.eval_freq if eval_env is not None else 0,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=False,
    )


def build_td3_bc(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import TD3BC

    return TD3BC(**_td3_bc_kwargs(args, env_spec, logger, eval_env))


def run_td3_bc(args: TD3BCArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_td3_bc)


registry.register("td3_bc", TD3BCArgs, run_td3_bc)
