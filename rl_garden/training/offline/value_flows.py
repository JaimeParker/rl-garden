"""Value Flows offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gymnasium import spaces

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineValueFlowsArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class ValueFlowsArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineValueFlowsArgs,
):
    """Value Flows offline pretraining. Box or Dict (vision) observations."""


def _value_flows_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    kwargs = {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "tau": args.tau,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "alpha": args.alpha,
        "flow_steps": args.flow_steps,
        "q_agg": args.q_agg,
        "normalize_q_loss": args.normalize_q_loss,
        "n_critics": args.n_critics,
        "actor_use_layer_norm": args.actor_use_layer_norm,
        "critic_use_layer_norm": args.critic_use_layer_norm,
        "actor_use_group_norm": args.actor_use_group_norm,
        "critic_use_group_norm": args.critic_use_group_norm,
        "num_groups": args.num_groups,
        "critic_dropout_rate": args.critic_dropout_rate,
        "kernel_init": args.kernel_init,
        "backbone_type": args.backbone_type,
        "activation_fn": args.activation_fn,
        "encoder_sharing": args.encoder_sharing,
        "min_reward": args.min_reward,
        "max_reward": args.max_reward,
        "ret_agg": args.ret_agg,
        "confidence_weight_temp": args.confidence_weight_temp,
        "dcfm_lambda": args.dcfm_lambda,
        "bcfm_lambda": args.bcfm_lambda,
        "clip_flow_returns": args.clip_flow_returns,
        "num_samples": args.num_samples,
        "policy_extraction": args.policy_extraction,
        "seed": args.seed,
        "device": args.device,
        "logger": logger,
        "std_log": args.std_log,
        "log_freq": args.log_freq,
        "eval_env": eval_env,
        "eval_freq": args.eval_freq if eval_env is not None else 0,
        "num_eval_steps": args.num_eval_steps,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
    }
    if isinstance(env_spec.single_observation_space, spaces.Dict):
        from rl_garden.common.cli_args import image_encoder_factory_from_args

        kwargs["image_encoder_factory"] = image_encoder_factory_from_args(args)
    return kwargs


def build_value_flows(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import ValueFlows
    from rl_garden.training.inspection import construct_agent

    return construct_agent(ValueFlows, **_value_flows_kwargs(args, env_spec, logger, eval_env))


def run_value_flows(args: ValueFlowsArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_value_flows)


registry.register("value_flows", ValueFlowsArgs, run_value_flows)
