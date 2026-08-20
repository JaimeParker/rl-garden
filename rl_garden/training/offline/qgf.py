"""QGF offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineQGFArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class QGFArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineQGFArgs,
):
    """QGF (Q-Guided Flow) offline pretraining. Box (state) observations only."""


def _qgf_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    return {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "horizon_length": args.horizon_length,
        "tau": args.tau,
        "actor_lr": args.actor_lr,
        "critic_value_lr": args.critic_value_lr,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "expectile": args.expectile,
        "q_agg": args.q_agg,
        "n_critics": args.n_critics,
        "denoise_steps": args.denoise_steps,
        "t_sampling": args.t_sampling,
        "sampling_mode": args.sampling_mode,
        "guidance_weight": args.guidance_weight,
        "denoised_action_approx": args.denoised_action_approx,
        "qgrad_step_size": args.qgrad_step_size,
        "qgrad_steps": args.qgrad_steps,
        "use_sign_gradient": args.use_sign_gradient,
        "actor_num_samples": args.actor_num_samples,
        "actor_use_layer_norm": args.actor_use_layer_norm,
        "critic_use_layer_norm": args.critic_use_layer_norm,
        "value_use_layer_norm": args.value_use_layer_norm,
        "kernel_init": args.kernel_init,
        "backbone_type": args.backbone_type,
        "activation_fn": args.activation_fn,
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


def build_qgf(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import QGF
    from rl_garden.training.inspection import construct_agent

    return construct_agent(QGF, **_qgf_kwargs(args, env_spec, logger, eval_env))


def run_qgf(args: QGFArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_qgf)


registry.register("qgf", QGFArgs, run_qgf)
