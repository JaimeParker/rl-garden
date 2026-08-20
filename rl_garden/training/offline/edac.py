"""EDAC offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineEDACArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class EDACArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineEDACArgs,
):
    """EDAC offline pretraining. Box observations only."""

    gamma: float = 0.99


def _edac_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    return {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "offline_sampling": args.offline_sampling,
        "eta": args.eta,
        "policy_lr": args.policy_lr,
        "q_lr": args.q_lr,
        "alpha_lr": args.alpha_lr,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "ent_coef": args.ent_coef,
        "target_entropy": args.target_entropy,
        "net_arch": list(args.net_arch),
        "n_critics": args.n_critics,
        "critic_subsample_size": args.critic_subsample_size,
        "seed": args.seed,
        "device": args.device,
        "logger": logger,
        "std_log": args.std_log,
        "log_freq": args.log_freq,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
    }


def build_edac(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import EDAC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(EDAC, **_edac_kwargs(args, env_spec, logger, eval_env))


def run_edac(args: EDACArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_edac)


registry.register("edac", EDACArgs, run_edac)
