"""AWAC offline-to-online training registration.

Builds ``Off2OnAWAC`` and reuses the same ``_runner.run_off2on`` orchestration
as ``iql``/``wsrl``/``calql``. AWAC needs no online-switch override, so this
preset mirrors ``Off2OnIQL``'s (no warmup, mixed replay retained by default).
Box observations only -- pass ``--obs_mode state`` (the ``EnvRunArgs``
default is ``rgb``).
"""
from dataclasses import dataclass
from typing import Literal

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.off2on._args import (
    AWACOff2OnTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on._registry import registry


@dataclass
class AWACOff2OnArgs(AWACOff2OnTrainingArgs, EnvBackendArgs):
    """AWAC off2on args: no warmup, mixed replay, adaptive ratio.

    Box observations only; pass ``--obs_mode state``.
    """

    warmup_steps: int = 0
    online_replay_mode: Literal["empty", "append", "mixed"] = "mixed"
    offline_data_ratio: float | str = "auto"


def build_awac(args: AWACOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnAWAC

    agent = Off2OnAWAC(
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
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
        awac_lambda=args.awac_lambda,
        exp_adv_max=args.exp_adv_max,
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
        initial_training_phase=initial_training_phase_from_args(args),
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
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_awac(args: AWACOff2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_awac, algorithm="awac")


registry.register("awac", AWACOff2OnArgs, run_awac)
