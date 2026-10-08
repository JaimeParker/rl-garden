"""Standard FQL offline-to-online registration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.networks import Activation, KernelInit
from rl_garden.training.off2on._args import Off2OnCommonArgs
from rl_garden.training.off2on._registry import registry


@dataclass
class FQLOff2OnArgs(Off2OnCommonArgs, ObservationArgs, EnvBackendArgs):
    """Standard FQL with an uninterrupted offline-to-online continuation."""

    alpha: float = 10.0
    flow_steps: int = 10
    q_agg: Literal["mean", "min"] = "mean"
    normalize_q_loss: bool = False
    hidden_dim: int = 512
    hidden_layers: int = 4
    activation_fn: Optional[Activation] = "gelu"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    n_critics: int = 2
    actor_use_layer_norm: bool = False
    kernel_init: Optional[KernelInit] = "xavier_uniform"
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    num_eval_episodes: Optional[int] = None
    reference_compatible: bool = False
    learning_starts: int = 0
    training_freq: int = 1
    utd: float = 1.0
    online_replay_mode: Literal["empty", "append", "mixed"] = "append"
    warmup_steps: int = 0


def build_fql(args, env, eval_env, logger, checkpoint_dir):
    import torch

    from rl_garden.algorithms import Off2OnFQL
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent

    if args.reference_compatible:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing
    agent = construct_agent(
        Off2OnFQL, env=env, eval_env=eval_env, **image_kwargs,
        buffer_size=args.buffer_size, buffer_device=args.buffer_device,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, training_freq=args.training_freq, utd=args.utd,
        bootstrap_at_done=args.bootstrap_at_done,
        tau=args.tau, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        weight_decay=args.weight_decay, use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule, lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps, lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm, alpha=args.alpha, flow_steps=args.flow_steps,
        q_agg=args.q_agg, normalize_q_loss=args.normalize_q_loss,
        net_arch=[args.hidden_dim] * args.hidden_layers, n_critics=args.n_critics,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        critic_dropout_rate=args.critic_dropout_rate, kernel_init=args.kernel_init,
        backbone_type=args.backbone_type, activation_fn=args.activation_fn,
        reference_compatible=args.reference_compatible,
        offline_sampling=args.offline_sampling, seed=args.seed, logger=logger,
        std_log=args.std_log, log_freq=args.log_freq, eval_freq=args.online_eval_freq or 0,
        num_eval_steps=args.num_eval_steps, num_eval_episodes=args.num_eval_episodes,
        checkpoint_dir=checkpoint_dir, checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_fql(args: FQLOff2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_fql, algorithm="fql")


def _off2on_fql_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnFQL

    return Off2OnFQL


registry.register("fql", FQLOff2OnArgs, run_fql, algorithm_cls=_off2on_fql_algorithm_cls)
