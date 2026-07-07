"""IQL offline-to-online training registration.

Builds ``Off2OnIQL`` and reuses the same ``_runner.run_off2on`` orchestration
as ``wsrl``/``calql``. Confirmed against the reference WSRL/IQL JAX
implementation: IQL needs no online-switch override, so this preset mirrors
``Off2OnCalQL``'s (no warmup, mixed replay retained by default) rather than
WSRL's warmup-then-discard preset.
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
    VisionIQLOff2OnTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on._registry import registry


@dataclass
class IQLOff2OnArgs(VisionIQLOff2OnTrainingArgs, EnvBackendArgs):
    """IQL off2on args: no warmup, mixed replay, adaptive ratio.

    For state obs pass --obs_mode state.
    """

    warmup_steps: int = 0
    online_replay_mode: Literal["empty", "append", "mixed"] = "mixed"
    offline_data_ratio: float | str = "auto"


def build_iql(args: IQLOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnIQL

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

    agent = Off2OnIQL(
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
        offline_sampling=args.offline_sampling,
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
        **image_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_iql(args: IQLOff2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_iql, algorithm="iql")


registry.register("iql", IQLOff2OnArgs, run_iql)
