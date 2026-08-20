"""ACFQL (Q-chunking's action-chunked, offline-to-online FQL) run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/_runner.py``)
unmodified -- it already does exactly what QC's own ``main.py`` does (load
offline data into ``agent.replay_buffer``, run offline gradient steps,
``switch_to_online_mode``, continue via ``learn()``), so only a
``build_acfql`` callback is needed here, matching ``wsrl.py``'s shape.

State observations only -- ``ChunkedTensorReplayBuffer`` is Box-only for v1.
``run_off2on``'s shared runner reads ``args.obs_mode`` unconditionally to
build the ``EnvRequest`` (unlike ``run_online``, it has no per-algorithm
``make_env_request`` callback), so ``obs_mode`` is still a real field on
``ACFQLArgs`` -- just fixed to ``"state"`` rather than exposed as a
vision-capable knob.

``ACFQLArgs`` extends ``Off2OnCommonArgs`` for the orchestration fields
``run_off2on`` reads directly (``num_offline_steps``, ``online_replay_mode``,
``offline_data_ratio``, ``dataset_backend``, etc.) and the FQL-recipe
defaults it overrides (``n_critics=2``, ``actor_use_layer_norm=False``,
matching plain FQL rather than CQL/IQL's SAC-style recipe). A few inherited
fields have no effect on ACFQL and are not read by ``build_acfql``:
``critic_subsample_size``, ``actor_use_group_norm``, ``critic_use_group_norm``,
``num_groups``, ``std_parameterization``, ``warmup_steps`` -- these exist on
``Off2OnCommonArgs`` for the CQL/IQL SAC-style actor-critic families sharing
it, not because FQL's flow-matching actor needs them.
"""

from __future__ import annotations


def build_acfql(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ACFQL
    from rl_garden.training.inspection import construct_agent

    agent = construct_agent(
        ACFQL,
        env=env,
        eval_env=eval_env,
        horizon_length=args.horizon_length,
        actor_type=args.actor_type,
        actor_num_samples=args.actor_num_samples,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
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
        alpha=args.alpha,
        flow_steps=args.flow_steps,
        q_agg=args.q_agg,
        normalize_q_loss=args.normalize_q_loss,
        net_arch=[args.hidden_dim] * args.hidden_layers,
        n_critics=args.n_critics,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        activation_fn=args.activation_fn,
        encoder_sharing=args.encoder_sharing,
        offline_sampling=args.offline_sampling,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.online_eval_freq or 0,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_acfql(args: "ACFQLArgs") -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_acfql, algorithm="acfql")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import Activation, KernelInit  # noqa: E402
from rl_garden.policies.acfql_policy import ActorType, EncoderSharing  # noqa: E402
from rl_garden.training.off2on._args import Off2OnCommonArgs  # noqa: E402
from rl_garden.training.off2on._registry import registry  # noqa: E402


@dataclass
class ACFQLArgs(Off2OnCommonArgs, EnvBackendArgs):
    """ACFQL -- Q-chunking's action-chunked, offline-to-online FQL (Li, Zhou,
    Levine 2025, ``3rd_party/qc/agents/acfql.py``). State observations only.
    """

    # run_off2on's shared runner (_runner.py) reads args.obs_mode
    # unconditionally to build the EnvRequest -- unlike run_online, it has no
    # per-algorithm make_env_request callback to hide this behind. Fixed to
    # "state": ACFQLCore is Box-only (raises TypeError on Dict obs downstream
    # if overridden), so this isn't exposed as a real vision-capable knob.
    obs_mode: str = "state"

    horizon_length: int = 5
    actor_type: ActorType = "distill-ddpg"
    actor_num_samples: int = 32
    alpha: float = 100.0
    flow_steps: int = 10
    q_agg: Literal["mean", "min"] = "mean"
    normalize_q_loss: bool = False
    hidden_dim: int = 512
    hidden_layers: int = 4
    activation_fn: Optional[Activation] = "gelu"
    encoder_sharing: EncoderSharing = "shared"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    n_critics: int = 2
    actor_use_layer_norm: bool = False
    kernel_init: Optional[KernelInit] = "xavier_uniform"


registry.register("acfql", ACFQLArgs, run_acfql)
