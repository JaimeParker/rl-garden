"""FAC (Flow Actor-Critic for Offline RL, ICLR 2026, arXiv 2602.18015)
offline-to-online run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/_runner.py``)
unmodified, matching ``fino.py``'s shape: only a ``build_fac`` callback is
needed here. Offline pretraining (BC-flow prelude -> dataset-logp cache ->
conservative actor-critic updates) is unchanged ``FACCore.train()``; the
online switch is handled entirely inside ``Off2OnFAC``/``FACCore`` (see
``rl_garden/algorithms/fac.py``'s module docstring).

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. Mirrors ``FINOOff2OnArgs``: fields inlined
against ``Off2OnCommonArgs``, ``ObservationArgs``, ``EnvBackendArgs`` rather
than adding a new class to ``off2on/_args.py`` -- ``Off2OnFAC`` has no
critic-flow/rejection-sampling machinery, so FINO's fields (``noise_scale``,
``beta``, ``num_samples``) are dropped and FAC's BC-pretrain/conservative-
penalty fields (``fac_alpha``, ``fac_lambda``, ``fac_threshold``,
``logp_method``, ``logp_hutch_probes``, ``weight_type``, ``bc_lr``,
``bc_batch_size``, ``bc_pretrain_epochs``, ``bc_pretrain_steps``) are added
instead, matching ``rl_garden/training/offline/_args.py``'s ``OfflineFACArgs``
defaults (which extend ``OfflineFQLArgs``' with ``q_agg="min"``,
``normalize_q_loss=True``).

As with ``FINOOff2OnArgs``, a few inherited ``Off2OnCommonArgs`` fields have
no effect on FAC and are not read by ``build_fac``: ``critic_subsample_size``,
``actor_use_group_norm``, ``critic_use_group_norm``, ``num_groups``,
``std_parameterization``, ``warmup_steps`` -- these exist on
``Off2OnCommonArgs`` for the CQL/IQL SAC-style actor-critic families sharing
it, not because FQL/FAC's flow-matching actor needs them (matching
``build_fino``/``build_floq``/``build_value_flows``, none of which thread
``initial_training_phase`` through either).

The reference's ``balanced_sampling`` (``main.py:212-216``, half offline
dataset / half replay buffer, every online step) maps to this repo's existing
``--online_replay_mode mixed --offline_data_ratio 0.5`` combination rather
than a new mechanism; matching FINO/FloQ's off2on files, neither overrides
``Off2OnCommonArgs``'s defaults (``online_replay_mode="empty"``,
``offline_data_ratio=0.0``), so balanced sampling is opt-in via those two
flags, not the default.
"""

from __future__ import annotations


def build_fac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnFAC
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent

    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    agent = construct_agent(
        Off2OnFAC,
        env=env,
        eval_env=eval_env,
        **image_kwargs,
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
        fac_alpha=args.fac_alpha,
        fac_lambda=args.fac_lambda,
        fac_threshold=args.fac_threshold,
        logp_method=args.logp_method,
        logp_hutch_probes=args.logp_hutch_probes,
        weight_type=args.weight_type,
        bc_lr=args.bc_lr,
        bc_batch_size=args.bc_batch_size,
        bc_pretrain_epochs=args.bc_pretrain_epochs,
        bc_pretrain_steps=args.bc_pretrain_steps,
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


def run_fac(args: "FACOff2OnArgs") -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_fac, algorithm="fac")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional  # noqa: E402

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import Activation, KernelInit  # noqa: E402
from rl_garden.training.off2on._args import Off2OnCommonArgs  # noqa: E402
from rl_garden.training.off2on._registry import registry  # noqa: E402


@dataclass
class FACOff2OnArgs(Off2OnCommonArgs, ObservationArgs, EnvBackendArgs):
    """FAC -- offline-to-online flow actor-critic with a BC-pretrain ->
    logp-cache -> conservative-actor-critic offline schedule, then online
    fine-tuning matching the reference's ``online_finetuning=True`` branch
    (Chae et al., ICLR 2026, arXiv:2602.18015, ``3rd_party/FAC/agents/fac.py``).
    State-only observations by default; pass ``--obs.rgb <camera>`` for
    CNN-based Dict/RGBD observations.
    """

    # FAC's paper default (arXiv 2602.18015, agents/fac.py:517) -- overrides
    # Off2OnCommonArgs's 0.99 default, matching offline/fac.py's FACArgs.
    gamma: float = 0.995
    alpha: float = 10.0  # unused by FAC; kept only for FQLCore-signature compatibility
    flow_steps: int = 10
    q_agg: Literal["mean", "min"] = "min"
    normalize_q_loss: bool = True
    hidden_dim: int = 512
    hidden_layers: int = 4
    activation_fn: Optional[Activation] = "gelu"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    n_critics: int = 2
    actor_use_layer_norm: bool = False
    kernel_init: Optional[KernelInit] = "xavier_uniform"

    # See rl_garden/algorithms/fac.py's module docstring for the reference
    # facts behind these defaults (matches offline/_args.py's OfflineFACArgs).
    fac_alpha: float = 1.0
    fac_lambda: float = 1.0
    fac_threshold: Literal[
        "batch_adaptive", "batch_wide_constant", "dataset_wide_constant"
    ] = "batch_adaptive"
    logp_method: Literal["exact", "hutch-rade", "hutch-gaus"] = "exact"
    logp_hutch_probes: int = 8
    weight_type: Literal["linear", "logarithmic", "convex", "concave"] = "linear"
    bc_lr: float = 3e-4
    bc_batch_size: Optional[int] = None
    bc_pretrain_epochs: int = 250
    bc_pretrain_steps: Optional[int] = None


def _off2_on_fac_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnFAC

    return Off2OnFAC


registry.register("fac", FACOff2OnArgs, run_fac, algorithm_cls=_off2_on_fac_algorithm_cls)
