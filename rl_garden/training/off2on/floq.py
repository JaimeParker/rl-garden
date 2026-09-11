"""FloQ (Farebrother et al., arXiv 2509.06863) offline-to-online run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/_runner.py``)
unmodified, matching ``acfql.py``'s shape: only a ``build_floq`` callback is
needed here.

Box observations by default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD
observations. Mirrors ``ACFQLArgs``: fields inlined against
``Off2OnCommonArgs``, ``VisionArgs``, ``EnvBackendArgs`` rather than adding a
new class to ``off2on/_args.py`` -- ``Off2OnFloQ`` has no action chunking, so
the chunking-only fields on ``ACFQLArgs`` (``horizon_length``, ``actor_type``,
``actor_num_samples``) are dropped and the 13 floq fields are added instead.

As with ``ACFQLArgs``, a few inherited ``Off2OnCommonArgs`` fields have no
effect on FloQ and are not read by ``build_floq``: ``critic_subsample_size``,
``actor_use_group_norm``, ``critic_use_group_norm``, ``num_groups``,
``std_parameterization``, ``warmup_steps`` -- these exist on
``Off2OnCommonArgs`` for the CQL/IQL SAC-style actor-critic families sharing
it, not because FQL/FloQ's flow-matching actor needs them.
"""

from __future__ import annotations


def build_floq(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnFloQ
    from rl_garden.common.cli_args import image_encoder_factory_from_args, image_keys_from_env
    from rl_garden.training.inspection import construct_agent

    is_visual = args.obs_mode != "state"
    image_kwargs: dict = {}
    if is_visual:
        image_kwargs = dict(
            image_encoder_factory=image_encoder_factory_from_args(args),
            image_keys=image_keys_from_env(env, args),
            image_fusion_mode=args.image_fusion_mode,
        )

    agent = construct_agent(
        Off2OnFloQ,
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
        encoder_sharing=args.encoder_sharing,
        r_min=args.r_min,
        r_max=args.r_max,
        flow_num_ensembles=args.flow_num_ensembles,
        noise_samples=args.noise_samples,
        noise_coverage=args.noise_coverage,
        critic_flow_steps=args.critic_flow_steps,
        train_at_zero_only=args.train_at_zero_only,
        embed_time=args.embed_time,
        time_embed_dim=args.time_embed_dim,
        use_prob_embed=args.use_prob_embed,
        num_bins=args.num_bins,
        sigma=args.sigma,
        reward_offset=args.reward_offset,
        critic_flow_net_arch=args.critic_flow_net_arch,
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


def run_floq(args: "FloQOff2OnArgs") -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_floq, algorithm="floq")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional  # noqa: E402

from rl_garden.common.cli_args import VisionArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import Activation, KernelInit  # noqa: E402
from rl_garden.policies.floq_policy import EncoderSharing  # noqa: E402
from rl_garden.training.off2on._args import Off2OnCommonArgs  # noqa: E402
from rl_garden.training.off2on._registry import registry  # noqa: E402


@dataclass
class FloQOff2OnArgs(Off2OnCommonArgs, VisionArgs, EnvBackendArgs):
    """FloQ -- offline-to-online flow-matching-critic FQL (Farebrother et
    al. 2025, ``3rd_party/floq/agents/floq.py``). Box observations by
    default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD observations.
    """

    # See ACFQLArgs' own precedent: defaults to "state" (not VisionArgs' own
    # "rgb" default) so callers that don't pass --obs_mode get state obs.
    obs_mode: str = "state"

    # 10.0, not ACFQL's 100.0: floq's own reference config
    # (3rd_party/floq/agents/floq.py get_config, alpha=10.0) inherits FQL's
    # actor-side BC coefficient unchanged (ACFQL's 100.0 is Q-chunking's own
    # action-chunked recipe, not floq's).
    alpha: float = 10.0
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

    # See rl_garden/algorithms/floq.py's module docstring for the reference
    # facts behind these defaults (OGBench singletask sparse-reward r_min/
    # r_max, HL-Gauss critic bins, noise coverage, etc.).
    r_min: float = -1.0
    r_max: float = 0.0
    flow_num_ensembles: int = 2
    noise_samples: int = 8
    noise_coverage: float = 0.1
    critic_flow_steps: int = 8
    train_at_zero_only: bool = False
    embed_time: bool = True
    time_embed_dim: int = 64
    use_prob_embed: bool = True
    num_bins: int = 51
    sigma: float = 16.0
    reward_offset: float = 0.01
    # Flow-critic velocity network width/depth. None falls back to net_arch
    # (hidden_dim x hidden_layers) -- the reference floq's block_width/
    # block_depth size only this network; actor and distilled critic stay
    # at net_arch, matching the README's cube presets (block_depth=2).
    critic_flow_net_arch: Optional[list[int]] = None


registry.register("floq", FloQOff2OnArgs, run_floq)
