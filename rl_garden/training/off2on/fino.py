"""FINO (Shin et al., "Flow Matching with Injected Noise for Offline-to-Online
RL", ICLR 2026) offline-to-online run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/_runner.py``)
unmodified, matching ``floq.py``'s shape: only a ``build_fino`` callback is
needed here.

Box observations by default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD
observations. Mirrors ``FloQOff2OnArgs``: fields inlined against
``Off2OnCommonArgs``, ``VisionArgs``, ``EnvBackendArgs`` rather than adding a
new class to ``off2on/_args.py`` -- ``Off2OnFINO`` has no critic-flow
machinery, so FloQ's flow-critic fields (``r_min``, ``r_max``,
``flow_num_ensembles``, etc.) are dropped and FINO's three new fields
(``noise_scale``, ``beta``, ``num_samples``) are added instead.

As with ``FloQOff2OnArgs``, a few inherited ``Off2OnCommonArgs`` fields have
no effect on FINO and are not read by ``build_fino``: ``critic_subsample_size``,
``actor_use_group_norm``, ``critic_use_group_norm``, ``num_groups``,
``std_parameterization``, ``warmup_steps`` -- these exist on
``Off2OnCommonArgs`` for the CQL/IQL SAC-style actor-critic families sharing
it, not because FQL/FINO's flow-matching actor needs them.
"""

from __future__ import annotations


def build_fino(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnFINO
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
        Off2OnFINO,
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
        noise_scale=args.noise_scale,
        beta=args.beta,
        num_samples=args.num_samples,
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


def run_fino(args: "FINOOff2OnArgs") -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_fino, algorithm="fino")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional  # noqa: E402

from rl_garden.common.cli_args import VisionArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import Activation, KernelInit  # noqa: E402
from rl_garden.policies.fino_policy import EncoderSharing  # noqa: E402
from rl_garden.training.off2on._args import Off2OnCommonArgs  # noqa: E402
from rl_garden.training.off2on._registry import registry  # noqa: E402


@dataclass
class FINOOff2OnArgs(Off2OnCommonArgs, VisionArgs, EnvBackendArgs):
    """FINO -- offline-to-online flow-matching actor with noise-injected
    BC-flow training and rejection-sampled inference (Shin et al., ICLR
    2026, ``3rd_party/FINO/agents/fino.py``). Box observations by default;
    pass ``--obs_mode rgb`` for CNN-based Dict/RGBD observations.
    """

    # See FloQOff2OnArgs' own precedent: defaults to "state" (not VisionArgs'
    # own "rgb" default) so callers that don't pass --obs_mode get state obs.
    obs_mode: str = "state"

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

    # See rl_garden/algorithms/fino.py's module docstring for the reference
    # facts behind these defaults.
    noise_scale: float = 0.1
    beta: float = 10.0
    num_samples: Optional[int] = None


registry.register("fino", FINOOff2OnArgs, run_fino)
