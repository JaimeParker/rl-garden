"""Value Flows (Dong et al., arXiv 2510.07650) offline-to-online run
function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/_runner.py``)
unmodified, matching ``floq.py``'s shape: only a ``build_value_flows``
callback is needed here.

Box observations by default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD
observations. Mirrors ``FloQOff2OnArgs``: fields inlined against
``Off2OnCommonArgs``, ``VisionArgs``, ``EnvBackendArgs`` rather than adding a
new class to ``off2on/_args.py`` -- ``Off2OnValueFlows`` has no action
chunking, so the FloQ-only flow-critic fields (``r_min``/``r_max``/
``flow_num_ensembles``/``noise_samples``/``noise_coverage``/
``critic_flow_steps``/``train_at_zero_only``/``embed_time``/
``time_embed_dim``/``use_prob_embed``/``num_bins``/``sigma``/
``reward_offset``) are dropped and Value Flows' own 9 fields
(``min_reward``/``max_reward``/``ret_agg``/``confidence_weight_temp``/
``dcfm_lambda``/``bcfm_lambda``/``clip_flow_returns``/``num_samples``/
``policy_extraction``) are added instead.

As with ``FloQOff2OnArgs``, a few inherited ``Off2OnCommonArgs`` fields have
no effect on Value Flows and are not read by ``build_value_flows``:
``critic_subsample_size``, ``actor_use_group_norm``, ``critic_use_group_norm``,
``num_groups``, ``std_parameterization``, ``warmup_steps``.
"""

from __future__ import annotations


def build_value_flows(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnValueFlows
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
        Off2OnValueFlows,
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
        min_reward=args.min_reward,
        max_reward=args.max_reward,
        ret_agg=args.ret_agg,
        confidence_weight_temp=args.confidence_weight_temp,
        dcfm_lambda=args.dcfm_lambda,
        bcfm_lambda=args.bcfm_lambda,
        clip_flow_returns=args.clip_flow_returns,
        num_samples=args.num_samples,
        policy_extraction=args.policy_extraction,
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


def run_value_flows(args: "ValueFlowsOff2OnArgs") -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_value_flows, algorithm="value_flows")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional  # noqa: E402

from rl_garden.common.cli_args import VisionArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import Activation, KernelInit  # noqa: E402
from rl_garden.policies.fql_policy import EncoderSharing  # noqa: E402
from rl_garden.training.off2on._args import Off2OnCommonArgs  # noqa: E402
from rl_garden.training.off2on._registry import registry  # noqa: E402


@dataclass
class ValueFlowsOff2OnArgs(Off2OnCommonArgs, VisionArgs, EnvBackendArgs):
    """Value Flows -- offline-to-online distributional flow-matching-critic
    FQL (Dong et al. 2025, ``3rd_party/value-flows/agents/value_flows.py``).
    Box observations by default; pass ``--obs_mode rgb`` for CNN-based
    Dict/RGBD observations.
    """

    # See FloQOff2OnArgs' own precedent: defaults to "state" (not VisionArgs'
    # own "rgb" default) so callers that don't pass --obs_mode get state obs.
    obs_mode: str = "state"

    # 10.0: the reference's own get_config alpha, matching FQL's actor-side
    # BC coefficient unchanged.
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

    # See rl_garden/algorithms/value_flows.py's module docstring for the
    # reference facts behind these defaults.
    min_reward: float = -1.0
    max_reward: float = 0.0
    ret_agg: Literal["mean", "min"] = "mean"
    confidence_weight_temp: float = 0.3
    dcfm_lambda: float = 1.0
    bcfm_lambda: float = 1.0
    clip_flow_returns: bool = True
    num_samples: int = 16
    policy_extraction: Literal["rs", "rpg"] = "rs"


registry.register("value_flows", ValueFlowsOff2OnArgs, run_value_flows)
