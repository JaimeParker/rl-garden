"""SACFlow run function. State observations by default; pass ``--obs.rgb
<camera>`` (with a Dict/RGBD-producing env backend) for CNN-based vision --
see ``rl_garden.algorithms.sac_flow.SACFlow``'s own docstring for exactly
which vision paths are supported (CombinedExtractor encoders; not ViT, not a
separate critic encoder)."""

from __future__ import annotations


def build_sac_flow(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.common.cli_args import resolve_obs_groups_config
    from rl_garden.algorithms import SACFlow
    from rl_garden.encoders.config import EncoderConfig
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }

    image_kwargs: dict = {}
    if args.obs.is_visual:
        if args.encoder.backbone == "vit":
            raise SystemExit(
                "sac_flow does not support --encoder.backbone vit: "
                "FlowMatchingActor has not been verified against a "
                "structured (ViT token) features extractor. Use a CNN "
                "encoder (e.g. plain_conv, resnet10/18, drqv2_conv, cnn3d)."
            )
        if args.critic_encoder != EncoderConfig():
            raise SystemExit(
                "sac_flow does not support --critic-encoder.* (a separate "
                "critic-only image encoder): SACFlowPolicy inherits "
                "SACPolicy's critic_extractor support unchanged, but this "
                "entrypoint deliberately does not wire a separate critic "
                "encoder config through to it."
            )
        image_kwargs = dict(
            encoder_config=args.encoder,
            obs_groups=resolve_obs_groups_config(args),
            image_augmentation_seed=args.seed + 1_000_003,
        )

    agent = construct_agent(
        SACFlow,
        env=env,
        eval_env=eval_env,
        denoising_steps=args.denoising_steps,
        noise_std=args.noise_std,
        flow_hidden_dims=[args.flow_hidden_dim] * args.flow_hidden_layers,
        flow_use_layer_norm=args.flow_use_layer_norm,
        **image_kwargs,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        nstep=args.nstep,
        tau=args.tau,
        bootstrap_at_done=args.bootstrap_at_done,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_tuning=args.alpha_tuning,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        alpha_lr=args.alpha_lr,
        q_landscape_diagnostics=args.q_landscape_diagnostics,
        q_landscape_num_actions=args.q_landscape_num_actions,
        q_landscape_batch_size=args.q_landscape_batch_size,
        q_mc_diagnostics=args.q_mc_diagnostics,
        initial_training_phase=sac_initial_training_phase_from_args(args),
        critic_impl=args.critic_impl,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        critic_use_layer_norm=args.critic_use_layer_norm,
        net_arch=net_arch,
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
    if args.load_actor_checkpoint is not None:
        # Inherited from SAC; expects a BC checkpoint with a matching actor
        # architecture. A flow actor's state_dict keys don't overlap with a
        # Gaussian BC actor's, so this will raise a clear "missing keys"
        # ValueError rather than silently loading nothing -- there is no
        # flow-compatible BC checkpoint format in this version.
        agent.load_actor_checkpoint(args.load_actor_checkpoint)
    return agent


def run_sac_flow(args: "SACFlowArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_sac_flow,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionSACFlowTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class SACFlowArgs(VisionSACFlowTrainingArgs, EnvBackendArgs):
    """SACFlow -- SAC with a flow-matching actor. State observations by
    default; pass ``--obs.rgb <camera>`` for CNN-based Dict/RGBD observations
    (not ``--encoder.backbone vit``, not ``--critic-encoder.*`` -- see
    ``SACFlow``'s own docstring).

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend custom``.
    """




def _sac_flow_algorithm_cls() -> type:
    from rl_garden.algorithms import SACFlow

    return SACFlow

registry.register("sac_flow", SACFlowArgs, run_sac_flow, algorithm_cls=_sac_flow_algorithm_cls)
