"""ACRLPD (Q-chunking's action-chunked RLPD) run function.

``ACRLPDCore._setup_model`` raises ``ObservationContractError`` before
anything is built if the resolved observation space carries any image key --
ACRLPD is state-only for v1 -- so ``--obs.rgb``/``--obs.depth`` are exposed
(for CLI/config uniformity with every other algorithm) but any non-default
value is rejected by the algorithm itself, not silently ignored.
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs

from rl_garden.hyperparameters.sac import ACRLPDParams


def build_acrlpd(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ACRLPD
    from rl_garden.common.cli_args import (
        policy_module_kwargs,
        resolve_critic_encoder_config,
        resolve_obs_groups_config,
    )
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online.args import sac_initial_training_phase_from_args

    if args.nstep != 1:
        raise ValueError(
            f"ACRLPD does not take nstep={args.nstep!r} (--nstep): "
            "its replay window is horizon_length (--horizon-length); use --nstep 1."
        )

    if args.load_actor_checkpoint is not None and args.horizon_length > 1:
        raise ValueError(
            "ACRLPD does not take load_actor_checkpoint (--load-actor-checkpoint) "
            "with horizon_length > 1: BC actor loading requires --horizon-length 1; "
            "leave the checkpoint unset for action chunks."
        )

    net_arch = args.net_arch if args.net_arch is not None else {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    learning_kwargs = params_kwargs(args, ACRLPDParams)
    learning_kwargs.pop("nstep")
    agent = construct_agent(
        ACRLPD,
        **{
            **learning_kwargs,
            "net_arch": net_arch,
        },
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        bootstrap_at_done=args.bootstrap_at_done,
        training_freq=args.training_freq,
        utd=args.utd,
        initial_training_phase=sac_initial_training_phase_from_args(args),
        critic_backbone_type=args.critic_backbone_type,
        q_landscape_diagnostics=args.q_landscape_diagnostics,
        q_landscape_num_actions=args.q_landscape_num_actions,
        q_landscape_batch_size=args.q_landscape_batch_size,
        q_mc_diagnostics=args.q_mc_diagnostics,
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
        **policy_module_kwargs(args),
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.load_actor_checkpoint is not None:
        agent.load_actor_checkpoint(args.load_actor_checkpoint)
    if args.offline_dataset is not None:
        if args.offline_buffer_size is None:
            raise ValueError(
                "--offline_buffer_size is required when --offline_dataset is set "
                "(ACRLPD's prior-data buffer has no cheap way to count "
                "h5 transitions ahead of time)."
            )
        loaded = agent.load_offline_replay_buffer(
            args.offline_dataset,
            backend="h5",
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
            success_key=args.success_key,
        )
        if args.std_log:
            print(
                "[acrlpd] "
                f"offline_dataset={args.offline_dataset} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio}",
                flush=True,
            )
    return agent


def run_acrlpd(args: "ACRLPDArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_acrlpd,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.cli_args import ObservationArgs, PolicyModuleArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online.args import SACTrainingArgs  # noqa: E402
from rl_garden.training.online.registry import registry  # noqa: E402


@dataclass
class ACRLPDArgs(ACRLPDParams, SACTrainingArgs, ObservationArgs, PolicyModuleArgs, EnvBackendArgs):
    """ACRLPD -- Q-chunking's action-chunked RLPD (Li, Zhou, Levine 2025,
    ``3rd_party/qc/agents/acrlpd.py``). State-only in practice: the
    algorithm itself rejects any image observation (see module docstring).

    Defaults matching the reference (n_critics=10, no REDQ subsampling,
    mean- not min-ensemble aggregation, no entropy backup) rather than plain
    RLPD's own defaults.

    ``--critic-only-steps`` above 0 is rejected: Q-chunking has no critic-only
    phase.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend custom``.
    """


    utd: float = 4.0



    offline_dataset: str | None = None
    offline_num_traj: int | None = None
    offline_buffer_size: int | None = None
    offline_data_ratio: float = 0.5
    success_key: str | None = None




def _acrlpd_algorithm_cls() -> type:
    from rl_garden.algorithms import ACRLPD

    return ACRLPD

registry.register(
    "acrlpd",
    ACRLPDArgs,
    run_acrlpd,
    algorithm_cls=_acrlpd_algorithm_cls,
    build_agent=build_acrlpd,
    training_only_inputs={
        "offline_dataset": None,
        "critic_only_steps": 0,
        "nstep": 1,
        "load_actor_checkpoint": None,
    },
)
