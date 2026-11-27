"""FINO (Shin et al., "Flow Matching with Injected Noise for Offline-to-Online
RL", ICLR 2026) offline-to-online run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/runner.py``)
unmodified, matching ``floq.py``'s shape: only a ``build_fino`` callback is
needed here.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. Mirrors ``FloQOff2OnArgs``: fields inlined
against ``FQLTrainingArgs``, ``ObservationArgs``, ``EnvBackendArgs`` rather
than adding a new class to ``off2on/_args.py`` -- ``Off2OnFINO`` has no critic-flow
machinery, so FloQ's flow-critic fields (``r_min``, ``r_max``,
``flow_num_ensembles``, etc.) are dropped and FINO's three new fields
(``noise_scale``, ``beta``, ``num_samples``) are added instead.

Group-norm options reach the constructor; the flow actor does not consume
``actor_use_group_norm``. ``critic_use_group_norm`` and ``num_groups``
configure the critic.
``warmup_steps`` maps to ``initial_training_phase`` (default 0 means no
warmup). Inherited ``critic_subsample_size``, ``actor_dropout_rate`` and
``std_parameterization`` have no corresponding FQL constructor options and
remain unused.
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.fql import FINOParams


def build_fino(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnFINO
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.off2on.args import initial_training_phase_from_args

    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    learning_kwargs = params_kwargs(args, Off2OnFINO.params_class)
    learning_kwargs["net_arch"] = list(args.net_arch) if args.net_arch is not None else [args.hidden_dim] * args.hidden_layers

    agent = construct_agent(
        Off2OnFINO,
        **learning_kwargs,
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
        actor_use_group_norm=args.actor_use_group_norm,
        initial_training_phase=initial_training_phase_from_args(args),
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
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_fino, algorithm="fino")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.off2on.args import FQLTrainingArgs  # noqa: E402
from rl_garden.training.off2on.registry import registry  # noqa: E402


@dataclass
class FINOOff2OnArgs(FINOParams, FQLTrainingArgs, ObservationArgs, EnvBackendArgs):
    """FINO -- offline-to-online flow-matching actor with noise-injected
    BC-flow training and rejection-sampled inference (Shin et al., ICLR
    2026, ``3rd_party/FINO/agents/fino.py``). State-only observations by
    default; pass ``--obs.rgb <camera>`` for CNN-based Dict/RGBD observations.
    """

    online_replay_mode: Literal["empty", "append", "mixed"] = "append"
    warmup_steps: int = 0
    hidden_dim: int = 512
    hidden_layers: int = 4


def _off2_on_fino_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnFINO

    return Off2OnFINO

registry.register(
    "fino",
    FINOOff2OnArgs,
    run_fino,
    algorithm_cls=_off2_on_fino_algorithm_cls,
    build_agent=build_fino,
)
