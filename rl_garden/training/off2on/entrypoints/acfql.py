"""ACFQL (Q-chunking's action-chunked, offline-to-online FQL) run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/runner.py``)
unmodified -- it already does exactly what QC's own ``main.py`` does (load
offline data into ``agent.replay_buffer``, run offline gradient steps,
``switch_to_online_mode``, continue via ``learn()``), so only a
``build_acfql`` callback is needed here, matching ``wsrl.py``'s shape.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations (``ChunkedReplayBuffer``, via
``ActionChunkingMixin._build_replay_buffer``). ``run_off2on``'s shared runner reads
``args.obs`` unconditionally to build the ``EnvRequest`` (unlike
``run_online``, it has no per-algorithm ``make_env_request`` callback).

``ACFQLArgs`` composes ``ACFQLParams`` with ``FQLTrainingArgs`` for the learning
and orchestration fields
``run_off2on`` reads directly (``num_offline_steps``, ``online_replay_mode``,
``offline_data_ratio``, ``dataset_backend``, etc.) and the FQL-recipe
defaults it overrides (``n_critics=2``, ``actor_use_layer_norm=False``,
matching plain FQL rather than CQL/IQL's SAC-style recipe).
Group-norm options reach the constructor: ``critic_use_group_norm`` and
``num_groups`` configure the critic, while the flow actor does not consume
``actor_use_group_norm``. ``warmup_steps`` maps to ``initial_training_phase``
(default 0 means no warmup). Inherited ``critic_subsample_size``,
``actor_dropout_rate`` and ``std_parameterization`` have no corresponding
FQL constructor options and remain unused.
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.fql import ACFQLParams


def build_acfql(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ACFQL
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

    learning_kwargs = params_kwargs(args, ACFQL.params_class)
    learning_kwargs["net_arch"] = list(args.net_arch) if args.net_arch is not None else [args.hidden_dim] * args.hidden_layers

    agent = construct_agent(
        ACFQL,
        **learning_kwargs,
        env=env,
        eval_env=eval_env,
        **image_kwargs,
        horizon_length=args.horizon_length,
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


def run_acfql(args: "ACFQLArgs") -> None:
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_acfql, algorithm="acfql")


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
class ACFQLArgs(ACFQLParams, FQLTrainingArgs, ObservationArgs, EnvBackendArgs):
    """ACFQL -- Q-chunking's action-chunked, offline-to-online FQL (Li, Zhou,
    Levine 2025, ``3rd_party/qc/agents/acfql.py``). State-only observations
    by default; pass ``--obs.rgb <camera>`` for CNN-based Dict/RGBD
    observations.
    """

    online_replay_mode: Literal["empty", "append", "mixed"] = "append"
    warmup_steps: int = 0
    horizon_length: int = 5
    hidden_dim: int = 512
    hidden_layers: int = 4


def _acfql_algorithm_cls() -> type:
    from rl_garden.algorithms import ACFQL

    return ACFQL

registry.register(
    "acfql", ACFQLArgs, run_acfql, algorithm_cls=_acfql_algorithm_cls, build_agent=build_acfql
)
