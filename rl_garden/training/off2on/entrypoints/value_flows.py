"""Value Flows (Dong et al., arXiv 2510.07650) offline-to-online run
function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/runner.py``)
unmodified, matching ``floq.py``'s shape: only a ``build_value_flows``
callback is needed here.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. ``ValueFlowsParams`` supplies learning
settings, while ``ValueFlowsTrainingArgs`` supplies orchestration and
compatibility flags.

Group-norm options reach the constructor; the flow actor does not consume
``actor_use_group_norm``. The replacement flow critic does not consume
``critic_use_group_norm`` or ``num_groups`` either.
``warmup_steps`` maps to ``initial_training_phase`` (default 0 means no
warmup). Inherited ``critic_subsample_size``, ``actor_dropout_rate`` and
``std_parameterization`` have no corresponding FQL constructor options and
remain unused.
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.fql import ValueFlowsParams


def build_value_flows(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnValueFlows
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

    learning_kwargs = params_kwargs(args, Off2OnValueFlows.params_class)
    learning_kwargs["net_arch"] = list(args.net_arch) if args.net_arch is not None else [args.hidden_dim] * args.hidden_layers

    agent = construct_agent(
        Off2OnValueFlows,
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
        critic_use_group_norm=args.critic_use_group_norm,
        num_groups=args.num_groups,
        initial_training_phase=initial_training_phase_from_args(args),
        critic_dropout_rate=args.critic_dropout_rate,
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
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_value_flows, algorithm="value_flows")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.off2on.args import ValueFlowsTrainingArgs  # noqa: E402
from rl_garden.training.off2on.registry import registry  # noqa: E402


@dataclass
class ValueFlowsOff2OnArgs(ValueFlowsParams, ValueFlowsTrainingArgs, ObservationArgs, EnvBackendArgs):
    """Value Flows -- offline-to-online distributional flow-matching-critic
    FQL (Dong et al. 2025, ``3rd_party/value-flows/agents/value_flows.py``).
    State-only observations by default; pass ``--obs.rgb <camera>`` for
    CNN-based Dict/RGBD observations.
    """

    online_replay_mode: Literal["empty", "append", "mixed"] = "append"
    warmup_steps: int = 0
    hidden_dim: int = 512
    hidden_layers: int = 4


def _off2_on_value_flows_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnValueFlows

    return Off2OnValueFlows

registry.register(
    "value_flows",
    ValueFlowsOff2OnArgs,
    run_value_flows,
    algorithm_cls=_off2_on_value_flows_algorithm_cls,
    build_agent=build_value_flows,
)
