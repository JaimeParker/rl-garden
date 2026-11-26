"""GAIL run function."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_gail(
    args,
    env,
    eval_env,
    logger,
    checkpoint_dir,
    *,
    algorithm_cls: type | None = None,
    **extra_kwargs: Any,
):
    """Build ``algorithm_cls`` (``GAIL`` when ``None``) from ``args``;
    ``extra_kwargs`` go to its constructor next to the PPO and GAIL kwargs."""
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online.entrypoints.ppo import (
        ppo_common_kwargs,
        ppo_observation_kwargs,
    )

    if algorithm_cls is None:
        from rl_garden.algorithms import GAIL

        algorithm_cls = GAIL
    agent = construct_agent(
        algorithm_cls,
        **ppo_common_kwargs(
            args, env, eval_env, logger, checkpoint_dir, ppo_observation_kwargs(args)
        ),
        demo_env_id=args.demo_env_id,
        demo_dataset_backend=args.demo_dataset_backend,
        demo_buffer_size=args.demo_buffer_size,
        demo_batch_size=args.demo_batch_size,
        n_disc_updates_per_round=args.n_disc_updates_per_round,
        disc_net_arch=args.disc_net_arch,
        disc_lr=args.disc_lr,
        **extra_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_gail(args: "GAILArgs", *, build_agent: Callable | None = None) -> None:
    """Train with ``build_agent`` (``build_gail`` when ``None``)."""
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_gail if build_agent is None else build_agent,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online.args import GAILTrainingArgs  # noqa: E402
from rl_garden.training.online.registry import registry  # noqa: E402


@dataclass
class GAILArgs(GAILTrainingArgs, ObservationArgs, EnvBackendArgs):
    """GAIL (Ho & Ermon 2016) -- PPO generator + adversarial discriminator.

    Observation and encoder flags (``--obs.*``, ``--encoder.*``,
    ``--obs-groups.*``, ``--critic-encoder.*``, ``--encoder-sharing``,
    ``--policy-graph``) reach the agent exactly as for ``ppo``. The
    discriminator reads critic-role features, so its input is whatever the
    critic group selects. The demo dataset only provides a flat ``state``
    key, so the critic group must resolve to ``state`` alone; any other
    critic key raises ``ObservationContractError`` at construction.

    Env backend: ``--env_backend d4rl_legacy`` (D4RL MuJoCo locomotion).
    Expert demonstrations are loaded separately via ``--demo_env_id``
    (typically the same task's ``-expert-v2`` dataset, e.g.
    ``halfcheetah-expert-v2`` for ``--env_id halfcheetah-medium-v2``).
    """


def _gail_algorithm_cls() -> type:
    from rl_garden.algorithms import GAIL

    return GAIL


registry.register(
    "gail",
    GAILArgs,
    run_gail,
    algorithm_cls=_gail_algorithm_cls,
    build_agent=build_gail,
    training_only_inputs={"demo_env_id": None},
)
