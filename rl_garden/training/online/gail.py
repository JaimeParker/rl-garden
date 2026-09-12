"""GAIL run function."""

from __future__ import annotations


def build_gail(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import GAIL
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online.ppo import _ppo_common_kwargs

    agent = construct_agent(
        GAIL,
        **_ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, {}),
        demo_env_id=args.demo_env_id,
        demo_dataset_backend=args.demo_dataset_backend,
        demo_buffer_size=args.demo_buffer_size,
        demo_batch_size=args.demo_batch_size,
        n_disc_updates_per_round=args.n_disc_updates_per_round,
        disc_net_arch=args.disc_net_arch,
        disc_lr=args.disc_lr,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_gail(args: "GAILArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_gail,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import GAILTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class GAILArgs(GAILTrainingArgs, ObservationArgs, EnvBackendArgs):
    """GAIL (Ho & Ermon 2016) -- PPO generator + adversarial discriminator.

    State-only in practice: ``GAILDiscriminator`` is a bespoke state-action
    MLP that rejects any image key at construction (see ``GAIL._setup_model``'s
    ``ObservationContractError`` guard), so ``build_gail`` never wires an
    encoder into the agent even though ``ObservationArgs`` is present (for
    CLI/config uniformity -- ``--obs.rgb`` parses but fails fast at
    construction rather than being silently ignored).

    Env backend: ``--env_backend d4rl_legacy`` (D4RL MuJoCo locomotion).
    Expert demonstrations are loaded separately via ``--demo_env_id``
    (typically the same task's ``-expert-v2`` dataset, e.g.
    ``halfcheetah-expert-v2`` for ``--env_id halfcheetah-medium-v2``).
    """


registry.register("gail", GAILArgs, run_gail)
