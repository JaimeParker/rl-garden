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

    State-only in practice: ``build_gail`` never forwards
    ``encoder_config``/``obs_groups``/``critic_encoder_config`` from the CLI
    (it calls ``_ppo_common_kwargs`` with an empty ``image_kwargs``), even
    though ``ObservationArgs`` is present (for CLI/config uniformity --
    ``--obs.rgb`` parses but is silently unused by this entrypoint). This is
    a CLI-wiring choice, not a discriminator limitation:
    ``GAILDiscriminator`` reads critic-role features
    (``self.policy.extract_critic_features``/``critic_features_dim``, see
    ``rl_garden/algorithms/gail.py``) and is schema-agnostic, so direct
    construction with ``obs_groups``/``critic_encoder_config`` (bypassing
    this CLI entrypoint) already supports asymmetric/image observations.

    Env backend: ``--env_backend d4rl_legacy`` (D4RL MuJoCo locomotion).
    Expert demonstrations are loaded separately via ``--demo_env_id``
    (typically the same task's ``-expert-v2`` dataset, e.g.
    ``halfcheetah-expert-v2`` for ``--env_id halfcheetah-medium-v2``).
    """


def _gail_algorithm_cls() -> type:
    from rl_garden.algorithms import GAIL

    return GAIL


registry.register("gail", GAILArgs, run_gail, algorithm_cls=_gail_algorithm_cls)
