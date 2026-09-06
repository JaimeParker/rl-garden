"""GAIL run function."""

from __future__ import annotations


def _gail_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest, should_create_eval_env

    backend_config = args.resolve_backend_config()
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=None,
        camera_height=None,
        include_state=True,
        per_camera_rgbd=False,
        frame_stack=1,
        num_eval_envs=args.num_eval_envs,
        create_eval_env=should_create_eval_env(args),
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )


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
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=_gail_env_request,
        build_agent=build_gail,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import GAILTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class GAILArgs(GAILTrainingArgs, EnvBackendArgs):
    """GAIL (Ho & Ermon 2016) -- PPO generator + adversarial discriminator.

    Env backend: ``--env_backend d4rl_legacy`` (D4RL MuJoCo locomotion).
    Expert demonstrations are loaded separately via ``--demo_env_id``
    (typically the same task's ``-expert-v2`` dataset, e.g.
    ``halfcheetah-expert-v2`` for ``--env_id halfcheetah-medium-v2``).
    """


registry.register("gail", GAILArgs, run_gail)
