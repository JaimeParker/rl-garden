"""SUPE run function.

Env wrapping happens here, at env-construction time, via
``SkillActionWrapper`` -- exactly like ``DPPO``'s ``ActionChunkWrapper``
wiring (``rl_garden/training/online/entrypoints/dppo.py:39-46``): ``SUPE`` itself only
ever sees an already-skill-wrapped ``env.single_action_space``.
"""

from __future__ import annotations

from rl_garden.hyperparameters.sac import ExPLOReParams

import numpy as np


def build_supe(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import SUPE
    from rl_garden.algorithms.supe import (
        build_frozen_opal_vae,
        embedded_opal_vae,
        load_opal_vae,
    )
    from rl_garden.common.cli_args import policy_module_kwargs
    from rl_garden.envs.wrappers import SkillActionWrapper
    from rl_garden.observations import ObservationContractError
    from rl_garden.training.inspection import (
        checkpoint_to_restore,
        construct_agent,
        warn_if_superseded,
    )
    from rl_garden.training.online.entrypoints.sac import sac_common_kwargs, sac_observation_kwargs

    obs_keys = set(env.single_observation_space.spaces)
    if obs_keys != {"state"}:
        raise ObservationContractError(
            "SUPE is state-only: OPAL's skill decoder reads the 'state' observation "
            f"alone, got keys {sorted(obs_keys)!r}."
        )
    obs_dim = int(np.prod(env.single_observation_space["state"].shape))
    # The agent moves the VAE, and with it the wrappers' decoder, to its device.
    record = embedded_opal_vae(checkpoint_to_restore(args))
    if record is not None:
        warn_if_superseded("--opal-checkpoint", args.opal_checkpoint, record["state_dict"])
        opal_vae = build_frozen_opal_vae(
            record["hyperparameters"], record["state_dict"], obs_dim, env.single_action_space,
            device="cpu",
        )
    else:
        opal_vae = load_opal_vae(
            args.opal_checkpoint, obs_dim, env.single_action_space, device="cpu",
        )
    skill_dim = opal_vae.skill_dim

    # Both envs decode stochastically: upstream's MetaPolicyActionWrapper.step()
    # (meta_env_wrapper.py:104-118) always calls the stochastic
    # `sample_skill_actions`, for both the train and eval instances
    # (train_finetuning_supe.py:185-203) -- its `eval` flag only toggles
    # reward-discount handling, never action determinism. The deterministic
    # decode path (`eval_skill_actions`, opal.py:454-460) exists upstream but
    # is used only by OPAL's own standalone pretraining eval loop
    # (run_opal.py), not by SUPE's online rollout.
    env = SkillActionWrapper(
        env, opal_vae.decoder, horizon=args.horizon, skill_dim=skill_dim, deterministic=False,
    )
    if eval_env is not None:
        eval_env = SkillActionWrapper(
            eval_env, opal_vae.decoder, horizon=args.horizon, skill_dim=skill_dim,
            deterministic=False,
        )

    agent = construct_agent(
        SUPE,
        opal_checkpoint=args.opal_checkpoint,
        opal_vae=opal_vae,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        **sac_common_kwargs(
            args, env, eval_env, logger, checkpoint_dir, sac_observation_kwargs(args),
            params_class=ExPLOReParams,
        ),
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
                "(RLPD's prior-data buffer has no cheap way to count "
                "h5/minari transitions ahead of time)."
            )
        loaded = agent.load_skill_relabeled_offline_buffer(
            args.offline_dataset,
            buffer_size=args.offline_buffer_size,
            chunk_size=args.horizon,
            discount=args.gamma,
            offline_data_ratio=args.offline_data_ratio,
            num_traj=args.offline_num_traj,
        )
        if args.std_log:
            print(
                "[supe] "
                f"offline_dataset={args.offline_dataset} "
                f"loaded_windows={loaded} "
                f"offline_data_ratio={args.offline_data_ratio} "
                f"offline_relabel_type={args.offline_relabel_type} "
                f"horizon={args.horizon} skill_dim={skill_dim}",
                flush=True,
            )
    return agent


def run_supe(args: "SUPEArgs") -> None:
    from rl_garden.training.online.runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    from rl_garden.common.env_args import make_env_request

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_supe,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.training.online.entrypoints.explore import ExPLOREArgs  # noqa: E402
from rl_garden.training.online.registry import registry  # noqa: E402


@dataclass
class SUPEArgs(ExPLOREArgs):
    """SUPE -- ExPLORe operating over OPAL's learned skill space instead of
    raw actions. State-based (Box observation space) only. Requires a
    pretrained OPAL checkpoint (``opal`` offline entrypoint).

    ``total_timesteps`` counts macro-steps (each is ``horizon`` raw env
    steps), matching ``DPPO``'s ``act_steps`` convention -- the raw env-step
    budget is ``total_timesteps * horizon``.
    """

    opal_checkpoint: str = ""
    horizon: int = 4


def _supe_algorithm_cls() -> type:
    from rl_garden.algorithms import SUPE

    return SUPE


registry.register(
    "supe",
    SUPEArgs,
    run_supe,
    algorithm_cls=_supe_algorithm_cls,
    build_agent=build_supe,
    training_only_inputs=("offline_dataset", "load_actor_checkpoint"),
)
