"""JSRL run function.

State observations only. Requires ``--guide_checkpoint`` (a frozen policy
from another algorithm's offline-pretrained checkpoint, see
``rl_garden/algorithms/jsrl.py``).
"""

from __future__ import annotations


def _jsrl_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest, should_create_eval_env

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
        backend_config=args.resolve_backend_config(),
    )


def build_jsrl(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import JSRL
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }

    agent = construct_agent(
        JSRL,
        env=env,
        eval_env=eval_env,
        guide_checkpoint=args.guide_checkpoint,
        guide_algorithm=args.guide_algorithm,
        max_horizon=args.max_horizon,
        n_curriculum_stages=args.n_curriculum_stages,
        tolerance=args.tolerance,
        window_size=args.window_size,
        guide_std_parameterization=args.guide_std_parameterization,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        nstep=args.nstep,
        tau=args.tau,
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
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_mode=args.actor_log_std_mode,
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
    return agent


def run_jsrl(args: "JSRLArgs") -> None:
    from rl_garden.training.online._runner import run_online

    if not args.guide_checkpoint:
        raise SystemExit("--guide_checkpoint is required for jsrl")
    run_online(
        args,
        obs_tag="state",
        make_env_request=_jsrl_env_request,
        build_agent=build_jsrl,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import JSRLTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class JSRLArgs(JSRLTrainingArgs, EnvBackendArgs):
    """JSRL (Jump-Start RL) -- SAC jump-started by a frozen guide policy.

    Requires ``--guide_checkpoint`` and ``--guide_algorithm``
    (``iql``/``calql``/``wsrl``/``awac``). State observations only.
    """


registry.register("jsrl", JSRLArgs, run_jsrl)
