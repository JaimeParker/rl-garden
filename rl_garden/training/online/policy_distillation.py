"""Teacher-student policy distillation run function.

v1 scope: the teacher is loaded from a PPO checkpoint only
(``PPOPolicy``-shaped). This is not an architectural limit --
``PolicyDistillation`` itself (``rl_garden/algorithms/policy_distillation.py``)
accepts any already-built, already-frozen ``BasePolicy`` as its teacher, so
any rl-garden algorithm can serve as one -- it's just that constructing a
bare policy module from a checkpoint requires knowing which algorithm
produced it, and PPO is the only one wired here. Adding another
``teacher_algorithm`` choice is an additive change to ``_build_teacher_policy``
below, not a redesign.
"""
from __future__ import annotations

from typing import Any


def _policy_distillation_env_request(args, run_name):
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


def _build_teacher_policy(args, env: Any):
    from gymnasium import spaces

    from rl_garden.common.checkpoint import load_checkpoint_file, load_filtered_state_dict
    from rl_garden.encoders.combined import CombinedExtractor
    from rl_garden.policies.ppo_policy import PPOPolicy

    obs_space = env.single_observation_space
    teacher_obs_space = spaces.Dict(
        {key: obs_space[key] for key in args.teacher_obs_keys}
    )
    features_extractor = CombinedExtractor(
        observation_space=teacher_obs_space,
        image_keys=(),
        use_proprio=False,
    )
    teacher_policy = PPOPolicy(
        observation_space=teacher_obs_space,
        action_space=env.single_action_space,
        features_extractor=features_extractor,
        net_arch=list(args.teacher_net_arch),
    )
    checkpoint = load_checkpoint_file(args.teacher_checkpoint, map_location="cpu")
    # PPO's own checkpoint schema (BaseAlgorithm.state_dict()["policy"]) --
    # loaded whole, non-strict, via the shared cross-algorithm primitive
    # rather than a bespoke inline load_state_dict() call. The
    # prefix-selective path (prefix=<submodule>) is for callers building a
    # critic-less target that needs only one submodule's weights out of a
    # full checkpoint -- not needed here since the full PPOPolicy shape is
    # reconstructed above.
    load_filtered_state_dict(
        teacher_policy, checkpoint["state"]["policy"], prefix="", strict=False
    )
    return teacher_policy


def build_policy_distillation(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import PolicyDistillation
    from rl_garden.training.inspection import construct_agent

    if not args.teacher_checkpoint:
        raise ValueError("policy_distillation requires --teacher_checkpoint.")
    if not args.teacher_obs_keys or not args.student_obs_keys:
        raise ValueError(
            "policy_distillation requires both --teacher_obs_keys and "
            "--student_obs_keys (Dict obs keys naming the privileged and "
            "realistic observation groups)."
        )

    teacher_policy = _build_teacher_policy(args, env)
    agent = construct_agent(
        PolicyDistillation,
        env=env,
        eval_env=eval_env,
        teacher_policy=teacher_policy,
        teacher_obs_keys=args.teacher_obs_keys,
        student_obs_keys=args.student_obs_keys,
        num_steps=args.num_steps,
        num_learning_epochs=args.num_learning_epochs,
        num_minibatches=args.num_minibatches,
        loss_type=args.loss_type,
        actor_lr=args.actor_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        net_arch=list(args.net_arch),
        tanh_squash=args.tanh_squash,
        seed=args.seed,
        device=args.device,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_policy_distillation(args: "PolicyDistillationArgs") -> None:
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=_policy_distillation_env_request,
        build_agent=build_policy_distillation,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import PolicyDistillationTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class PolicyDistillationArgs(PolicyDistillationTrainingArgs, EnvBackendArgs):
    """Teacher-student on-policy distillation -- state-only.

    Env backend: ``--env_backend isaaclab`` is the motivating one (see
    ``rl_garden/envs/isaaclab/env.py``'s TODO for exposing a privileged obs
    group), but any Dict-obs backend with the right keys works.
    """


registry.register("policy_distillation", PolicyDistillationArgs, run_policy_distillation)
