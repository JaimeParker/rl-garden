"""Residual RL + HIL-SERL-style split actor/learner training."""
from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Optional, Sequence

import torch


def _residual_hil_serl_env_request(args, run_name, *, create_eval_env: bool = False):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest

    is_visual = args.obs_mode != "state"
    backend_config = args.resolve_backend_config()
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    return EnvRequest(
        env_id=args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width if is_visual else None,
        camera_height=args.camera_height if is_visual else None,
        include_state=args.include_state if is_visual else True,
        per_camera_rgbd=args.per_camera_rgbd if is_visual else False,
        frame_stack=args.frame_stack,
        reward_scale=1.0,
        reward_bias=0.0,
        num_eval_envs=1,
        create_eval_env=create_eval_env,
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )


def _build_env(args, env_request, *, enable_teleop: bool, enable_classifier: bool):
    from rl_garden.envs.backend_registry import make_training_envs

    if args.env_backend not in {"franka_real", "maniskill"}:
        raise ValueError(
            "residual_hil_serl currently supports only env_backend='franka_real' "
            f"or 'maniskill', got {args.env_backend!r}."
        )
    if args.control_mode != "pd_ee_twist":
        raise ValueError(
            "residual_hil_serl requires --control_mode pd_ee_twist so teleop "
            f"actions have the expected twist convention; got {args.control_mode!r}."
        )

    env, _ = make_training_envs(args.env_backend, env_request)
    if getattr(env, "num_envs", 1) != 1:
        raise ValueError(f"HITL actor/learner expects num_envs=1, got {env.num_envs}.")

    if args.env_backend == "franka_real" and args.convert_obs_rotation:
        from rl_garden.envs.wrappers.rotvec_obs import RotvecObsWrapper

        env = RotvecObsWrapper(env)

    if (
        args.env_backend == "franka_real"
        and enable_classifier
        and args.classifier_checkpoint is not None
    ):
        from rl_garden.common.utils import get_device
        from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
        from rl_garden.models.reward.success.model import load_classifier_fn

        image_keys = args.classifier_image_keys or tuple(
            k for k in env.single_observation_space.spaces if k != "state"
        )
        classifier_fn = load_classifier_fn(
            args.classifier_checkpoint,
            env.single_observation_space,
            image_keys,
            device=get_device("auto"),
        )
        env = RewardClassifierWrapper(env, classifier_fn, threshold=args.classifier_threshold)

    if enable_teleop:
        from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

        env = TeleopInterventionWrapper(
            env,
            device=args.teleop_device,
            record_gripper=args.teleop_record_gripper,
            teleop_init_timeout_s=args.teleop_init_timeout_s,
        )

    return env


def build_residual_hil_serl(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ResidualHilSerlSAC
    from rl_garden.common.cli_args import (
        image_encoder_factory_from_args,
        image_keys_from_env,
        vit_sac_kwargs_from_args,
    )
    from rl_garden.training.online._args import sac_initial_training_phase_from_args
    from rl_garden.training.online.residual_sac import _make_base_action_provider

    if args.load_actor_checkpoint is not None:
        raise ValueError(
            "ResidualHilSerlSAC does not support --load_actor_checkpoint; "
            "use --load_checkpoint for ResidualHilSerlSAC checkpoints or "
            "--base_ckpt_path for the frozen base policy."
        )

    is_visual = args.obs_mode != "state"
    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    image_kwargs: dict[str, Any] = {}
    if is_visual:
        factory = image_encoder_factory_from_args(args)
        image_keys = image_keys_from_env(env, args)
        image_kwargs = dict(
            image_keys=image_keys,
            image_encoder_factory=factory,
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=args.frame_stack > 1,
            image_augmentation=args.image_augmentation,
            random_shift_pad=args.image_random_shift_pad,
            image_augmentation_seed=args.seed + 1_000_003,
            **vit_sac_kwargs_from_args(args, image_keys),
        )

    base_action_provider = _make_base_action_provider(args, env)
    agent = ResidualHilSerlSAC(
        env=env,
        eval_env=eval_env,
        base_action_provider=base_action_provider,
        residual_action_scale=args.residual_action_scale,
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
        **image_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.offline_dataset_path is not None:
        loaded = agent.load_offline_replay_buffer(
            args.offline_dataset_path,
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
        )
        if args.std_log:
            print(
                "[residual_hil_serl] "
                f"offline_dataset={args.offline_dataset_path} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio}",
                flush=True,
            )
    return agent


class ResidualHilSerlActorLoop:
    """Actor loop that pushes ResidualSAC replay fields over the sync client."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        sync_client: Any,
        control_hz: float = 10.0,
        deterministic: bool = False,
        seed: int = 1,
    ) -> None:
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError(
                f"ResidualHilSerlActorLoop drives exactly one env; got env.num_envs="
                f"{getattr(env, 'num_envs', None)!r}."
            )
        self.env = env
        self.agent = agent
        self.sync_client = sync_client
        self.control_period = 1.0 / control_hz
        self.deterministic = deterministic
        self.seed = seed

    def _maybe_refresh_policy(self) -> None:
        params = self.sync_client.latest_policy_params()
        if params is not None:
            self.agent.policy.load_state_dict(params)

    def _select_action(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        agent = self.agent
        with torch.no_grad():
            if agent._cached_base_actions is None:
                base_actions = agent._base_naction(obs)
            else:
                base_actions = agent._cached_base_actions
                agent._cached_base_actions = None
            unit_residual = agent.policy.predict(
                agent._obs_to_policy_device(obs),
                base_actions=base_actions,
                deterministic=self.deterministic,
            ).detach()
            final_actions = agent._combine_base_residual(base_actions, unit_residual)
            env_action = agent.action_scaler.unscale(final_actions)
        return env_action, final_actions, base_actions

    def run(self, total_steps: Optional[int] = None) -> None:
        self.sync_client.start()
        self.agent.policy.eval()
        try:
            obs, _ = self.env.reset(seed=self.seed)
            self.agent._on_env_reset(obs)
            step = 0
            while total_steps is None or step < total_steps:
                loop_start = time.perf_counter()
                self._maybe_refresh_policy()

                env_action, final_actions, base_actions = self._select_action(obs)
                env_action = env_action.to(self._env_device(obs))
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)

                intervened = "intervene_action" in info
                replay_action = final_actions
                if intervened:
                    replay_action = self.agent.action_scaler.scale(
                        info["intervene_action"].to(self.agent.device)
                    ).clamp(-1.0, 1.0)

                done = terminated | truncated
                with torch.no_grad():
                    next_base_actions = self.agent._base_naction(next_obs)

                self.sync_client.push_transition(
                    {
                        "obs": obs,
                        "next_obs": next_obs,
                        "action": replay_action,
                        "reward": reward,
                        "done": done,
                        "base_actions": base_actions,
                        "next_base_actions": next_base_actions,
                        "intervened": intervened,
                    }
                )

                if bool(done.any()):
                    obs, _ = self.env.reset(seed=self.seed)
                    self.agent._on_env_reset(obs)
                else:
                    obs = next_obs
                    self.agent._cached_base_actions = next_base_actions.detach()
                step += 1

                sleep_for = self.control_period - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.sync_client.stop()

    @staticmethod
    def _env_device(obs) -> torch.device:
        sample = next(iter(obs.values())) if isinstance(obs, dict) else obs
        return sample.device


def _run_actor(args) -> None:
    from rl_garden.real_world.hil_serl import HilSerlActorSyncClient

    env_request = _residual_hil_serl_env_request(
        args, run_name="hitl_residual_actor", create_eval_env=False
    )
    env = _build_env(args, env_request, enable_teleop=True, enable_classifier=True)
    scratch_args = dataclasses.replace(
        args,
        buffer_size=8,
        load_checkpoint=None,
        offline_dataset_path=None,
        eval_freq=0,
        num_envs=1,
        num_eval_envs=1,
    )
    agent = build_residual_hil_serl(
        scratch_args, env, None, logger=None, checkpoint_dir=None
    )
    sync_client = HilSerlActorSyncClient(f"http://{args.sync_host}:{args.sync_port}")
    loop = ResidualHilSerlActorLoop(
        env,
        agent,
        sync_client,
        control_hz=args.control_hz,
        deterministic=args.deterministic_actor,
        seed=args.seed,
    )
    loop.run()


def _run_learner(args) -> None:
    from rl_garden.common import Logger, seed_everything
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden.real_world.hil_serl import HilSerlLearnerLoop

    seed_everything(args.seed)
    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__residual_hil_serl__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.log_dir, run_name, "checkpoints")

    resolved_config = persist_resolved_config(
        args,
        training_phase="hitl",
        algorithm="residual_hil_serl",
        run_name=run_name,
        log_dir=args.log_dir,
    )
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=resolved_config,
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group or args.env_id,
    )

    env_request = _residual_hil_serl_env_request(
        args, run_name=run_name, create_eval_env=False
    )
    env = _build_env(args, env_request, enable_teleop=False, enable_classifier=False)
    agent = build_residual_hil_serl(
        args, env, None, logger=logger, checkpoint_dir=checkpoint_dir
    )
    agent.init_demo_buffer(args.demo_buffer_size, args.demo_data_ratio)

    loop = HilSerlLearnerLoop(
        agent,
        host=args.sync_host,
        port=args.sync_port,
        checkpoint_dir=checkpoint_dir,
        buffer_period=args.buffer_period,
        demo_dataset_paths=args.demo_dataset_paths,
        train_freq=args.train_freq,
        publish_freq=args.publish_freq,
    )
    try:
        loop.run()
    finally:
        logger.close()


def _run_eval(args) -> None:
    if args.load_checkpoint is None:
        raise ValueError("residual_hil_serl eval requires --load_checkpoint.")
    env_request = _residual_hil_serl_env_request(
        args, run_name="hitl_residual_eval", create_eval_env=False
    )
    env = _build_env(args, env_request, enable_teleop=False, enable_classifier=True)
    eval_args = dataclasses.replace(
        args,
        buffer_size=8,
        offline_dataset_path=None,
        eval_freq=0,
        num_envs=1,
        num_eval_envs=1,
    )
    agent = build_residual_hil_serl(
        eval_args, env, None, logger=None, checkpoint_dir=None
    )
    agent.policy.eval()

    successes = 0
    returns: list[float] = []
    obs, _ = env.reset(seed=args.seed)
    agent._on_env_reset(obs)
    for episode in range(args.eval_n_trajs):
        episode_return = 0.0
        while True:
            action = agent.get_action(obs, deterministic=True, return_info=False)
            obs, reward, terminated, truncated, info = env.step(
                action.to(ResidualHilSerlActorLoop._env_device(obs))
            )
            episode_return += float(reward.sum())
            if bool((terminated | truncated).any()):
                success = bool(terminated.any())
                successes += int(success)
                returns.append(episode_return)
                print(
                    f"[eval] episode={episode} success={success} "
                    f"return={episode_return:.3f}",
                    flush=True,
                )
                obs, _ = env.reset(seed=args.seed)
                agent._on_env_reset(obs)
                break

    success_rate = successes / args.eval_n_trajs
    mean_return = sum(returns) / len(returns) if returns else 0.0
    print(
        f"[eval] success_rate={success_rate:.3f} mean_return={mean_return:.3f} "
        f"n_trajs={args.eval_n_trajs}",
        flush=True,
    )


def run_residual_hil_serl(args: "ResidualHilSerlArgs") -> None:
    if args.role == "actor":
        _run_actor(args)
    elif args.role == "learner":
        _run_learner(args)
    else:
        _run_eval(args)


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402

from rl_garden.training.hitl._args import HITLArgs  # noqa: E402
from rl_garden.training.hitl._registry import registry  # noqa: E402
from rl_garden.training.online.residual_sac import ResidualSACArgs  # noqa: E402
from rl_garden.training.real_world._args import FrankaRealConfig  # noqa: E402


@dataclass
class ResidualHilSerlArgs(HITLArgs, ResidualSACArgs):
    """Residual RL with HIL-SERL-style split actor/learner intervention data."""

    franka_real: FrankaRealConfig = field(default_factory=FrankaRealConfig)
    env_backend: str = "maniskill"
    num_envs: int = 1
    num_eval_envs: int = 1
    control_mode: str = "pd_ee_twist"

    classifier_checkpoint: Optional[str] = None
    classifier_threshold: float = 0.5
    classifier_image_keys: Sequence[str] = field(default_factory=tuple)
    convert_obs_rotation: bool = False


registry.register("residual_hil_serl", ResidualHilSerlArgs, run_residual_hil_serl)
