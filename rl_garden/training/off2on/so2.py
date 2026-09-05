"""SO2 offline-to-online training registration.

Builds ``Off2OnSO2`` and reuses the same ``_runner.run_off2on`` orchestration
as ``wsrl``/``calql``/``iql`` -- SO2's offline gradient-step loop and online
switch are algorithm-agnostic, so no new orchestration code is needed here.

Faithful reproduction of upstream's offline-buffer FIFO churn (see
``so2.py``'s ``_MirroringTensorReplayBuffer``) additionally requires
``--buffer_size`` set to match the loaded dataset's transition count -- a
usage note, not a new mechanism.
"""
from dataclasses import dataclass
from typing import Literal

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.off2on._args import (
    SO2Off2OnTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on._registry import registry


@dataclass
class SO2Off2OnArgs(SO2Off2OnTrainingArgs, EnvBackendArgs):
    """SO2 off2on args: no warmup, mixed replay, fixed ratio.

    For state obs pass --obs_mode state. Env backend:
    ``--env_backend d4rl_legacy`` for D4RL MuJoCo locomotion
    (halfcheetah/hopper/walker2d) or AntMaze, matching the paper's own
    benchmark suite -- already supported, no new backend work needed.
    """

    # run_off2on's shared runner (_runner.py) reads args.obs_mode
    # unconditionally to build the EnvRequest -- unlike run_online, it has no
    # per-algorithm make_env_request callback to hide this behind. Fixed to
    # "state": SO2Core is Box-only (raises TypeError on Dict obs downstream
    # if overridden), so this isn't exposed as a real vision-capable knob.
    obs_mode: str = "state"

    warmup_steps: int = 0
    online_replay_mode: Literal["empty", "append", "mixed"] = "mixed"
    # 1 - concat_online_ratio (upstream's default concat_online_ratio=0.1).
    offline_data_ratio: float | str = 0.9
    bootstrap_at_done: Literal["always", "never", "truncated"] = "always"
    num_eval_episodes: int | None = None


def build_so2(args: SO2Off2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnSO2
    from rl_garden.training.inspection import construct_agent

    agent = construct_agent(
        Off2OnSO2,
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        bootstrap_at_done=args.bootstrap_at_done,
        offline_sampling=args.offline_sampling,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        target_smoothing_noise_std=args.target_smoothing_noise_std,
        target_smoothing_noise_clip_min=args.target_smoothing_noise_clip_min,
        target_smoothing_noise_clip_max=args.target_smoothing_noise_clip_max,
        initial_training_phase=initial_training_phase_from_args(args),
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_so2(args: SO2Off2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_so2, algorithm="so2")


registry.register("so2", SO2Off2OnArgs, run_so2)
