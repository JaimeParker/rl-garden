"""Generate RGB demo H5 by rolling out a privileged state-SAC oracle.

The oracle policy (a ``Box(state_dim,)`` checkpoint, e.g. trained with
``obs_mode="state"``) drives rollouts in a ``obs_mode="rgb+state"`` env. Each
recorded transition keeps ``rgb_<camera>`` plus ``state[:, :record_state_dim]``,
matching a vision SAC's ``obs_mode="rgb", per_camera_rgbd=True`` observation
space, so the resulting H5 can be consumed via
``train_wsrl_rgbd.py --offline_dataset_path``.

The oracle's full ``state`` (``state_dim``) is fed to the policy directly;
``record_state_dim`` only controls how much of it is written to the H5.

Example:
    python examples/generate_state_policy_rgb_demos.py \
        --checkpoint_path runs/stackcube_sac_state_joint_delta_1m_seed1__20260609/checkpoints/final.pt \
        --output_path demos/stackcube_state_oracle_rgb.h5 \
        --total_transitions 50000
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

try:
    import typeguard

    if not hasattr(typeguard, "TypeCheckError"):
        typeguard.TypeCheckError = TypeError
except ImportError:
    pass

from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.common import seed_everything
from rl_garden.common.types import Obs
from rl_garden.datasets import CollectionStats, PolicySource, WSRLTrajectoryWriter
from rl_garden.datasets.wsrl_generation import _extract_success, _index_tree, _tree_to_device
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    checkpoint_path: str
    output_path: str
    total_transitions: int

    env_id: str = "StackCube-v1"
    control_mode: str = "pd_joint_delta_pos"
    state_dim: int = 48
    record_state_dim: int = 25
    action_dim: int = 8
    num_envs: int = 16
    camera_width: int = 64
    camera_height: int = 64
    seed: int = 1
    device: str = "auto"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    eval_episodes: int = 20
    eval_only: bool = False
    stochastic_collect: bool = False
    strict_checkpoint: bool = True


def _make_env(args: Args):
    return make_maniskill_env(
        ManiSkillEnvConfig(
            env_id=args.env_id,
            num_envs=args.num_envs,
            obs_mode="rgb+state",
            include_state=True,
            control_mode=args.control_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            per_camera_rgbd=True,
            ignore_terminations=False,
            record_metrics=True,
        )
    )


def _make_agent(args: Args, num_envs: int) -> SAC:
    spec = OfflineEnvSpec(
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(args.state_dim,), dtype=np.float32
        ),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(args.action_dim,), dtype=np.float32),
        num_envs=num_envs,
    )
    return SAC(
        env=spec,
        eval_env=None,
        buffer_size=max(1024, num_envs),
        buffer_device="cpu",
        batch_size=1,
        learning_starts=1,
        eval_freq=0,
        seed=args.seed,
        device=args.device,
        std_log=False,
    )


def _record_obs(obs: Obs, record_state_dim: int) -> Obs:
    assert isinstance(obs, dict)
    recorded = dict(obs)
    recorded["state"] = obs["state"][:, :record_state_dim]
    return recorded


def evaluate_state_policy(agent: SAC, env: Any, *, episodes: int, deterministic: bool) -> dict:
    """Evaluate the oracle policy (reads ``obs["state"]``) in a Dict-obs env."""
    obs, _ = env.reset()
    returns = torch.zeros(env.num_envs)
    lengths = torch.zeros(env.num_envs)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    successes = 0

    while len(completed_returns) < episodes:
        policy_obs = _tree_to_device(obs["state"], agent.device)
        with torch.no_grad():
            actions = agent.policy.predict(policy_obs, deterministic=deterministic).detach()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        rewards_cpu = rewards.detach().cpu().float()
        done_cpu = (terminations | truncations).detach().cpu().bool()
        returns += rewards_cpu
        lengths += 1

        for env_idx, done in enumerate(done_cpu.tolist()):
            if not done or len(completed_returns) >= episodes:
                continue
            success = _extract_success(infos, env_idx)
            successes += int(bool(success))
            completed_returns.append(float(returns[env_idx].item()))
            completed_lengths.append(float(lengths[env_idx].item()))
            returns[env_idx] = 0
            lengths[env_idx] = 0
        obs = next_obs

    return {
        "success_rate": successes / episodes,
        "average_return": float(np.mean(completed_returns)),
        "average_length": float(np.mean(completed_lengths)),
        "episodes": episodes,
    }


def collect_state_policy_rgb_dataset(
    *,
    agent: SAC,
    env: Any,
    writer: WSRLTrajectoryWriter,
    source: PolicySource,
    target_transitions: int,
    record_state_dim: int,
    deterministic: bool,
) -> CollectionStats:
    """Roll out the oracle and write ``rgb_*`` + ``state[:record_state_dim]`` transitions."""
    if target_transitions <= 0:
        return CollectionStats()

    obs, _ = env.reset()
    recorded_obs = _record_obs(obs, record_state_dim)
    episode_obs: list[list[Obs]] = [[_index_tree(recorded_obs, i)] for i in range(env.num_envs)]
    episode_actions: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    episode_rewards: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    episode_terminated: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    episode_truncated: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    stats = CollectionStats()
    print(f"[collect] starting collection for source {source.name}")

    while stats.transitions < target_transitions:
        policy_obs = _tree_to_device(obs["state"], agent.device)
        with torch.no_grad():
            actions = agent.policy.predict(policy_obs, deterministic=deterministic).detach()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = (terminations | truncations).detach().cpu().bool()

        recorded_next_obs = _record_obs(next_obs, record_state_dim)
        final_obs = infos.get("final_observation")
        recorded_final_obs = _record_obs(final_obs, record_state_dim) if final_obs is not None else None

        for env_idx in range(env.num_envs):
            action_i = _index_tree(actions, env_idx)
            reward_i = rewards[env_idx].detach().cpu()
            term_i = terminations[env_idx].detach().cpu()
            trunc_i = truncations[env_idx].detach().cpu()
            if bool(done[env_idx]) and recorded_final_obs is not None:
                next_obs_i = _index_tree(recorded_final_obs, env_idx)
            else:
                next_obs_i = _index_tree(recorded_next_obs, env_idx)

            episode_actions[env_idx].append(action_i)
            episode_rewards[env_idx].append(reward_i)
            episode_terminated[env_idx].append(term_i)
            episode_truncated[env_idx].append(trunc_i)
            episode_obs[env_idx].append(next_obs_i)

            if not bool(done[env_idx]):
                continue

            success = bool(_extract_success(infos, env_idx))
            written = writer.write_episode(
                obs=episode_obs[env_idx],
                actions=episode_actions[env_idx],
                rewards=episode_rewards[env_idx],
                terminated=episode_terminated[env_idx],
                truncated=episode_truncated[env_idx],
                source=source,
                success=success,
            )
            stats.episodes += 1
            stats.transitions += written
            stats.successes += int(success)
            episode_obs[env_idx] = [_index_tree(recorded_next_obs, env_idx)]
            episode_actions[env_idx] = []
            episode_rewards[env_idx] = []
            episode_terminated[env_idx] = []
            episode_truncated[env_idx] = []

        obs = next_obs

    print(
        f"[collect] finished collection for source {source.name}: "
        f"{stats.episodes} episodes, {stats.transitions} transitions, "
        f"success rate {stats.success_rate:.2%}"
    )
    return stats


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)

    env = _make_env(args)
    try:
        agent = _make_agent(args, env.num_envs)
        agent.load(
            args.checkpoint_path,
            strict=args.strict_checkpoint,
            load_replay_buffer=False,
            load_optimizers=False,
        )

        eval_result = evaluate_state_policy(
            agent, env, episodes=args.eval_episodes, deterministic=not args.stochastic_collect
        )
        print(
            "[eval] "
            f"checkpoint={args.checkpoint_path} "
            f"success_rate={eval_result['success_rate']:.3f} "
            f"return={eval_result['average_return']:.3f} "
            f"length={eval_result['average_length']:.1f}",
            flush=True,
        )
        if args.eval_only:
            return

        source = PolicySource(
            tier="success",
            name=Path(args.checkpoint_path).stem,
            path=Path(args.checkpoint_path),
            target_transitions=args.total_transitions,
            success_rate=eval_result["success_rate"],
        )
        metadata = {
            "env_id": args.env_id,
            "obs_mode": "rgb",
            "control_mode": args.control_mode,
            "source_checkpoint": str(args.checkpoint_path),
            "stochastic_collect": args.stochastic_collect,
        }
        with WSRLTrajectoryWriter(args.output_path, metadata=metadata) as writer:
            stats = collect_state_policy_rgb_dataset(
                agent=agent,
                env=env,
                writer=writer,
                source=source,
                target_transitions=args.total_transitions,
                record_state_dim=args.record_state_dim,
                deterministic=not args.stochastic_collect,
            )

        print(
            f"[done] dataset={args.output_path} episodes={stats.episodes} "
            f"transitions={stats.transitions} success_rate={stats.success_rate:.3f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
