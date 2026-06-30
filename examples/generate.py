"""Data generation utilities.

Usage:
    # Generate WSRL-compatible H5 dataset from SAC checkpoints
    python examples/generate.py wsrl_dataset \\
        --checkpoint_dir runs/PickCube-v1__sac_state__1/checkpoints \\
        --output_path demos/pickcube_wsrl_state.h5 \\
        --total_transitions 200000 \\
        --obs_mode state

    # Generate RGB demo H5 by rolling out a privileged state-SAC oracle
    python examples/generate.py rgb_demos \\
        --checkpoint_path runs/<run>/checkpoints/final.pt \\
        --output_path demos/stackcube_state_oracle_rgb.h5 \\
        --total_transitions 50000
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

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
from rl_garden.common.cli_args import (
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
    vit_sac_kwargs_from_args,
)
from rl_garden.common.types import Obs
from rl_garden.datasets import (
    CheckpointScore,
    CollectionStats,
    PolicySource,
    WSRLTrajectoryWriter,
    collect_policy_dataset,
    discover_checkpoints,
    evaluate_policy_success,
    normalize_mix,
    select_policy_sources,
)
from rl_garden.datasets.wsrl_generation import _extract_success, _index_tree, _tree_to_device
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


# ---------------------------------------------------------------------------
# wsrl_dataset subcommand
# ---------------------------------------------------------------------------

@dataclass
class WsrlDatasetArgs:
    """Generate WSRL-compatible H5 dataset from SAC checkpoints."""

    checkpoint_dir: str
    output_path: str
    total_transitions: int

    env_id: str = "PickCube-v1"
    obs_mode: Literal["state", "rgb", "rgbd"] = "state"
    include_state: bool = True
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: Optional[str] = None
    num_envs: int = 16
    seed: int = 1
    device: str = "auto"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    render_mode: str = "rgb_array"

    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    encoder: Literal["plain_conv", "resnet10", "resnet18", "vit"] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["random_shift", "none"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: Optional[int] = None
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False

    policy_mix: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.1)
    tier_thresholds: tuple[float, float, float] = (0.2, 0.6, 0.8)
    eval_episodes: int = 20
    stochastic_collect: bool = False
    use_random_failure_fallback: bool = True
    strict_checkpoint: bool = True
    report_path: Optional[str] = None
    selection_only: bool = False
    source_report_path: Optional[str] = None


def _wsrl_make_env(args: WsrlDatasetArgs):
    return make_maniskill_env(
        ManiSkillEnvConfig(
            env_id=args.env_id,
            num_envs=args.num_envs,
            obs_mode=args.obs_mode,
            include_state=args.include_state,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            reward_mode=args.reward_mode,
            camera_width=args.camera_width if args.obs_mode != "state" else None,
            camera_height=args.camera_height if args.obs_mode != "state" else None,
            ignore_terminations=False,
            record_metrics=True,
        )
    )


def _wsrl_make_agent(args: WsrlDatasetArgs, env):
    common_kwargs = dict(
        env=env,
        eval_env=None,
        buffer_size=max(1024, args.num_envs),
        buffer_device="cpu",
        batch_size=1,
        learning_starts=1,
        eval_freq=0,
        seed=args.seed,
        device=args.device,
        std_log=False,
    )
    if args.obs_mode == "state":
        return SAC(**common_kwargs)

    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_obs_mode(args.obs_mode)
    return SAC(
        **common_kwargs,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
        **vit_sac_kwargs_from_args(args, image_keys),
    )


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _default_report_path(args: WsrlDatasetArgs) -> Path:
    return (
        Path(args.report_path)
        if args.report_path is not None
        else Path(args.output_path).with_suffix(".selection.json")
    )


def _source_from_dict(data: dict) -> PolicySource:
    path = data.get("path")
    return PolicySource(
        tier=data["tier"],
        name=data["name"],
        path=None if path in {None, ""} else Path(path),
        target_transitions=int(data["target_transitions"]),
        success_rate=float(data["success_rate"]),
        fallback_reason=data.get("fallback_reason"),
    )


def _load_sources_from_report(
    path: str | Path,
) -> tuple[list[CheckpointScore], list[PolicySource]]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = [
        CheckpointScore(
            path=Path(score["path"]),
            success_rate=float(score["success_rate"]),
            average_return=float(score["average_return"]),
            average_length=float(score["average_length"]),
            episodes=int(score["episodes"]),
        )
        for score in report.get("scores", [])
    ]
    sources = [_source_from_dict(source) for source in report["sources"]]
    return scores, sources


def _write_report(
    *,
    args: WsrlDatasetArgs,
    scores: list[CheckpointScore],
    sources: list[PolicySource],
    collection_stats: list[dict] | None = None,
) -> Path:
    report = {
        "args": asdict(args),
        "scores": [asdict(score) for score in scores],
        "sources": [asdict(source) for source in sources],
        "collection_stats": collection_stats or [],
        "output_path": args.output_path,
    }
    report_path = _default_report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report_path


def _generate_wsrl_dataset(args: WsrlDatasetArgs) -> None:
    if args.selection_only and args.source_report_path is not None:
        raise ValueError("--selection_only cannot be combined with --source_report_path.")

    seed_everything(args.seed)

    env = _wsrl_make_env(args)
    agent = _wsrl_make_agent(args, env)

    if args.source_report_path is None:
        checkpoint_paths = discover_checkpoints(args.checkpoint_dir)
        scores: list[CheckpointScore] = []
        for path in checkpoint_paths:
            agent.load(
                path,
                strict=args.strict_checkpoint,
                load_replay_buffer=False,
                load_optimizers=False,
            )
            score = evaluate_policy_success(agent, env, episodes=args.eval_episodes)
            scores.append(
                CheckpointScore(
                    path=path,
                    success_rate=score.success_rate,
                    average_return=score.average_return,
                    average_length=score.average_length,
                    episodes=score.episodes,
                )
            )
            print(
                "[eval] "
                f"checkpoint={path.name} "
                f"success_rate={score.success_rate:.3f} "
                f"return={score.average_return:.3f} "
                f"length={score.average_length:.1f}",
                flush=True,
            )

        sources = select_policy_sources(
            scores,
            total_transitions=args.total_transitions,
            policy_mix=args.policy_mix,
            thresholds=args.tier_thresholds,
            use_random_failure_fallback=args.use_random_failure_fallback,
        )
    else:
        scores, sources = _load_sources_from_report(args.source_report_path)

    report_path = _write_report(args=args, scores=scores, sources=sources)
    if args.selection_only:
        print(f"[done] selection_report={report_path}", flush=True)
        env.close()
        return

    metadata = {
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": "" if args.reward_mode is None else args.reward_mode,
        "policy_mix": normalize_mix(args.policy_mix),
        "tier_thresholds": args.tier_thresholds,
        "stochastic_collect": args.stochastic_collect,
    }
    collection_stats: list[dict] = []
    with WSRLTrajectoryWriter(args.output_path, metadata=metadata) as writer:
        for source in sources:
            if source.path is None:
                source_agent = None
            else:
                agent.load(
                    source.path,
                    strict=args.strict_checkpoint,
                    load_replay_buffer=False,
                    load_optimizers=False,
                )
                source_agent = agent
            stats = collect_policy_dataset(
                agent=source_agent,
                env=env,
                writer=writer,
                source=source,
                deterministic=not args.stochastic_collect,
                device=agent.device,
            )
            collection_stats.append(
                {
                    "tier": source.tier,
                    "source": source.name,
                    "target_transitions": source.target_transitions,
                    "written_transitions": stats.transitions,
                    "episodes": stats.episodes,
                    "successes": stats.successes,
                    "success_rate": stats.success_rate,
                }
            )
            print(
                "[collect] "
                f"tier={source.tier} "
                f"source={source.name} "
                f"target_transitions={source.target_transitions} "
                f"written_transitions={stats.transitions} "
                f"episodes={stats.episodes} "
                f"success_rate={stats.success_rate:.3f}",
                flush=True,
            )

    report_path = _write_report(
        args=args,
        scores=scores,
        sources=sources,
        collection_stats=collection_stats,
    )
    print(f"[done] dataset={args.output_path} report={report_path}", flush=True)

    env.close()


# ---------------------------------------------------------------------------
# rgb_demos subcommand
# ---------------------------------------------------------------------------

@dataclass
class RgbDemosArgs:
    """Generate RGB demo H5 by rolling out a privileged state-SAC oracle.

    The oracle policy (a Box(state_dim,) checkpoint, e.g. trained with
    obs_mode="state") drives rollouts in a obs_mode="rgb+state" env. The
    resulting H5 can be consumed via train_off2on.py wsrl --offline_dataset_path.
    """

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


def _rgb_make_env(args: RgbDemosArgs):
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


def _rgb_make_agent(args: RgbDemosArgs, num_envs: int) -> SAC:
    spec = OfflineEnvSpec(
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(args.state_dim,), dtype=np.float32
        ),
        action_space=spaces.Box(
            low=-1.0, high=1.0, shape=(args.action_dim,), dtype=np.float32
        ),
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


def _evaluate_state_policy(
    agent: SAC, env: Any, *, episodes: int, deterministic: bool
) -> dict:
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


def _collect_state_policy_rgb_dataset(
    *,
    agent: SAC,
    env: Any,
    writer: WSRLTrajectoryWriter,
    source: PolicySource,
    target_transitions: int,
    record_state_dim: int,
    deterministic: bool,
) -> CollectionStats:
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
        recorded_final_obs = (
            _record_obs(final_obs, record_state_dim) if final_obs is not None else None
        )

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


def _generate_rgb_demos(args: RgbDemosArgs) -> None:
    seed_everything(args.seed)

    env = _rgb_make_env(args)
    try:
        agent = _rgb_make_agent(args, env.num_envs)
        agent.load(
            args.checkpoint_path,
            strict=args.strict_checkpoint,
            load_replay_buffer=False,
            load_optimizers=False,
        )

        eval_result = _evaluate_state_policy(
            agent,
            env,
            episodes=args.eval_episodes,
            deterministic=not args.stochastic_collect,
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
            stats = _collect_state_policy_rgb_dataset(
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = tyro.cli(
        Union[
            Annotated[WsrlDatasetArgs, tyro.conf.subcommand("wsrl_dataset")],
            Annotated[RgbDemosArgs, tyro.conf.subcommand("rgb_demos")],
        ]
    )
    if isinstance(args, WsrlDatasetArgs):
        _generate_wsrl_dataset(args)
    else:
        _generate_rgb_demos(args)


if __name__ == "__main__":
    main()
