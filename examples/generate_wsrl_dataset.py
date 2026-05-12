"""Generate WSRL-compatible H5 datasets from SAC/RGBDSAC checkpoints.

Example:
    python examples/generate_wsrl_dataset.py \
        --checkpoint_dir runs/PickCube-v1__sac_state__1/checkpoints \
        --output_path demos/pickcube_wsrl_state.h5 \
        --total_transitions 200000 \
        --obs_mode state
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro

try:
    import typeguard

    if not hasattr(typeguard, "TypeCheckError"):
        typeguard.TypeCheckError = TypeError
except ImportError:
    pass

from rl_garden.algorithms import RGBDSAC, SAC
from rl_garden.common import seed_everything
from rl_garden.common.cli_args import image_encoder_factory_from_args, image_keys_from_obs_mode
from rl_garden.datasets import (
    CheckpointScore,
    PolicySource,
    WSRLTrajectoryWriter,
    collect_policy_dataset,
    discover_checkpoints,
    evaluate_policy_success,
    normalize_mix,
    select_policy_sources,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
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
    encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False

    policy_mix: tuple[float, float, float] = (0.3, 0.3, 0.4)
    tier_thresholds: tuple[float, float] = (0.2, 0.8)
    eval_episodes: int = 20
    stochastic_collect: bool = False
    use_random_failure_fallback: bool = True
    strict_checkpoint: bool = True
    report_path: Optional[str] = None
    selection_only: bool = False
    source_report_path: Optional[str] = None


def _make_env(args: Args):
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


def _make_agent(args: Args, env):
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
    return RGBDSAC(
        **common_kwargs,
        image_keys=image_keys_from_obs_mode(args.obs_mode),
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
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


def _default_report_path(args: Args) -> Path:
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


def _load_sources_from_report(path: str | Path) -> tuple[list[CheckpointScore], list[PolicySource]]:
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
    args: Args,
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


def main() -> None:
    args = tyro.cli(Args)
    if args.selection_only and args.source_report_path is not None:
        raise ValueError("--selection_only cannot be combined with --source_report_path.")

    seed_everything(args.seed)

    env = _make_env(args)
    agent = _make_agent(args, env)

    if args.source_report_path is None:
        checkpoint_paths = discover_checkpoints(args.checkpoint_dir)
        scores: list[CheckpointScore] = []
        for path in checkpoint_paths:
            agent.load(path, strict=args.strict_checkpoint, load_replay_buffer=False, load_optimizers=False)
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


if __name__ == "__main__":
    main()
