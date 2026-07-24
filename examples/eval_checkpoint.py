"""Evaluate a saved rl-garden checkpoint in a live environment.

This is a generic checkpoint evaluation entrypoint: it builds the requested
algorithm, loads model weights without replay/optimizer state, then runs policy
evaluation until the requested number of completed episodes is collected.

Example:

    python -u tools/evaluation/eval_checkpoint.py \
      --phase offline --algorithm calql \
      --checkpoint-path runs/.../calql_offline_pretrained.pt \
      --env-id PickCube-v1 --obs-mode state \
      --num-eval-envs 16 --num-eval-episodes 50 \
      --capture-video false --output-json /tmp/calql_eval.json
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
import importlib
import json
from pathlib import Path
import re
import time
from typing import Any, Literal, Mapping, Optional

import torch
import tyro

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.checkpoint import _canonical_algorithm_class, load_checkpoint_file
from rl_garden.common.env_args import (
    EnvBackendArgs,
    ManiSkillConfig,
    MinariConfig,
    MujocoConfig,
    MujocoWarpConfig,
    RoboTwinConfig,
)
from rl_garden.common.resolved_config import resolved_config_json
from rl_garden.envs.backend_registry import (
    EnvRequest,
    make_evaluation_env,
    make_training_envs,
)

Phase = Literal["auto", "online", "offline", "off2on"]


@dataclass
class EvalCheckpointArgs(EnvBackendArgs):
    checkpoint_path: str = ""
    phase: Phase = "auto"
    algorithm: str = "auto"
    config_path: Optional[str] = None

    env_id: str = "PickCube-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"
    num_eval_envs: int = 16
    num_eval_episodes: int = 50
    max_eval_steps: Optional[int] = None
    seed: int = 1
    device: str = "auto"
    buffer_device: str = "cpu"

    include_state: bool = True
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    per_camera_rgbd: bool = False
    frame_stack: int = 1
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    capture_video: bool = False
    video_fps: int = 30
    eval_output_dir: Optional[str] = None
    output_json: Optional[str] = None
    strict: bool = True

    log_type: str = "none"
    std_log: bool = True

    maniskill: ManiSkillConfig = field(default_factory=ManiSkillConfig)
    robotwin: RoboTwinConfig = field(default_factory=RoboTwinConfig)
    minari: MinariConfig = field(default_factory=MinariConfig)
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)
    mujoco_warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)


_PHASES: tuple[str, ...] = ("online", "offline", "off2on")


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _algorithm_from_class(algorithm_class: Any) -> str | None:
    if not isinstance(algorithm_class, str):
        return None
    canonical = _canonical_algorithm_class(algorithm_class)
    name = str(canonical)
    if name.startswith("Off2On"):
        name = name.removeprefix("Off2On")
    known = {
        "AWAC": "awac",
        "BC": "bc",
        "CalQL": "calql",
        "CQL": "cql",
        "DrQv2": "drqv2",
        "FlashSAC": "flash_sac",
        "IQL": "iql",
        "PPO": "ppo",
        "RLPD": "rlpd",
        "SAC": "sac",
        "TD3": "td3",
        "TD3BC": "td3_bc",
        "TDMPC2": "tdmpc2",
        "WSRL": "wsrl",
    }
    if name in known:
        return known[name]
    return _snake_case(name)


def _registry_for_phase(phase: str):
    module = importlib.import_module(f"rl_garden.training.{phase}._registry")
    registry = module.registry
    registry.discover()
    return registry


def _resolve_phase_and_algorithm(
    requested_phase: str,
    requested_algorithm: str,
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    metadata = checkpoint.get("metadata", {})
    inferred_algorithm = _algorithm_from_class(metadata.get("algorithm_class"))
    algorithm = requested_algorithm if requested_algorithm != "auto" else inferred_algorithm
    if algorithm is None:
        raise SystemExit(
            "Could not infer algorithm from checkpoint metadata; pass --algorithm."
        )

    phases = _PHASES if requested_phase == "auto" else (requested_phase,)
    matches: list[tuple[str, str]] = []
    for phase in phases:
        registry = _registry_for_phase(phase)
        if algorithm in registry.entries():
            matches.append((phase, algorithm))

    if not matches:
        raise SystemExit(
            f"Algorithm {algorithm!r} is not registered for phase "
            f"{requested_phase!r}."
        )
    if requested_phase == "auto" and len(matches) > 1:
        choices = ", ".join(f"{phase}:{alg}" for phase, alg in matches)
        raise SystemExit(
            f"Checkpoint algorithm {algorithm!r} is ambiguous across phases "
            f"({choices}); pass --phase explicitly."
        )
    return matches[0]


def _set_existing_attr(target: Any, key: str, value: Any) -> None:
    if not hasattr(target, key):
        return
    current = getattr(target, key)
    if is_dataclass(current) and isinstance(value, Mapping):
        _apply_mapping_to_args(current, value)
    else:
        setattr(target, key, value)


def _apply_mapping_to_args(args: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        _set_existing_attr(args, key, value)


def _config_args(config_path: Path) -> Mapping[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config.get("args", config)
    if not isinstance(args, Mapping):
        raise SystemExit(f"Config {config_path} does not contain an args mapping.")
    return args


def _default_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent.parent / "config.json"


def _load_training_args(
    phase: str,
    algorithm: str,
    eval_args: EvalCheckpointArgs,
    checkpoint: Mapping[str, Any],
) -> Any:
    registry = _registry_for_phase(phase)
    entry = registry.entries()[algorithm]
    args = entry.args_cls()

    config_path = (
        Path(eval_args.config_path)
        if eval_args.config_path is not None
        else _default_config_path(Path(eval_args.checkpoint_path))
    )
    if config_path.exists():
        _apply_mapping_to_args(args, _config_args(config_path))

    metadata = checkpoint.get("metadata", {})
    hyperparameters = metadata.get("hyperparameters", {})
    if isinstance(hyperparameters, Mapping):
        _apply_mapping_to_args(args, hyperparameters)

    overrides = {
        "env_backend": eval_args.env_backend,
        "env_id": eval_args.env_id,
        "obs_mode": eval_args.obs_mode,
        "control_mode": eval_args.control_mode,
        "render_mode": eval_args.render_mode,
        "num_envs": eval_args.num_eval_envs,
        "num_eval_envs": eval_args.num_eval_envs,
        "num_eval_steps": eval_args.max_eval_steps
        or max(eval_args.num_eval_episodes * 1000, 1),
        "seed": eval_args.seed,
        "device": eval_args.device,
        "buffer_device": eval_args.buffer_device,
        "include_state": eval_args.include_state,
        "camera_width": eval_args.camera_width,
        "camera_height": eval_args.camera_height,
        "per_camera_rgbd": eval_args.per_camera_rgbd,
        "frame_stack": eval_args.frame_stack,
        "reward_scale": eval_args.reward_scale,
        "reward_bias": eval_args.reward_bias,
        "capture_video": eval_args.capture_video,
        "video_fps": eval_args.video_fps,
        "eval_output_dir": eval_args.eval_output_dir,
        "eval_freq": 1,
        "log_type": eval_args.log_type,
        "std_log": eval_args.std_log,
        "load_checkpoint": None,
        "load_replay_buffer": False,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": False,
        "save_final_checkpoint": False,
    }
    _apply_mapping_to_args(args, overrides)
    _apply_mapping_to_args(
        args,
        {
            "maniskill": eval_args.maniskill,
            "robotwin": eval_args.robotwin,
            "minari": eval_args.minari,
            "mujoco": eval_args.mujoco,
            "mujoco_warp": eval_args.mujoco_warp,
        },
    )
    return args


def _eval_record_dir(eval_args: EvalCheckpointArgs) -> str | None:
    if not eval_args.capture_video:
        return None
    if eval_args.eval_output_dir is not None:
        return eval_args.eval_output_dir
    checkpoint_path = Path(eval_args.checkpoint_path)
    return str(checkpoint_path.parent / "eval_videos")


def _env_request(args: Any, eval_args: EvalCheckpointArgs) -> EnvRequest:
    is_visual = args.obs_mode != "state"
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width if is_visual else None,
        camera_height=args.camera_height if is_visual else None,
        include_state=args.include_state if is_visual else True,
        per_camera_rgbd=args.per_camera_rgbd if is_visual else False,
        frame_stack=getattr(args, "frame_stack", 1),
        reward_scale=getattr(args, "reward_scale", 1.0),
        reward_bias=getattr(args, "reward_bias", 0.0),
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=_eval_record_dir(eval_args),
        capture_video=eval_args.capture_video,
        video_fps=eval_args.video_fps,
        num_eval_steps=args.num_eval_steps,
        create_eval_env=True,
        backend_config=args.resolve_backend_config(),
    )


def _training_module(phase: str, algorithm: str):
    return importlib.import_module(
        f"rl_garden.training.{phase}.{algorithm.replace('-', '_')}"
    )


def _builder_from_module(module: Any, algorithm: str):
    expected = f"build_{algorithm.replace('-', '_')}"
    if hasattr(module, expected):
        return getattr(module, expected)
    builders = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("build_") and callable(getattr(module, name))
    ]
    if len(builders) != 1:
        raise SystemExit(
            f"Could not identify a unique build_* function in {module.__name__}."
        )
    return builders[0]


def _build_agent(
    phase: str,
    algorithm: str,
    args: Any,
    eval_args: EvalCheckpointArgs,
):
    logger = Logger(log_type="none")
    req = _env_request(args, eval_args)
    module = _training_module(phase, algorithm)
    build_agent = _builder_from_module(module, algorithm)

    if phase == "offline":
        eval_env = make_evaluation_env(args.env_backend, req)
        env_spec = OfflineEnvSpec(
            eval_env.single_observation_space,
            eval_env.single_action_space,
            num_envs=1,
        )
        agent = build_agent(args, env_spec, logger, eval_env)
        return agent, None, eval_env

    train_env, eval_env = make_training_envs(args.env_backend, req)
    agent = build_agent(args, train_env, eval_env, logger, None)
    return agent, train_env, eval_env


def _reset_env(env: Any, seed: int):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _to_cpu_1d(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)
    return torch.as_tensor(value).detach().cpu().reshape(-1)


def _done_mask(infos: Mapping[str, Any], terminations: Any, truncations: Any) -> torch.Tensor:
    if "_final_info" in infos:
        return _to_cpu_1d(infos["_final_info"]).bool()
    return (_to_cpu_1d(terminations).bool() | _to_cpu_1d(truncations).bool())


def _append_metric_values(
    metrics: dict[str, list[float]],
    episode: Mapping[str, Any],
    mask: torch.Tensor,
    remaining: int,
) -> int:
    appended = 0
    for key, value in episode.items():
        values = _to_cpu_1d(value)
        if values.numel() == mask.numel():
            values = values[mask]
        take = min(values.numel(), remaining)
        if take <= 0:
            continue
        metrics.setdefault(key, []).extend(float(v) for v in values[:take])
        appended = max(appended, take)
    return appended


def evaluate_agent(
    agent: Any,
    eval_env: Any,
    *,
    seed: int,
    num_eval_episodes: int,
    max_eval_steps: int,
) -> dict[str, float]:
    if eval_env is None:
        raise SystemExit("Evaluation environment was not created.")

    agent.policy.eval()
    obs, _ = _reset_env(eval_env, seed)
    agent._eval_start_hook()

    metrics: dict[str, list[float]] = {}
    running_returns = torch.zeros(eval_env.num_envs, dtype=torch.float32)
    completed = 0
    steps = 0

    try:
        while completed < num_eval_episodes and steps < max_eval_steps:
            with torch.no_grad():
                env_action, critic_action = agent._eval_action_and_critic_action(obs)
                obs_before = obs
                obs, rewards, terminations, truncations, infos = eval_env.step(env_action)
                agent._eval_step_hook(
                    obs_before,
                    critic_action,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                )

            rewards_cpu = _to_cpu_1d(rewards).float()
            running_returns[: rewards_cpu.numel()] += rewards_cpu
            mask = _done_mask(infos, terminations, truncations)
            remaining = num_eval_episodes - completed

            appended = 0
            if isinstance(infos, Mapping) and "final_info" in infos:
                final_info = infos["final_info"]
                if isinstance(final_info, Mapping) and "episode" in final_info:
                    episode = final_info["episode"]
                    if isinstance(episode, Mapping):
                        appended = _append_metric_values(
                            metrics, episode, mask, remaining
                        )

            if appended == 0 and mask.any():
                done_returns = running_returns[mask[: running_returns.numel()]]
                take = min(done_returns.numel(), remaining)
                metrics.setdefault("return", []).extend(
                    float(v) for v in done_returns[:take]
                )
                appended = take

            if mask.any():
                running_returns[mask[: running_returns.numel()]] = 0.0
            completed += appended
            steps += 1
    finally:
        agent.policy.train()

    if completed < num_eval_episodes:
        raise SystemExit(
            f"Only completed {completed}/{num_eval_episodes} episodes within "
            f"max_eval_steps={max_eval_steps}."
        )

    out = {
        key: float(torch.tensor(values, dtype=torch.float32).mean().item())
        for key, values in sorted(metrics.items())
        if values
    }
    out["episodes_completed"] = float(completed)
    out["eval_steps"] = float(steps)
    out.update(agent._eval_finalize_hook())
    return out


def _headline_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "return" in metrics:
        out["average_return"] = metrics["return"]
    for key in ("success_at_end", "success_once", "success"):
        if key in metrics:
            out["success_rate"] = metrics[key]
            break
    for key in ("success_at_end", "success_once", "success", "episodes_completed"):
        if key in metrics:
            out[key] = metrics[key]
    return out


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(resolved_config_json(payload) + "\n", encoding="utf-8")


def evaluate_checkpoint(eval_args: EvalCheckpointArgs) -> dict[str, Any]:
    if not eval_args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required.")
    if eval_args.num_eval_episodes <= 0:
        raise SystemExit("--num-eval-episodes must be positive.")

    checkpoint_path = Path(eval_args.checkpoint_path)
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    phase, algorithm = _resolve_phase_and_algorithm(
        eval_args.phase,
        eval_args.algorithm,
        checkpoint,
    )
    args = _load_training_args(phase, algorithm, eval_args, checkpoint)
    seed_everything(args.seed)
    enable_fast_math()

    agent, train_env, eval_env = _build_agent(phase, algorithm, args, eval_args)
    try:
        agent.load(
            checkpoint_path,
            strict=eval_args.strict,
            load_replay_buffer=False,
            load_optimizers=False,
        )
        max_eval_steps = eval_args.max_eval_steps or max(eval_args.num_eval_episodes * 1000, 1)
        metrics = evaluate_agent(
            agent,
            eval_env,
            seed=eval_args.seed,
            num_eval_episodes=eval_args.num_eval_episodes,
            max_eval_steps=max_eval_steps,
        )
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()

    result = {
        "checkpoint_path": str(checkpoint_path),
        "phase": phase,
        "algorithm": algorithm,
        "env_backend": args.env_backend,
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "num_eval_envs": args.num_eval_envs,
        "num_eval_episodes": eval_args.num_eval_episodes,
        "timestamp": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        "headline": _headline_metrics(metrics),
        "metrics": metrics,
    }
    if eval_args.output_json:
        result["output_json"] = eval_args.output_json
        _write_json(eval_args.output_json, result)
    return result


def main() -> None:
    result = evaluate_checkpoint(tyro.cli(EvalCheckpointArgs))
    print("=== Evaluation Results ===", flush=True)
    for key, value in result["headline"].items():
        print(f"{key}: {value:.4f}", flush=True)
    print("--- all metrics ---", flush=True)
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.4f}", flush=True)
    if "output_json" in result:
        print(f"output_json: {result['output_json']}", flush=True)


if __name__ == "__main__":
    main()
