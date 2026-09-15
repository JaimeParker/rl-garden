"""Evaluate every numbered residual Peg checkpoint in a run directory.

Example:
    CUDA_VISIBLE_DEVICES=0 python examples/eval_residual_sac_rgbd_peg_checkpoints.py \
      runs/residual-hilserl-nohil \
      --num_eval_envs 16 \
      --num_eval_steps 150 \
      --save_curve
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, fields
from importlib import import_module
from pathlib import Path
from typing import Any, Optional


_MONITOR_RE = re.compile(
    r"\[sync\] learner link .*?\breceived=(?P<received>\d+)\b.*?"
    r"\bglobal_update=(?P<global_update>\d+)\b"
)


@dataclass
class EvalResidualCheckpointSuccessArgs:
    """Batch-evaluate residual Peg checkpoints and save success-vs-transition data."""

    run_path: str
    num_eval_envs: int = 16
    num_eval_steps: int = 100
    output_path: Optional[str] = None
    save_curve: bool = False
    curve_path: Optional[str] = None
    save_video: bool = False
    video_output_dir: Optional[str] = None
    seed: Optional[int] = None
    device: str = "auto"
    buffer_device: str = "cpu"
    strict: bool = True


@dataclass(frozen=True)
class NumberedCheckpoint:
    n: int
    path: Path


@dataclass(frozen=True)
class StepSample:
    global_update: int
    received_transitions: int
    source_path: Path


@dataclass(frozen=True)
class StepMapping:
    received_transitions: Optional[int]
    source: str
    update_gap: Optional[int]


@dataclass(frozen=True)
class EvalRow:
    checkpoint_name: str
    checkpoint_path: str
    checkpoint_n: int
    checkpoint_global_update: int
    checkpoint_internal_global_step: int
    received_transitions: Optional[int]
    success_rate: float
    success_at_end: Optional[float]
    success_once: Optional[float]
    return_: Optional[float]
    mapping_source: str
    mapping_update_gap: Optional[int]


@dataclass(frozen=True)
class CurvePoint:
    received_transitions: int
    success_rate: float
    checkpoints: int


def _eval_helpers() -> Any:
    try:
        return import_module("examples.eval_residual_sac_rgbd_peg")
    except ModuleNotFoundError:
        return import_module("eval_residual_sac_rgbd_peg")


def _load_checkpoint_file(path: str | Path, map_location: str) -> dict[str, Any]:
    from rl_garden.common.checkpoint import load_checkpoint_file

    return load_checkpoint_file(path, map_location=map_location)


def _resolve_paths(run_path: str | Path) -> tuple[Path, Path]:
    path = Path(run_path)
    checkpoint_dir = path if path.name == "checkpoints" else path / "checkpoints"
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else path
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    return run_dir, checkpoint_dir


def checkpoint_number(path: str | Path) -> int:
    stem = Path(path).stem
    if not stem.startswith("checkpoint_"):
        raise ValueError(f"Expected checkpoint_<N>.pt, got {Path(path).name!r}.")
    suffix = stem.removeprefix("checkpoint_")
    if not suffix.isdigit():
        raise ValueError(f"Checkpoint suffix is not numeric: {Path(path).name!r}.")
    return int(suffix)


def discover_numbered_checkpoints(checkpoint_dir: str | Path) -> list[NumberedCheckpoint]:
    checkpoints: list[NumberedCheckpoint] = []
    for path in Path(checkpoint_dir).glob("checkpoint_*.pt"):
        suffix = path.stem.removeprefix("checkpoint_")
        if suffix.isdigit():
            checkpoints.append(NumberedCheckpoint(int(suffix), path))
    checkpoints.sort(key=lambda item: item.n)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt found in {checkpoint_dir}")
    return checkpoints


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _set_if_present(args: Any, data: dict[str, Any], name: str) -> None:
    if name not in data:
        return
    value = data[name]
    if name == "fixed_peg_xy" and value is not None:
        value = tuple(value)
    setattr(args, name, value)


def eval_args_from_run_config(
    batch_args: EvalResidualCheckpointSuccessArgs,
    run_dir: Path,
    first_checkpoint: Path,
) -> Any:
    helpers = _eval_helpers()
    config = _load_run_config(run_dir)
    config_args = config.get("args", {}) if isinstance(config.get("args"), dict) else {}
    eval_args = helpers.EvalResidualRGBDPegArgs(
        checkpoint_path=str(first_checkpoint),
        robot_uids=None if config_args else "panda_wristcam_gripper_closed",
        fix_peg_pose=None if config_args else False,
        fix_box=None if config_args else True,
        fixed_peg_xy=None if config_args else (-0.05, -0.15),
        fixed_peg_z_rot_deg=None if config_args else 67.5,
        save_video=False,
    )

    eval_field_names = {field.name for field in fields(helpers.EvalResidualRGBDPegArgs)}
    blocked = {
        "checkpoint_path",
        "num_eval_envs",
        "num_eval_steps",
        "output_dir",
        "video_name",
        "save_video",
    }
    for name in sorted(eval_field_names - blocked):
        _set_if_present(eval_args, config_args, name)

    maniskill = config_args.get("maniskill")
    if isinstance(maniskill, dict):
        for name in ("sim_backend", "render_backend", "reward_mode", "env_kwargs_json"):
            _set_if_present(eval_args, maniskill, name)

    eval_args.checkpoint_path = str(first_checkpoint)
    eval_args.num_eval_envs = batch_args.num_eval_envs
    eval_args.num_eval_steps = batch_args.num_eval_steps
    eval_args.device = batch_args.device
    eval_args.buffer_device = batch_args.buffer_device
    eval_args.strict = batch_args.strict
    eval_args.save_video = False
    if batch_args.seed is not None:
        eval_args.seed = batch_args.seed
    return eval_args


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _contains_run_name(path: Path, run_name: str) -> bool:
    if not path.exists():
        return False
    return run_name in _read_text(path)


def find_step_log_paths(run_dir: Path) -> list[Path]:
    project_root = run_dir.parent.parent
    run_name = run_dir.name
    out: list[Path] = []

    wandb_root = project_root / "wandb"
    if wandb_root.is_dir():
        for wandb_run in sorted(wandb_root.glob("run-*")):
            output_log = wandb_run / "files" / "output.log"
            debug_log = wandb_run / "logs" / "debug.log"
            if output_log.exists() and (
                _contains_run_name(output_log, run_name)
                or _contains_run_name(debug_log, run_name)
            ):
                out.append(output_log)
        latest = wandb_root / "latest-run" / "files" / "output.log"
        if not out and latest.exists():
            out.append(latest)

    logs_root = project_root / "logs"
    if logs_root.is_dir():
        for log_path in sorted(logs_root.glob("*.log")):
            if _contains_run_name(log_path, run_name):
                out.append(log_path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def parse_step_samples_from_text(text: str, source_path: str | Path = "") -> list[StepSample]:
    source = Path(source_path)
    samples: list[StepSample] = []
    for match in _MONITOR_RE.finditer(text):
        samples.append(
            StepSample(
                global_update=int(match.group("global_update")),
                received_transitions=int(match.group("received")),
                source_path=source,
            )
        )
    return samples


def load_step_samples(paths: list[Path]) -> list[StepSample]:
    samples: list[StepSample] = []
    for path in paths:
        samples.extend(parse_step_samples_from_text(_read_text(path), path))
    samples.sort(key=lambda sample: (sample.global_update, sample.received_transitions))
    return samples


def map_update_to_received(samples: list[StepSample], checkpoint_n: int) -> StepMapping:
    if not samples:
        return StepMapping(None, "missing", None)

    exact = [sample for sample in samples if sample.global_update == checkpoint_n]
    if exact:
        sample = max(exact, key=lambda item: item.received_transitions)
        return StepMapping(sample.received_transitions, "exact_monitor_log", 0)

    nearest = min(
        samples,
        key=lambda sample: (
            abs(sample.global_update - checkpoint_n),
            sample.global_update,
            sample.received_transitions,
        ),
    )
    return StepMapping(
        nearest.received_transitions,
        "nearest_monitor_log",
        abs(nearest.global_update - checkpoint_n),
    )


def _checkpoint_metadata(path: Path) -> tuple[int, int]:
    checkpoint = _load_checkpoint_file(path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    return int(metadata.get("global_update", 0)), int(metadata.get("global_step", 0))


def _metric(metrics: dict[str, float], key: str) -> Optional[float]:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _success_rate(metrics: dict[str, float]) -> float:
    for key in ("success_at_end", "success_once", "success"):
        value = _metric(metrics, key)
        if value is not None:
            return value
    return float("nan")


def evaluate_checkpoint(
    agent: Any,
    eval_env: Any,
    eval_args: Any,
    checkpoint_path: Path,
) -> dict[str, float]:
    import torch

    helpers = _eval_helpers()
    writer = None
    agent.load(
        checkpoint_path,
        strict=eval_args.strict,
        load_replay_buffer=False,
        load_optimizers=False,
    )
    agent.policy.eval()
    agent.base_action_provider.bind_env(eval_env)
    agent.base_action_provider.reset()

    try:
        obs, _ = eval_env.reset(seed=eval_args.seed)
        if eval_args.save_video:
            if not 0 <= eval_args.video_env_index < eval_args.num_eval_envs:
                raise ValueError(
                    f"video_env_index must be in [0, {eval_args.num_eval_envs}), "
                    f"got {eval_args.video_env_index}."
                )
            frame = helpers.combined_camera_frame(
                obs,
                base_key=eval_args.base_camera_key,
                wrist_key=eval_args.wrist_camera_key,
                env_index=eval_args.video_env_index,
                scale=eval_args.video_scale,
            )
            writer = helpers.FFMpegVideoWriter(
                helpers._video_path(eval_args),
                fps=eval_args.video_fps,
                frame_shape=frame.shape,
                codec=eval_args.video_codec,
            )
            writer.write(frame)

        metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(eval_args.num_eval_steps):
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True, return_info=False)
                obs, _, _, _, infos = eval_env.step(action)
            helpers._append_final_metrics(metrics, infos)
            if writer is not None:
                writer.write(
                    helpers.combined_camera_frame(
                        obs,
                        base_key=eval_args.base_camera_key,
                        wrist_key=eval_args.wrist_camera_key,
                        env_index=eval_args.video_env_index,
                        scale=eval_args.video_scale,
                    )
                )
        return helpers._summarize_metrics(metrics)
    finally:
        if writer is not None:
            writer.close()


def _row_dict(row: EvalRow) -> dict[str, Any]:
    return {
        "checkpoint_name": row.checkpoint_name,
        "checkpoint_path": row.checkpoint_path,
        "checkpoint_n": row.checkpoint_n,
        "checkpoint_global_update": row.checkpoint_global_update,
        "checkpoint_internal_global_step": row.checkpoint_internal_global_step,
        "received_transitions": row.received_transitions,
        "success_rate": row.success_rate,
        "success_at_end": row.success_at_end,
        "success_once": row.success_once,
        "return": row.return_,
        "mapping_source": row.mapping_source,
        "mapping_update_gap": row.mapping_update_gap,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_rows_csv(rows: list[EvalRow], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_row_dict(rows[0]).keys()) if rows else [
        "checkpoint_name",
        "checkpoint_path",
        "checkpoint_n",
        "checkpoint_global_update",
        "checkpoint_internal_global_step",
        "received_transitions",
        "success_rate",
        "success_at_end",
        "success_once",
        "return",
        "mapping_source",
        "mapping_update_gap",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in _row_dict(row).items()})
    return out


def aggregate_curve_points(rows: list[EvalRow]) -> list[CurvePoint]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.received_transitions is None or math.isnan(row.success_rate):
            continue
        buckets[row.received_transitions].append(row.success_rate)
    return [
        CurvePoint(step, sum(values) / len(values), len(values))
        for step, values in sorted(buckets.items())
    ]


def _curve_csv_path(curve_path: Path) -> Path:
    return curve_path.with_suffix(".csv")


def default_video_output_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "eval_checkpoint_videos"


def save_curve(points: list[CurvePoint], path: str | Path) -> Path:
    if not points:
        raise ValueError("No rows with received_transitions are available for plotting.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--save_curve requires matplotlib. Install matplotlib or rerun without "
            "--save_curve."
        ) from exc

    xs = [point.received_transitions for point in points]
    ys = [point.success_rate for point in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("received transitions")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def write_curve_csv(points: list[CurvePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["received_transitions", "success_rate", "checkpoints"],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "received_transitions": point.received_transitions,
                    "success_rate": point.success_rate,
                    "checkpoints": point.checkpoints,
                }
            )
    return out


def run_batch_eval(args: EvalResidualCheckpointSuccessArgs) -> list[EvalRow]:
    from rl_garden.common import seed_everything
    from rl_garden.common.utils import get_device

    helpers = _eval_helpers()
    run_dir, checkpoint_dir = _resolve_paths(args.run_path)
    checkpoints = discover_numbered_checkpoints(checkpoint_dir)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else run_dir / "eval_checkpoint_success.csv"
    )
    curve_path = (
        Path(args.curve_path)
        if args.curve_path
        else run_dir / "eval_checkpoint_success.png"
    )
    video_output_dir = (
        Path(args.video_output_dir)
        if args.video_output_dir
        else default_video_output_dir(run_dir)
    )

    step_log_paths = find_step_log_paths(run_dir)
    step_samples = load_step_samples(step_log_paths)
    if not step_samples:
        print(
            "[eval_checkpoints] no learner monitor log samples found; "
            "received_transitions will be empty.",
            flush=True,
        )

    eval_args = eval_args_from_run_config(args, run_dir, checkpoints[0].path)
    first_checkpoint = _load_checkpoint_file(checkpoints[0].path, map_location="cpu")
    helpers._apply_checkpoint_config(eval_args, first_checkpoint)

    seed_everything(eval_args.seed)
    device = get_device(eval_args.device)
    eval_env = helpers._make_eval_env(eval_args)

    rows: list[EvalRow] = []
    try:
        agent = helpers._make_agent(eval_args, eval_env, device)
        for checkpoint in checkpoints:
            eval_args.checkpoint_path = str(checkpoint.path)
            eval_args.save_video = args.save_video
            if args.save_video:
                eval_args.output_dir = str(video_output_dir)
                eval_args.video_name = f"{checkpoint.path.stem}.mp4"
            metrics = evaluate_checkpoint(agent, eval_env, eval_args, checkpoint.path)
            checkpoint_global_update, checkpoint_global_step = _checkpoint_metadata(checkpoint.path)
            mapping = map_update_to_received(step_samples, checkpoint.n)
            row = EvalRow(
                checkpoint_name=checkpoint.path.name,
                checkpoint_path=str(checkpoint.path),
                checkpoint_n=checkpoint.n,
                checkpoint_global_update=checkpoint_global_update,
                checkpoint_internal_global_step=checkpoint_global_step,
                received_transitions=mapping.received_transitions,
                success_rate=_success_rate(metrics),
                success_at_end=_metric(metrics, "success_at_end"),
                success_once=_metric(metrics, "success_once"),
                return_=_metric(metrics, "return"),
                mapping_source=mapping.source,
                mapping_update_gap=mapping.update_gap,
            )
            rows.append(row)
            received = "na" if row.received_transitions is None else str(row.received_transitions)
            print(
                "[eval_checkpoints] "
                f"checkpoint={row.checkpoint_name} "
                f"checkpoint_n={row.checkpoint_n} "
                f"received_transitions={received} "
                f"success_rate={row.success_rate:.4f}",
                flush=True,
            )
    finally:
        eval_env.close()

    write_rows_csv(rows, output_path)
    print(f"[eval_checkpoints] wrote {output_path}", flush=True)

    if args.save_curve:
        points = aggregate_curve_points(rows)
        write_curve_csv(points, _curve_csv_path(curve_path))
        save_curve(points, curve_path)
        print(f"[eval_checkpoints] wrote {curve_path}", flush=True)
    return rows


def main() -> None:
    import tyro

    run_batch_eval(tyro.cli(EvalResidualCheckpointSuccessArgs))


if __name__ == "__main__":
    main()
