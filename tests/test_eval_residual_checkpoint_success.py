import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from eval_residual_sac_rgbd_peg_checkpoints import (
    CurvePoint,
    EvalResidualCheckpointSuccessArgs,
    EvalRow,
    aggregate_curve_points,
    default_video_output_dir,
    discover_numbered_checkpoints,
    eval_args_from_run_config,
    find_step_log_paths,
    map_update_to_received,
    parse_step_samples_from_text,
    write_curve_csv,
    write_rows_csv,
)
from plot_residual_checkpoint_success_curves import (
    NamedCurve,
    default_label,
    load_curve_points,
    load_named_curves,
)


def test_discover_numbered_checkpoints_sorts_and_ignores_final(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_400.pt").touch()
    (tmp_path / "checkpoint_200.pt").touch()
    (tmp_path / "checkpoint_bad.pt").touch()
    (tmp_path / "final.pt").touch()

    checkpoints = discover_numbered_checkpoints(tmp_path)

    assert [(item.n, item.path.name) for item in checkpoints] == [
        (200, "checkpoint_200.pt"),
        (400, "checkpoint_400.pt"),
    ]


def test_parse_step_samples_and_map_exact_nearest_missing() -> None:
    text = "\n".join(
        [
            "[sync] learner link received=1200 replay_len=1200 global_update=100",
            "[sync] learner link received=1800 replay_len=1800 global_update=200",
            "[sync] learner link received=2800 replay_len=2800 global_update=300",
        ]
    )
    samples = parse_step_samples_from_text(text, "output.log")

    exact = map_update_to_received(samples, 200)
    nearest = map_update_to_received(samples, 260)
    missing = map_update_to_received([], 200)

    assert exact.received_transitions == 1800
    assert exact.source == "exact_monitor_log"
    assert exact.update_gap == 0
    assert nearest.received_transitions == 2800
    assert nearest.source == "nearest_monitor_log"
    assert nearest.update_gap == 40
    assert missing.received_transitions is None
    assert missing.source == "missing"
    assert missing.update_gap is None


def test_find_step_log_paths_matches_run_name_in_wandb_debug(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "my-run"
    run_dir.mkdir(parents=True)
    wandb_run = tmp_path / "wandb" / "run-20260806_065501-abc123"
    (wandb_run / "files").mkdir(parents=True)
    (wandb_run / "logs").mkdir(parents=True)
    output_log = wandb_run / "files" / "output.log"
    debug_log = wandb_run / "logs" / "debug.log"
    output_log.write_text("[sync] learner link received=10 global_update=1\n")
    debug_log.write_text("config: {'run_name': 'my-run'}\n")

    assert find_step_log_paths(run_dir) == [output_log]


def test_eval_args_from_run_config_uses_config_and_cli_overrides(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "runs" / "my-run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {
                    "env_backend": "maniskill",
                    "env_id": "PegInsertionSidePegOnly-v1",
                    "obs_mode": "rgb",
                    "include_state": True,
                    "encoder": "vit",
                    "image_fusion_mode": "per_key",
                    "vit_embed_dim": 192,
                    "per_camera_rgbd": False,
                    "control_mode": "pd_ee_twist",
                    "seed": 7,
                    "maniskill": {
                        "sim_backend": "gpu",
                        "render_backend": "gpu",
                        "reward_mode": None,
                        "env_kwargs_json": "{\"max_episode_steps\": 150}",
                    },
                }
            }
        )
    )
    batch_args = EvalResidualCheckpointSuccessArgs(
        run_path=str(run_dir),
        num_eval_envs=3,
        num_eval_steps=9,
        seed=None,
    )

    eval_args = eval_args_from_run_config(
        batch_args, run_dir, run_dir / "checkpoints" / "checkpoint_1.pt"
    )

    assert eval_args.encoder == "vit"
    assert eval_args.vit_embed_dim == 192
    assert eval_args.control_mode == "pd_ee_twist"
    assert eval_args.env_kwargs_json == "{\"max_episode_steps\": 150}"
    assert eval_args.num_eval_envs == 3
    assert eval_args.num_eval_steps == 9
    assert eval_args.seed == 7
    assert eval_args.robot_uids is None
    assert eval_args.fix_box is None


def test_aggregate_curve_points_averages_duplicate_received_steps() -> None:
    rows = [
        _row("checkpoint_1.pt", 1, 1000, 0.0),
        _row("checkpoint_2.pt", 2, 1000, 1.0),
        _row("checkpoint_3.pt", 3, 2000, 0.5),
        _row("checkpoint_4.pt", 4, None, 1.0),
    ]

    assert aggregate_curve_points(rows) == [
        CurvePoint(received_transitions=1000, success_rate=0.5, checkpoints=2),
        CurvePoint(received_transitions=2000, success_rate=0.5, checkpoints=1),
    ]


def test_write_rows_and_curve_csv_fields(tmp_path: Path) -> None:
    row_path = tmp_path / "rows.csv"
    curve_path = tmp_path / "curve.csv"
    rows = [_row("checkpoint_2.pt", 2, 1000, 0.25)]
    points = [CurvePoint(received_transitions=1000, success_rate=0.25, checkpoints=1)]

    write_rows_csv(rows, row_path)
    write_curve_csv(points, curve_path)

    with row_path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["checkpoint_name"] == "checkpoint_2.pt"
    assert row["checkpoint_n"] == "2"
    assert row["checkpoint_internal_global_step"] == "0"
    assert row["received_transitions"] == "1000"
    assert row["success_rate"] == "0.25"
    assert row["mapping_source"] == "exact_monitor_log"

    with curve_path.open(newline="", encoding="utf-8") as f:
        curve = next(csv.DictReader(f))
    assert curve == {
        "received_transitions": "1000",
        "success_rate": "0.25",
        "checkpoints": "1",
    }


def test_video_cli_defaults_are_opt_in(tmp_path: Path) -> None:
    args = EvalResidualCheckpointSuccessArgs(run_path=str(tmp_path))

    assert args.save_video is False
    assert args.video_output_dir is None
    assert default_video_output_dir(tmp_path) == tmp_path / "eval_checkpoint_videos"


def test_plot_curves_loads_multiple_csvs_and_averages_duplicate_steps(
    tmp_path: Path,
) -> None:
    a = tmp_path / "run_a" / "eval_checkpoint_success.csv"
    b = tmp_path / "custom.csv"
    a.parent.mkdir()
    _write_plot_csv(a, [(1000, 0.0), (1000, 1.0), (2000, 0.5)])
    _write_plot_csv(b, [(1500, 0.25)])

    assert default_label(a) == "run_a"
    assert default_label(b) == "custom"
    assert load_curve_points(a) == [(1000, 0.5), (2000, 0.5)]
    assert load_named_curves([str(a), str(b)], ("A", "B")) == [
        NamedCurve(label="A", points=[(1000, 0.5), (2000, 0.5)]),
        NamedCurve(label="B", points=[(1500, 0.25)]),
    ]


def test_plot_curves_rejects_label_count_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval_checkpoint_success.csv"
    _write_plot_csv(csv_path, [(1000, 0.0)])

    with pytest.raises(ValueError, match="labels"):
        load_named_curves([str(csv_path)], ("a", "b"))


def _write_plot_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["received_transitions", "success_rate"],
        )
        writer.writeheader()
        for step, rate in rows:
            writer.writerow({"received_transitions": step, "success_rate": rate})


def _row(
    name: str,
    checkpoint_n: int,
    received_transitions: int | None,
    success_rate: float,
) -> EvalRow:
    return EvalRow(
        checkpoint_name=name,
        checkpoint_path=name,
        checkpoint_n=checkpoint_n,
        checkpoint_global_update=checkpoint_n,
        checkpoint_internal_global_step=0,
        received_transitions=received_transitions,
        success_rate=success_rate,
        success_at_end=success_rate,
        success_once=None,
        return_=None,
        mapping_source="exact_monitor_log",
        mapping_update_gap=0,
    )
