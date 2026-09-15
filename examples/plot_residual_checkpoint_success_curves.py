"""Plot multiple residual checkpoint success CSVs on one figure.

Example:
    python examples/plot_residual_checkpoint_success_curves.py \
      --csv-paths runs/run_a/eval_checkpoint_success.csv \
                  runs/run_b/eval_checkpoint_success.csv \
      --labels run_a run_b \
      --output-path runs/residual_success_compare.png
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PlotResidualCheckpointSuccessCurvesArgs:
    """Plot one success-rate curve per CSV on a shared axis."""

    csv_paths: list[str]
    output_path: str = "residual_checkpoint_success_curves.png"
    labels: tuple[str, ...] = ()
    title: Optional[str] = None


@dataclass(frozen=True)
class NamedCurve:
    label: str
    points: list[tuple[int, float]]


def _parse_optional_int(value: str | None) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> Optional[float]:
    if value is None or value == "":
        return None
    parsed = float(value)
    return None if math.isnan(parsed) else parsed


def default_label(path: str | Path) -> str:
    csv_path = Path(path)
    if csv_path.name == "eval_checkpoint_success.csv":
        return csv_path.parent.name
    return csv_path.stem


def load_curve_points(path: str | Path) -> list[tuple[int, float]]:
    """Load and aggregate one CSV's received-transition success curve."""
    buckets: dict[int, list[float]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "received_transitions" not in (reader.fieldnames or ()):
            raise ValueError(f"CSV is missing received_transitions column: {path}")
        if "success_rate" not in (reader.fieldnames or ()):
            raise ValueError(f"CSV is missing success_rate column: {path}")
        for row in reader:
            step = _parse_optional_int(row.get("received_transitions"))
            rate = _parse_optional_float(row.get("success_rate"))
            if step is None or rate is None:
                continue
            buckets[step].append(rate)
    points = [
        (step, sum(values) / len(values))
        for step, values in sorted(buckets.items())
        if values
    ]
    if not points:
        raise ValueError(f"CSV has no plottable points: {path}")
    return points


def load_named_curves(
    csv_paths: list[str],
    labels: tuple[str, ...] = (),
) -> list[NamedCurve]:
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")
    if labels and len(labels) != len(csv_paths):
        raise ValueError(
            f"labels must be empty or match csv_paths length: "
            f"labels={len(labels)}, csv_paths={len(csv_paths)}"
        )
    curves: list[NamedCurve] = []
    for idx, path in enumerate(csv_paths):
        label = labels[idx] if labels else default_label(path)
        curves.append(NamedCurve(label=label, points=load_curve_points(path)))
    return curves


def save_curves_plot(
    curves: list[NamedCurve],
    output_path: str | Path,
    *,
    title: Optional[str] = None,
) -> Path:
    if not curves:
        raise ValueError("At least one curve is required.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install matplotlib before running "
            "this script."
        ) from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for curve in curves:
        xs = [step for step, _ in curve.points]
        ys = [rate for _, rate in curve.points]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=curve.label)
    ax.set_xlabel("received transitions")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    import tyro

    args = tyro.cli(PlotResidualCheckpointSuccessCurvesArgs)
    curves = load_named_curves(args.csv_paths, args.labels)
    out = save_curves_plot(curves, args.output_path, title=args.title)
    print(f"[plot_curves] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
