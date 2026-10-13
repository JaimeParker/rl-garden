"""Explicit official-FQL OGBench training presets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FQLPreset:
    """Complete official-main recipe for one trained task and phase."""

    name: str
    dataset_id: str
    offline_steps: int
    online_steps: int
    alpha: float
    discount: float
    q_agg: str
    reward_task_id: int


PRESETS: dict[str, FQLPreset] = {
    "cube-single-offline": FQLPreset("cube-single-offline", "cube-single-play-singletask-v0", 1_000_000, 0, 300.0, 0.99, "mean", 2),
    "cube-double-off2on": FQLPreset("cube-double-off2on", "cube-double-play-singletask-v0", 1_000_000, 1_000_000, 300.0, 0.99, "mean", 2),
    "scene-off2on": FQLPreset("scene-off2on", "scene-play-singletask-v0", 1_000_000, 1_000_000, 300.0, 0.99, "mean", 2),
    "antmaze-large-task1-offline": FQLPreset("antmaze-large-task1-offline", "antmaze-large-navigate-singletask-task1-v0", 1_000_000, 0, 10.0, 0.99, "min", 1),
    "humanoidmaze-medium-task2-offline": FQLPreset("humanoidmaze-medium-task2-offline", "humanoidmaze-medium-navigate-singletask-task2-v0", 1_000_000, 0, 30.0, 0.995, "mean", 2),
    "antsoccer-arena-task4-off2on": FQLPreset("antsoccer-arena-task4-off2on", "antsoccer-arena-navigate-singletask-task4-v0", 1_000_000, 1_000_000, 30.0, 0.995, "mean", 4),
    "puzzle-3x3-task4-offline": FQLPreset("puzzle-3x3-task4-offline", "puzzle-3x3-play-singletask-task4-v0", 1_000_000, 0, 1000.0, 0.99, "mean", 4),
    "puzzle-4x4-task4-off2on": FQLPreset("puzzle-4x4-task4-off2on", "puzzle-4x4-play-singletask-task4-v0", 1_000_000, 1_000_000, 1000.0, 0.99, "mean", 4),
}
