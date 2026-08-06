from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import rl_garden.envs.robotwin.adapter as robotwin_adapter
from rl_garden.envs.robotwin import RoboTwinEnvConfig
from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter


class UnStableError(Exception):
    pass


def _raw_obs() -> dict[str, Any]:
    return {
        "observation": {
            "head_camera": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }


def _adapter(env_seed: int = 10) -> RoboTwinTaskAdapter:
    cfg = RoboTwinEnvConfig(device="cpu", reward_mode="sparse")
    return RoboTwinTaskAdapter(
        0,
        cfg,
        {"left_robot_file": "/tmp/left", "right_robot_file": "/tmp/right"},
        env_seed=env_seed,
    )


class _Viewer:
    def __init__(self, events: list[Any], fail: bool) -> None:
        self.events = events
        self.fail = fail

    def close(self) -> None:
        self.events.append("viewer")
        if self.fail:
            raise RuntimeError("viewer close failed")


class _Task:
    def __init__(
        self,
        *,
        unstable: bool = False,
        setup_error: Exception | None = None,
        obs_error: Exception | None = None,
        cleanup_fails: bool = False,
    ) -> None:
        self.unstable = unstable
        self.setup_error = setup_error
        self.obs_error = obs_error
        self.cleanup_fails = cleanup_fails
        self.setup_seeds: list[int] = []
        self.close_calls: list[bool] = []
        self.events: list[Any] = []
        self.eval_video_ffmpeg = object()
        self.viewer = _Viewer(self.events, cleanup_fails)
        self.cameras = object()
        self.robot = object()
        self.scene = object()
        self.renderer = object()
        self.engine = object()
        self.reward = object()

    def setup_demo(self, *, now_ep_num: int, seed: int, **kwargs) -> None:
        del now_ep_num, kwargs
        self.setup_seeds.append(seed)
        if self.setup_error is not None:
            raise self.setup_error
        if self.unstable:
            raise UnStableError(f"unstable seed {seed}")

    def get_obs(self) -> dict[str, Any]:
        if self.obs_error is not None:
            raise self.obs_error
        return _raw_obs()

    def get_instruction(self) -> str:
        return "cleanup test"

    def _del_eval_video_ffmpeg(self) -> None:
        self.events.append("video")
        if self.cleanup_fails:
            raise RuntimeError("video close failed")
        del self.eval_video_ffmpeg

    def close_env(self, clear_cache: bool = True) -> None:
        self.close_calls.append(clear_cache)
        self.events.append(
            (
                "close_env",
                self.viewer,
                self.cameras,
                self.robot,
                self.scene,
                self.renderer,
                self.engine,
                self.reward,
                "is_in_hand" in self.__dict__,
            )
        )
        if self.cleanup_fails:
            raise RuntimeError("native close failed")


def test_close_releases_resources_and_only_adapter_helper(monkeypatch):
    adapter = _adapter()
    task = _Task()
    adapter.task = task
    adapter._joint_target_fk = object()  # type: ignore[assignment]
    adapter._install_helpers()
    gc_calls = []
    monkeypatch.setattr(robotwin_adapter.gc, "collect", lambda: gc_calls.append(True))

    adapter.close(clear_cache=True)

    assert task.events[:2] == ["video", "viewer"]
    assert task.events[2] == (
        "close_env",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    assert getattr(task, "eval_video_ffmpeg", None) is None
    assert adapter.task is None
    assert adapter._joint_target_fk is None
    assert gc_calls == [True]

    class NativeHelperTask(_Task):
        def is_in_hand(self, actor):
            return actor

    native_adapter = _adapter()
    native_task = NativeHelperTask()
    native_adapter.task = native_task
    native_adapter.close(clear_cache=False)
    assert native_task.is_in_hand("actor") == "actor"


def test_setup_error_cleanup_is_best_effort_and_preserves_original(
    monkeypatch, caplog
):
    original_error = ValueError("setup failed")
    task = _Task(setup_error=original_error, cleanup_fails=True)
    monkeypatch.setattr(robotwin_adapter, "make_task", lambda *args, **kwargs: task)
    gc_calls = []
    monkeypatch.setattr(robotwin_adapter.gc, "collect", lambda: gc_calls.append(True))
    adapter = _adapter()

    with pytest.raises(ValueError) as caught:
        adapter.reset()
    adapter.close(clear_cache=True)

    assert caught.value is original_error
    assert task.close_calls == [True]
    assert task.viewer is None
    assert task.cameras is None
    assert task.robot is None
    assert task.scene is None
    assert task.renderer is None
    assert task.engine is None
    assert task.reward is None
    assert task.eval_video_ffmpeg is None
    assert adapter.task is None
    assert gc_calls == [True]
    assert "RoboTwin cleanup encountered errors" in caplog.text


def test_unstable_reset_creates_a_fresh_task_for_each_seed(monkeypatch):
    tasks = [_Task(unstable=True), _Task(unstable=True), _Task()]
    created: list[_Task] = []

    def make_task(*args, **kwargs):
        del args, kwargs
        task = tasks[len(created)]
        created.append(task)
        return task

    monkeypatch.setattr(robotwin_adapter, "make_task", make_task)
    gc_calls = []
    monkeypatch.setattr(robotwin_adapter.gc, "collect", lambda: gc_calls.append(True))
    adapter = _adapter(env_seed=10)

    obs = adapter.reset()

    assert created == tasks
    assert [task.setup_seeds for task in tasks] == [[10], [11], [12]]
    assert [task.close_calls for task in tasks] == [[True], [True], []]
    assert adapter.task is tasks[-1]
    assert adapter.env_seed == 12
    assert adapter._completed_resets == 1
    assert obs["_env_seed"] == 12
    assert gc_calls == [True, True]


def test_post_setup_error_is_cleaned_without_counting_completed_reset(monkeypatch):
    original_error = LookupError("observation failed")
    task = _Task(obs_error=original_error)
    monkeypatch.setattr(robotwin_adapter, "make_task", lambda *args, **kwargs: task)
    adapter = _adapter()

    with pytest.raises(LookupError) as caught:
        adapter.reset()

    assert caught.value is original_error
    assert task.close_calls == [True]
    assert adapter.task is None
    assert adapter._completed_resets == 0
