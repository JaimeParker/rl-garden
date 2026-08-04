from __future__ import annotations

import numpy as np

import rl_garden.envs.robotwin.adapter as robotwin_adapter
from rl_garden.envs.robotwin import RoboTwinEnvConfig
from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter
from rl_garden.envs.robotwin.executor import ThreadedRoboTwinExecutor


class _ResetTask:
    def __init__(self):
        self.close_calls = []

    def setup_demo(self, **kwargs):
        self.step_lim = kwargs["step_lim"]

    def get_obs(self):
        return {
            "observation": {
                "head_camera": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }

    def get_instruction(self):
        return "reset task"

    def close_env(self, clear_cache=True):
        self.close_calls.append(clear_cache)


class _InlineFuture:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        del timeout
        return self.value


class _InlinePool:
    def submit(self, fn, *args):
        return _InlineFuture(fn(*args))


class _ExecutorEnv:
    def __init__(self, env_id):
        self.env_id = env_id
        self.get_obs_calls = 0

    def reset(self, seed):
        return {"source": "reset", "env_id": self.env_id, "seed": seed}

    def get_obs(self):
        self.get_obs_calls += 1
        return {"source": "get_obs", "env_id": self.env_id}


def _make_adapter(monkeypatch, clear_cache_freq):
    tasks = []

    def fake_make_task(*args, **kwargs):
        del args, kwargs
        task = _ResetTask()
        tasks.append(task)
        return task

    monkeypatch.setattr(robotwin_adapter, "make_task", fake_make_task)
    cfg = RoboTwinEnvConfig(
        device="cpu",
        reward_mode="sparse",
        clear_cache_freq=clear_cache_freq,
    )
    adapter = RoboTwinTaskAdapter(
        0,
        cfg,
        {"left_robot_file": "/tmp/left", "right_robot_file": "/tmp/right"},
    )
    return adapter, tasks


def _make_executor(num_envs):
    executor = object.__new__(ThreadedRoboTwinExecutor)
    executor.num_envs = num_envs
    executor.pool = _InlinePool()
    executor.envs = [_ExecutorEnv(env_id) for env_id in range(num_envs)]
    return executor


def test_cache_clear_frequency_counts_completed_resets(monkeypatch):
    adapter, tasks = _make_adapter(monkeypatch, clear_cache_freq=2)

    adapter.reset()
    adapter.elapsed_steps = 400
    adapter.reset()
    adapter.elapsed_steps = 1
    adapter.reset()

    assert tasks[0].close_calls == [False]
    assert tasks[1].close_calls == [True]


def test_nonpositive_cache_frequency_disables_periodic_clear(monkeypatch):
    adapter, tasks = _make_adapter(monkeypatch, clear_cache_freq=0)

    adapter.reset()
    adapter.elapsed_steps = 400
    adapter.reset()

    assert tasks[0].close_calls == [False]


def test_full_reset_reuses_adapter_observations():
    executor = _make_executor(2)

    observations = executor.reset(env_seeds=[10, 20])

    assert observations == [
        {"source": "reset", "env_id": 0, "seed": 10},
        {"source": "reset", "env_id": 1, "seed": 20},
    ]
    assert [env.get_obs_calls for env in executor.envs] == [0, 0]


def test_partial_reset_only_reads_untouched_envs():
    executor = _make_executor(3)

    observations = executor.reset(env_indices=[1], env_seeds=[42])

    assert observations == [
        {"source": "get_obs", "env_id": 0},
        {"source": "reset", "env_id": 1, "seed": 42},
        {"source": "get_obs", "env_id": 2},
    ]
    assert [env.get_obs_calls for env in executor.envs] == [1, 0, 1]
