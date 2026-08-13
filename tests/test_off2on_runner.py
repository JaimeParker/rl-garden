"""Tests for off2on/_runner.py's Minari wiring helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.training.off2on import _runner
from rl_garden.training.off2on._runner import (
    _apply_online_eval_freq,
    _offline_update_loop,
    _require_continuous_action_space,
    _resolve_env_id,
    _set_offline_probe,
)
from rl_garden.training.off2on.calql import CalQLOff2OnArgs
from rl_garden.training.off2on.iql import IQLOff2OnArgs
from rl_garden.training.off2on.wsrl import WSRLOff2OnArgs


def _args(**overrides):
    defaults = {
        "dataset_backend": "h5",
        "env_id": "PickCube-v1",
        "offline_dataset": None,
        "env_backend": "maniskill",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_env_id_binds_to_minari_dataset_when_default(monkeypatch):
    args = _args(
        dataset_backend="minari",
        env_id="PickCube-v1",
        offline_dataset="D4RL/antmaze/umaze-v1",
    )
    assert _resolve_env_id(args) == "D4RL/antmaze/umaze-v1"


def test_resolve_env_id_respects_explicit_override():
    args = _args(
        dataset_backend="minari",
        env_id="D4RL/antmaze/large-play-v1",
        offline_dataset="D4RL/antmaze/umaze-v1",
    )
    assert _resolve_env_id(args) == "D4RL/antmaze/large-play-v1"


def test_resolve_env_id_unaffected_for_h5():
    args = _args(dataset_backend="h5", env_id="PickCube-v1")
    assert _resolve_env_id(args) == "PickCube-v1"


def test_require_continuous_action_space_allows_box():
    env = SimpleNamespace(
        single_action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    )
    _require_continuous_action_space(env, _args())


def test_require_continuous_action_space_rejects_discrete():
    env = SimpleNamespace(single_action_space=spaces.Discrete(4))
    with pytest.raises(ValueError, match="Discrete"):
        _require_continuous_action_space(
            env, _args(env_backend="minari", env_id="atari/pong/expert-v0")
        )


def _run_off2on_and_capture_create_eval_env(monkeypatch, tmp_path, **arg_overrides):
    captured = {}

    def fake_make_training_envs(backend_name, req):
        del backend_name
        captured["create_eval_env"] = req.create_eval_env
        env = SimpleNamespace(
            single_action_space=spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ),
            close=lambda: None,
        )
        eval_env = SimpleNamespace(close=lambda: None) if req.create_eval_env else None
        return env, eval_env

    monkeypatch.setattr(_runner, "make_training_envs", fake_make_training_envs)

    def build_agent(args, env, eval_env, logger, checkpoint_dir):
        from rl_garden.training.inspection import construct_agent

        del args, env, eval_env, logger, checkpoint_dir
        agent = construct_agent(MagicMock)
        agent.checkpoint_dir = None
        agent.save_final_checkpoint = False
        return agent

    args = WSRLOff2OnArgs(
        log_type="none",
        log_dir=str(tmp_path),
        num_offline_steps=0,
        num_online_steps=0,
        save_final_checkpoint=False,
        **arg_overrides,
    )
    _runner.run_off2on(args, build_agent=build_agent, algorithm="wsrl")
    return captured["create_eval_env"]


def test_run_off2on_skips_eval_env_when_eval_freq_zero(monkeypatch, tmp_path):
    created = _run_off2on_and_capture_create_eval_env(
        monkeypatch, tmp_path, eval_freq=0
    )
    assert created is False


def test_run_off2on_builds_eval_env_when_eval_freq_positive(monkeypatch, tmp_path):
    created = _run_off2on_and_capture_create_eval_env(
        monkeypatch, tmp_path, eval_freq=1000, num_eval_envs=4
    )
    assert created is True


def test_run_off2on_builds_eval_env_when_only_offline_eval_freq_set(
    monkeypatch, tmp_path
):
    """Regression for the codex review's claim #5: eval_freq=0 with a
    phase-specific frequency set must still build an eval env -- previously
    should_create_eval_env() only checked eval_freq, silently disabling both
    new frequencies.
    """
    created = _run_off2on_and_capture_create_eval_env(
        monkeypatch, tmp_path, eval_freq=0, offline_eval_freq=50000, num_eval_envs=4
    )
    assert created is True


def test_run_off2on_builds_eval_env_when_only_online_eval_freq_set(
    monkeypatch, tmp_path
):
    created = _run_off2on_and_capture_create_eval_env(
        monkeypatch, tmp_path, eval_freq=0, online_eval_freq=2000, num_eval_envs=4
    )
    assert created is True


def test_run_off2on_closes_envs_when_builder_fails(monkeypatch, tmp_path):
    closed = {"train": False, "eval": False}

    def fake_make_training_envs(backend_name, req):
        del backend_name, req
        train_env = SimpleNamespace(
            single_action_space=spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            close=lambda: closed.__setitem__("train", True),
        )
        eval_env = SimpleNamespace(
            close=lambda: closed.__setitem__("eval", True),
        )
        return train_env, eval_env

    monkeypatch.setattr(_runner, "make_training_envs", fake_make_training_envs)
    args = WSRLOff2OnArgs(
        log_type="none",
        log_dir=str(tmp_path),
        num_offline_steps=0,
        num_online_steps=0,
        save_final_checkpoint=False,
        eval_freq=1,
        num_eval_envs=1,
    )

    def failing_builder(*unused):
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        _runner.run_off2on(
            args,
            build_agent=failing_builder,
            algorithm="wsrl",
        )

    assert closed == {"train": True, "eval": True}


def _run_off2on_and_capture_num_eval_steps(
    monkeypatch, tmp_path, args_cls, algorithm="calql", **arg_overrides
):
    captured = {}

    def fake_make_training_envs(backend_name, req):
        del backend_name
        captured["num_eval_steps"] = req.num_eval_steps
        env = SimpleNamespace(
            single_action_space=spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ),
            close=lambda: None,
        )
        eval_env = SimpleNamespace(close=lambda: None) if req.create_eval_env else None
        return env, eval_env

    monkeypatch.setattr(_runner, "make_training_envs", fake_make_training_envs)

    def build_agent(args, env, eval_env, logger, checkpoint_dir):
        from rl_garden.training.inspection import construct_agent

        del args, env, eval_env, logger, checkpoint_dir
        agent = construct_agent(MagicMock)
        agent.checkpoint_dir = None
        agent.save_final_checkpoint = False
        return agent

    args = args_cls(
        log_type="none",
        log_dir=str(tmp_path),
        num_offline_steps=0,
        num_online_steps=0,
        save_final_checkpoint=False,
        **arg_overrides,
    )
    _runner.run_off2on(args, build_agent=build_agent, algorithm=algorithm)
    return captured["num_eval_steps"]


def test_run_off2on_eval_step_cap_derived_from_episode_horizon(monkeypatch, tmp_path):
    num_eval_steps = _run_off2on_and_capture_num_eval_steps(
        monkeypatch,
        tmp_path,
        CalQLOff2OnArgs,
        eval_freq=1000,
        num_eval_envs=4,
        num_eval_episodes=10,
        eval_episode_horizon=1_000,
    )
    assert num_eval_steps == 10_000


def test_run_off2on_iql_eval_step_cap_derived_from_episode_horizon(
    monkeypatch, tmp_path
):
    num_eval_steps = _run_off2on_and_capture_num_eval_steps(
        monkeypatch,
        tmp_path,
        IQLOff2OnArgs,
        algorithm="iql",
        eval_freq=0,
        online_eval_freq=1000,
        num_eval_envs=1,
        num_eval_episodes=100,
        eval_episode_horizon=1_000,
    )
    assert num_eval_steps == 100_000


def test_run_off2on_eval_step_cap_ignores_horizon_without_episode_target(
    monkeypatch, tmp_path
):
    # WSRLOff2OnArgs has no num_eval_episodes field (only Cal-QL off2on does),
    # so a horizon alone can't size a budget -- must fall back to the default
    # and warn, not silently derive something.
    with pytest.warns(RuntimeWarning, match="was ignored"):
        num_eval_steps = _run_off2on_and_capture_num_eval_steps(
            monkeypatch,
            tmp_path,
            WSRLOff2OnArgs,
            eval_freq=1000,
            num_eval_envs=4,
            eval_episode_horizon=1_000,
        )
    assert num_eval_steps == 50


# ---------------------------------------------------------------------------
# Phase-specific eval frequencies (offline: gradient-update units, online:
# env-step units) -- see _offline_update_loop's `eval_freq` param and
# _apply_online_eval_freq.
# ---------------------------------------------------------------------------


def test_apply_online_eval_freq_overrides_agent_eval_freq_when_set():
    agent = SimpleNamespace(eval_freq=25)
    args = SimpleNamespace(online_eval_freq=2000)
    _apply_online_eval_freq(agent, args)
    assert agent.eval_freq == 2000


def test_apply_online_eval_freq_leaves_agent_eval_freq_when_unset():
    agent = SimpleNamespace(eval_freq=25)
    args = SimpleNamespace(online_eval_freq=None)
    _apply_online_eval_freq(agent, args)
    assert agent.eval_freq == 25


def test_apply_online_eval_freq_leaves_agent_eval_freq_when_field_absent():
    # Non-Cal-QL off2on algorithms (e.g. IQL/AWAC) don't expose this field.
    agent = SimpleNamespace(eval_freq=25)
    args = SimpleNamespace()
    _apply_online_eval_freq(agent, args)
    assert agent.eval_freq == 25


class _FakeOfflineEvalAgent:
    """Minimal stand-in for `_offline_update_loop`'s `agent` argument."""

    def __init__(self, eval_freq: int) -> None:
        self.eval_freq = eval_freq
        self.eval_env = object()  # non-None: enables the eval branch
        self.checkpoint_freq = 0
        self.eval_calls: list[int] = []

    def train(
        self, gradient_steps: int, compute_info: bool = False
    ) -> dict[str, float]:
        return {}

    def _evaluate(self) -> dict[str, float]:
        self.eval_calls.append(1)
        return {}

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step


def test_offline_update_loop_uses_explicit_eval_freq_over_agent_eval_freq():
    from rl_garden.common.logger import Logger

    # agent.eval_freq=1000 would never fire in 4 steps; the explicit
    # offline_eval_freq=2 (gradient-update units) should fire twice.
    agent = _FakeOfflineEvalAgent(eval_freq=1000)
    _offline_update_loop(
        agent,
        steps=4,
        logger=Logger(log_type="none"),
        log_freq=0,
        std_log=False,
        eval_freq=2,
    )
    assert len(agent.eval_calls) == 2


def test_offline_update_loop_falls_back_to_agent_eval_freq_when_none():
    from rl_garden.common.logger import Logger

    agent = _FakeOfflineEvalAgent(eval_freq=2)
    _offline_update_loop(
        agent,
        steps=4,
        logger=Logger(log_type="none"),
        log_freq=0,
        std_log=False,
        eval_freq=None,
    )
    assert len(agent.eval_calls) == 2


def test_set_offline_probe_uses_sampleable_size_not_len():
    """Regression: _set_offline_probe used to gate readiness on
    len(replay_buffer), but MCReplayBufferMixin.sample() can raise when
    sampleable_size == 0 even if len() > 0 (an all-open-trajectory buffer).
    Build a buffer where sampleable_size < len() (one trailing incomplete
    episode) and confirm probe_size respects sampleable_size, not len()."""
    from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    buffer = MCTensorReplayBuffer(
        obs_space, act_space, 1, 8, gamma=0.9, storage_device="cpu", sample_device="cpu"
    )
    # Two complete episodes, then one still-open trailing step.
    for is_last in (True, True, False):
        buffer.add(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 1),
            torch.ones(1),
            torch.zeros(1),
            episode_end=torch.tensor([is_last]),
        )
    assert buffer.sampleable_size == 2
    assert len(buffer) == 3

    agent = SimpleNamespace(
        batch_size=10, replay_buffer=buffer, set_offline_probe_batch=MagicMock()
    )
    logger = MagicMock()

    _set_offline_probe(agent, logger, std_log=False)

    agent.set_offline_probe_batch.assert_called_once()
    probed = agent.set_offline_probe_batch.call_args[0][0]
    assert probed.obs.shape[0] == buffer.sampleable_size
