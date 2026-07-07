"""Tests for off2on/_runner.py's Minari wiring helpers."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.training.off2on import _runner
from rl_garden.training.off2on._runner import (
    _require_continuous_action_space,
    _resolve_env_id,
)
from rl_garden.training.off2on.wsrl import WSRLOff2OnArgs


def _args(**overrides):
    defaults = dict(
        dataset_source="maniskill_h5",
        env_id="PickCube-v1",
        offline_dataset_path=None,
        env_backend="maniskill",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_env_id_binds_to_minari_dataset_when_default(monkeypatch):
    args = _args(
        dataset_source="minari",
        env_id="PickCube-v1",
        offline_dataset_path="D4RL/antmaze/umaze-v1",
    )
    assert _resolve_env_id(args) == "D4RL/antmaze/umaze-v1"


def test_resolve_env_id_respects_explicit_override():
    args = _args(
        dataset_source="minari",
        env_id="D4RL/antmaze/large-play-v1",
        offline_dataset_path="D4RL/antmaze/umaze-v1",
    )
    assert _resolve_env_id(args) == "D4RL/antmaze/large-play-v1"


def test_resolve_env_id_unaffected_for_maniskill_h5():
    args = _args(dataset_source="maniskill_h5", env_id="PickCube-v1")
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
            single_action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            close=lambda: None,
        )
        eval_env = SimpleNamespace(close=lambda: None) if req.create_eval_env else None
        return env, eval_env

    monkeypatch.setattr(_runner, "make_training_envs", fake_make_training_envs)

    def build_agent(args, env, eval_env, logger, checkpoint_dir):
        del args, env, eval_env, logger, checkpoint_dir
        agent = MagicMock()
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
