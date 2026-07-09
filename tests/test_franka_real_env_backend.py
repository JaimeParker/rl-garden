"""Tests for the franka_real env backend's EnvRequest -> config translation."""
from __future__ import annotations

from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.franka_real import FrankaRealBackend
from rl_garden.training.real_world._args import FrankaRealConfig


def _req(**overrides) -> EnvRequest:
    defaults = dict(
        env_id="unused",
        num_envs=8,
        obs_mode="rgb",
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        camera_width=128,
        camera_height=128,
        num_eval_envs=4,
        reward_scale=2.0,
        reward_bias=0.5,
        backend_config=FrankaRealConfig(
            bridge_url="http://robot-pc:5000",
            action_scale_pos=0.05,
            gripper_threshold=0.3,
            max_episode_steps=50,
        ),
    )
    defaults.update(overrides)
    return EnvRequest(**defaults)


def test_make_cfg_ignores_num_envs_and_forwards_backend_fields():
    cfg = FrankaRealBackend._make_cfg(_req())
    assert cfg.bridge_url == "http://robot-pc:5000"
    assert cfg.action_scale == (0.05, 0.1)
    assert cfg.gripper_threshold == 0.3
    assert cfg.max_episode_steps == 50
    assert cfg.reward_scale == 2.0
    assert cfg.reward_bias == 0.5


def test_make_cfg_defaults_when_backend_config_is_none():
    cfg = FrankaRealBackend._make_cfg(_req(backend_config=None))
    assert cfg.bridge_url == "http://localhost:5000"
    assert cfg.action_scale == (0.02, 0.1)


def test_make_cfg_env_kwargs_json_overrides_safety_box():
    backend_config = FrankaRealConfig(env_kwargs_json='{"safety_box_low": [0.1, 0.1, 0.1]}')
    cfg = FrankaRealBackend._make_cfg(_req(backend_config=backend_config))
    assert cfg.safety_box_low == [0.1, 0.1, 0.1]


def test_make_train_env_and_make_eval_env_both_build_the_same_style_env(monkeypatch):
    captured = []

    def _fake_make_franka_real_env(cfg):
        captured.append(cfg)
        return "sentinel-env"

    monkeypatch.setattr(
        "rl_garden.envs.franka_real.make_franka_real_env", _fake_make_franka_real_env
    )
    req = _req()
    assert FrankaRealBackend.make_train_env(req) == "sentinel-env"
    assert FrankaRealBackend.make_eval_env(req) == "sentinel-env"
    assert len(captured) == 2
