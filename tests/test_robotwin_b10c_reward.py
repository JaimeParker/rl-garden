from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_garden.common.env_args import RoboTwinConfig
from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.robotwin import RoboTwinBackend
from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter
from rl_garden.envs.robotwin.config import RoboTwinEnvConfig


class _FakeDenseReward:
    def __init__(self, *values: float, fail: bool = False) -> None:
        self._values = iter(values)
        self.value = 0.0
        self.fail = fail
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1
        self.value = float(next(self._values))

    def is_fail(self) -> bool:
        return self.fail

    def compute_reward(self) -> float:
        return self.value


def _adapter(
    cfg: RoboTwinEnvConfig,
    *,
    potential: float = 0.0,
    last_dense_reward: float = 0.0,
    fail: bool = False,
) -> RoboTwinTaskAdapter:
    adapter = RoboTwinTaskAdapter(0, cfg, {}, env_seed=0)
    adapter.task = SimpleNamespace(reward=_FakeDenseReward(potential, fail=fail))
    adapter.last_dense_reward = last_dense_reward
    return adapter


def _robotwin_request(rt: RoboTwinConfig) -> EnvRequest:
    return EnvRequest(
        env_id="open_laptop",
        num_envs=2,
        num_eval_envs=3,
        obs_mode="rgb",
        control_mode="delta_joint_pos",
        render_mode="rgb_array",
        seed=1,
        camera_width=64,
        camera_height=64,
        capture_video=False,
        reward_scale=1.0,
        reward_bias=0.0,
        backend_config=rt,
    )


def test_robotwin_b10c_reward_args_default_to_legacy_absolute_mode() -> None:
    rt = RoboTwinConfig()
    cfg = RoboTwinBackend._make_cfg(_robotwin_request(rt), is_eval=False)

    assert rt.reward_mode == "dense"
    assert rt.reward_shaping_mode == "absolute"
    assert rt.use_relative_reward is False
    assert rt.dense_success_reward == 1.0
    assert rt.potential_discount == 0.99
    assert rt.potential_weight == 5.0
    assert rt.dense_weight == 0.03
    assert rt.relative_weight == 3.0
    assert rt.step_penalty == 0.003
    assert rt.stall_threshold == 1e-4
    assert rt.stall_penalty == 0.035
    assert rt.backtrack_penalty == 0.06

    assert cfg.reward_mode == "dense"
    assert cfg.reward_shaping_mode == "absolute"


def test_robotwin_backend_forwards_all_b10c_reward_args() -> None:
    rt = RoboTwinConfig(
        reward_mode="dense",
        reward_shaping_mode="hybrid",
        use_relative_reward=False,
        dense_success_reward=10.0,
        potential_discount=0.99,
        potential_weight=5.0,
        dense_weight=0.03,
        relative_weight=3.0,
        step_penalty=0.003,
        stall_threshold=0.0001,
        stall_penalty=0.035,
        backtrack_penalty=0.06,
    )

    cfg = RoboTwinBackend._make_cfg(_robotwin_request(rt), is_eval=True)

    assert cfg.reward_mode == "dense"
    assert cfg.reward_shaping_mode == "hybrid"
    assert cfg.use_relative_reward is False
    assert cfg.dense_success_reward == 10.0
    assert cfg.potential_discount == 0.99
    assert cfg.potential_weight == 5.0
    assert cfg.dense_weight == 0.03
    assert cfg.relative_weight == 3.0
    assert cfg.step_penalty == 0.003
    assert cfg.stall_threshold == 0.0001
    assert cfg.stall_penalty == 0.035
    assert cfg.backtrack_penalty == 0.06


def test_absolute_default_reward_matches_existing_dense_behavior() -> None:
    adapter = _adapter(RoboTwinEnvConfig(), potential=0.42)

    assert adapter._compute_reward(success=False) == pytest.approx(0.42)
    assert adapter.last_reward_components["dense"] == pytest.approx(0.42)

    adapter = _adapter(RoboTwinEnvConfig(), potential=0.42)
    assert adapter._compute_reward(success=True) == pytest.approx(1.0)
    assert adapter.task.reward.update_calls == 0


def test_legacy_use_relative_reward_keeps_existing_scaled_delta_behavior() -> None:
    cfg = RoboTwinEnvConfig(
        use_relative_reward=True,
        reward_scale=2.0,
        reward_bias=-1.0,
    )
    adapter = _adapter(cfg, potential=0.4, last_dense_reward=0.0)

    assert adapter._compute_reward(success=False) == pytest.approx(-0.2)
    assert adapter.last_dense_reward == pytest.approx(-0.2)


def test_relative_and_potential_modes_use_dense_potential_delta() -> None:
    relative = _adapter(
        RoboTwinEnvConfig(
            reward_shaping_mode="relative",
            dense_success_reward=10.0,
        ),
        potential=2.2,
        last_dense_reward=2.0,
    )
    assert relative._compute_reward(success=True) == pytest.approx(10.2)

    potential = _adapter(
        RoboTwinEnvConfig(
            reward_shaping_mode="potential",
            potential_discount=0.99,
            potential_weight=5.0,
        ),
        potential=2.2,
        last_dense_reward=2.0,
    )
    assert potential._compute_reward(success=False) == pytest.approx(0.89)


def test_hybrid_reward_combines_dense_relative_step_stall_backtrack() -> None:
    cfg = RoboTwinEnvConfig(
        reward_shaping_mode="hybrid",
        dense_success_reward=10.0,
        potential_discount=0.99,
        potential_weight=5.0,
        dense_weight=0.03,
        relative_weight=3.0,
        step_penalty=0.003,
        stall_threshold=0.0001,
        stall_penalty=0.035,
        backtrack_penalty=0.06,
    )

    progress = _adapter(cfg, potential=2.1, last_dense_reward=2.0)
    assert progress._compute_reward(success=False) == pytest.approx(0.36)
    assert progress.last_reward_components["step"] == pytest.approx(-0.003)
    assert progress.last_reward_components["stall"] == 0.0
    assert progress.last_reward_components["backtrack"] == 0.0

    stall = _adapter(cfg, potential=2.00005, last_dense_reward=2.0)
    assert stall._compute_reward(success=False) == pytest.approx(0.0221515)
    assert stall.last_reward_components["stall"] == pytest.approx(-0.035)

    backtrack = _adapter(cfg, potential=1.9, last_dense_reward=2.0)
    assert backtrack._compute_reward(success=False) == pytest.approx(-0.306)
    assert backtrack.last_reward_components["backtrack"] == pytest.approx(-0.06)

    success = _adapter(cfg, potential=2.2, last_dense_reward=2.0)
    assert success._compute_reward(success=True) == pytest.approx(10.666)
    assert success.last_reward_components["step"] == 0.0


def test_reward_config_validation_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="Unsupported reward_mode"):
        RoboTwinEnvConfig(reward_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported reward_shaping_mode"):
        RoboTwinEnvConfig(reward_shaping_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="potential_discount"):
        RoboTwinEnvConfig(potential_discount=1.01)
    with pytest.raises(ValueError, match="stall_penalty"):
        RoboTwinEnvConfig(stall_penalty=-1.0)
    with pytest.raises(ValueError, match="use_relative_reward"):
        RoboTwinEnvConfig(
            use_relative_reward=True,
            reward_shaping_mode="hybrid",
        )
