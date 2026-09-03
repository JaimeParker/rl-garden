from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_garden.envs.robomimic.config import RobomimicEnvConfig
from rl_garden.envs.robomimic.env import make_robomimic_env
from rl_garden.buffers.robomimic_dataset import ROBOMIMIC_LOW_DIM_OBS_KEYS

_KEY_DIMS = {
    "object": 10,
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "robot0_gripper_qvel": 2,
    "robot0_joint_pos": 7,
    "robot0_joint_pos_cos": 7,
    "robot0_joint_pos_sin": 7,
    "robot0_joint_vel": 7,
}
_OBS_DIM = sum(_KEY_DIMS.values())


class _FakeUnderlyingRobosuiteEnv:
    """Stand-in for robosuite's own env object (``EnvRobosuite.env``) --
    only its ``close()`` is used by rl-garden's adapter, since robomimic's
    ``EnvBase``/``EnvRobosuite`` itself exposes no ``close()`` method."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeRobosuiteEnv:
    """Stand-in for robomimic's ``EnvBase`` (old 4-tuple gym API, Dict obs)."""

    metadata: ClassVar[dict] = {}

    def __init__(self, *, omit_key: str | None = None) -> None:
        self.action_dimension = 7
        self.steps = 0
        self.env = _FakeUnderlyingRobosuiteEnv()
        # robomimic's real env force-activates extra observables (any
        # "joint_pos"/"eef_vel" match) not recorded into the offline hdf5 --
        # "robot0_eef_vel_lin" here stands in for that, and must be silently
        # ignored (not included) by the select-and-filter flatten.
        self._extra_key = "robot0_eef_vel_lin"
        self._omit_key = omit_key

    def _obs(self):
        obs = {
            key: np.full(dim, self.steps, dtype=np.float64)
            for key, dim in _KEY_DIMS.items()
            if key != self._omit_key
        }
        obs[self._extra_key] = np.zeros(6, dtype=np.float64)
        return obs

    def reset(self):
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.steps += 1
        success = self.steps == 2
        # EnvRobosuite.is_done() always returns False -- robosuite envs
        # always rollout to a fixed horizon.
        return self._obs(), float(success), False, {"is_success": {"task": success}}

    # Deliberately no close() here -- EnvRobosuite itself exposes none;
    # rl-garden's adapter must reach through to self.env.close() instead.


def _cfg(**kwargs) -> RobomimicEnvConfig:
    params = dict(env_id="Lift", num_envs=1, device="cpu", horizon=3)
    params.update(kwargs)
    return RobomimicEnvConfig(**params)


def test_robomimic_env_exposes_torch_vector_contract_and_success(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.robomimic.env._make_robosuite_env",
        lambda cfg: _FakeRobosuiteEnv(),
    )
    env = make_robomimic_env(_cfg(terminate_on_success=True, horizon=5))

    obs, _ = env.reset()
    assert obs.shape == (1, _OBS_DIM)
    assert obs.dtype == torch.float32
    action = torch.zeros((1, 7))
    _, reward1, terminated1, _, _ = env.step(action)
    _, reward2, terminated2, _, info2 = env.step(action)

    assert reward1.tolist() == [0.0]
    assert not terminated1.any()
    assert reward2.tolist() == [1.0]
    assert terminated2.tolist() == [True]
    assert info2["final_info"]["episode"]["success_at_end"].tolist() == [1.0]
    env.close()


def test_robomimic_env_close_delegates_to_underlying_env():
    # Regression test: EnvRobosuite itself exposes no close() method (found
    # via a real end-to-end smoke test against actual robosuite/robomimic --
    # an earlier version of this adapter called self.legacy_env.close()
    # directly and crashed with AttributeError at env teardown). The
    # adapter must reach through to self.legacy_env.env.close() instead.
    from rl_garden.envs.robomimic.env import _RobomimicGymV21Adapter

    fake = _FakeRobosuiteEnv()
    adapter = _RobomimicGymV21Adapter(fake)
    adapter.close()
    assert fake.env.closed is True


def test_robomimic_missing_obs_key_raises(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.robomimic.env._make_robosuite_env",
        lambda cfg: _FakeRobosuiteEnv(omit_key="robot0_gripper_qvel"),
    )
    with pytest.raises(KeyError, match="robot0_gripper_qvel"):
        make_robomimic_env(_cfg())


def test_robomimic_horizon_truncates_without_termination(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.robomimic.env._make_robosuite_env",
        lambda cfg: _FakeRobosuiteEnv(),
    )
    # terminate_on_success left False (default): no success wrapper, so
    # `is_done()` always False -> truncation is the only source of episode
    # end, firing exactly at horizon.
    env = make_robomimic_env(_cfg(horizon=3))
    env.reset()
    for step in range(1, 4):
        _, _, terminated, truncated, _ = env.step(torch.zeros((1, 7)))
        assert not terminated.any()
        if step < 3:
            assert not truncated.any()
        else:
            assert truncated.tolist() == [True]
    env.close()


def test_robomimic_reward_transform_is_explicit(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.robomimic.env._make_robosuite_env",
        lambda cfg: _FakeRobosuiteEnv(),
    )
    env = make_robomimic_env(_cfg(reward_scale=10.0, reward_bias=-5.0))
    env.reset()
    _, reward, _, _, _ = env.step(torch.zeros((1, 7)))
    assert reward.tolist() == [-5.0]  # step 1 raw reward is 0.0 -> 0*10 - 5
    env.close()


def test_make_robomimic_env_selects_vector_backend_by_num_envs(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.robomimic.env._make_robosuite_env",
        lambda cfg: _FakeRobosuiteEnv(),
    )
    calls = []
    real_sync = gym.vector.SyncVectorEnv

    def fake_sync(env_fns, **kwargs):
        calls.append("sync")
        return real_sync(env_fns, **kwargs)

    def fake_async(env_fns, **kwargs):
        calls.append("async")
        return real_sync(env_fns, **kwargs)

    monkeypatch.setattr("rl_garden.envs.robomimic.env.gym.vector.SyncVectorEnv", fake_sync)
    monkeypatch.setattr("rl_garden.envs.robomimic.env.gym.vector.AsyncVectorEnv", fake_async)

    env1 = make_robomimic_env(_cfg(num_envs=1))
    env1.close()
    env4 = make_robomimic_env(_cfg(num_envs=4))
    env4.close()

    assert calls == ["sync", "async"]
