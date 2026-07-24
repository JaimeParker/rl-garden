from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.policies.base import BasePolicy


class _DummyPolicy(BasePolicy):
    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic
        return torch.zeros(obs.shape[0], 1)


class _DummyEnv:
    num_envs = 2
    single_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))


class _PartialDoneEvalEnv(_DummyEnv):
    """Vector env where only some sub-envs finish on a given step.

    Regression fixture for the bug where `_evaluate()` averaged the full-width
    `final_info` array instead of filtering it down with `_final_info`, mixing
    in stale values from sub-envs that hadn't finished yet.
    """

    def __init__(self, returns: torch.Tensor, done_mask: torch.Tensor) -> None:
        self.returns = returns
        self.done_mask = done_mask
        self.step_idx = 0

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(2, 4), {}

    def step(self, actions):
        del actions
        obs = torch.zeros(2, 4)
        rewards = torch.zeros(2)
        terminations = self.done_mask
        truncations = torch.zeros(2, dtype=torch.bool)
        infos = {
            "final_info": {"episode": {"return": self.returns}},
            "_final_info": self.done_mask,
        }
        self.step_idx += 1
        return obs, rewards, terminations, truncations, infos


class _Algo(BaseAlgorithm):
    def _setup_model(self) -> None:
        pass

    def learn(self, total_timesteps: int) -> "BaseAlgorithm":
        del total_timesteps
        return self


def _algo(eval_env, num_eval_steps=1) -> _Algo:
    algo = _Algo(env=_DummyEnv(), eval_env=eval_env, device="cpu")
    algo.policy = _DummyPolicy()
    algo.num_eval_steps = num_eval_steps
    return algo


def test_evaluate_masks_by_final_info_not_done_subenvs():
    # env 0 finishes this step (return=10.0); env 1 is still running (return=99.0
    # is a stale full-width value that must be excluded from the average).
    eval_env = _PartialDoneEvalEnv(
        returns=torch.tensor([10.0, 99.0]),
        done_mask=torch.tensor([True, False]),
    )
    metrics = _algo(eval_env)._evaluate()

    assert metrics["return"] == pytest.approx(10.0)


def test_evaluate_returns_empty_when_no_subenv_finished():
    eval_env = _PartialDoneEvalEnv(
        returns=torch.tensor([10.0, 99.0]),
        done_mask=torch.tensor([False, False]),
    )
    metrics = _algo(eval_env)._evaluate()

    assert metrics == {}


def test_evaluate_returns_empty_dict_when_eval_env_is_none():
    assert _algo(None)._evaluate() == {}


class _RawGymKeyEvalEnv(_DummyEnv):
    """final_info shaped like gymnasium.wrappers.RecordEpisodeStatistics
    (used by the mujoco/minari env backends): raw "r"/"l"/"t" keys plus
    "_r"/"_l"/"_t" mask siblings injected by VectorEnv._add_info."""

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(2, 4), {}

    def step(self, actions):
        del actions
        obs = torch.zeros(2, 4)
        rewards = torch.zeros(2)
        terminations = torch.tensor([True, True])
        truncations = torch.zeros(2, dtype=torch.bool)
        infos = {
            "final_info": {
                "episode": {
                    "r": torch.tensor([10.0, 20.0]),
                    "l": torch.tensor([5.0, 6.0]),
                    "t": torch.tensor([0.01, 0.02]),
                    "_r": torch.tensor([True, True]),
                    "_l": torch.tensor([True, True]),
                    "_t": torch.tensor([True, True]),
                }
            },
            "_final_info": torch.tensor([True, True]),
        }
        return obs, rewards, terminations, truncations, infos


def test_evaluate_aliases_raw_gym_episode_keys_and_drops_wall_clock_time():
    metrics = _algo(_RawGymKeyEvalEnv())._evaluate()

    assert metrics["return"] == pytest.approx(15.0)
    assert metrics["episode_len"] == pytest.approx(5.5)
    assert "t" not in metrics
    assert "episode_time" not in metrics
    assert "_r" not in metrics
    assert "_l" not in metrics
    assert "_t" not in metrics


class _SuccessKeyEvalEnv(_DummyEnv):
    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(2, 4), {}

    def step(self, actions):
        del actions
        obs = torch.zeros(2, 4)
        rewards = torch.zeros(2)
        terminations = torch.tensor([True, True])
        truncations = torch.zeros(2, dtype=torch.bool)
        infos = {
            "final_info": {"episode": {"success": torch.tensor([1.0, 0.0])}},
            "_final_info": torch.tensor([True, True]),
        }
        return obs, rewards, terminations, truncations, infos


def test_evaluate_aliases_success_to_success_at_end():
    metrics = _algo(_SuccessKeyEvalEnv())._evaluate()

    assert metrics["success_at_end"] == pytest.approx(0.5)
    assert "success" not in metrics
