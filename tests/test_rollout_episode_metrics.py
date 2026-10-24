"""Rollout episode stats are normalised to canonical names (return/episode_len)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import PPO, SAC
from rl_garden.common.eval_metrics import canonical_episode_metric_key

RAW_EPISODE = {"r": 2.0, "l": 3.0, "t": 0.1, "_r": True}
CANONICAL_EPISODE = {"return": 2.0, "success_at_end": 1.0}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("r", "return"),
        ("l", "episode_len"),
        ("success", "success_at_end"),
        ("is_success", "success_at_end"),
        ("return", "return"),
        ("success_at_end", "success_at_end"),
        ("t", None),
        ("_r", None),
    ],
)
def test_canonical_episode_metric_key(raw, expected):
    assert canonical_episode_metric_key(raw) == expected


class _FinalInfoVecEnv:
    """Every step all envs finish an episode and report ``episode_stats``."""

    num_envs = 2

    def __init__(self, episode_stats: dict) -> None:
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_envs, 2), dtype=np.float32)
        self._episode_stats = episode_stats

    def reset(self, seed=None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        del actions
        episode = {
            k: (
                torch.full((self.num_envs,), v, dtype=torch.bool)
                if isinstance(v, bool)
                else torch.full((self.num_envs,), float(v))
            )
            for k, v in self._episode_stats.items()
        }
        infos = {
            "final_info": {"episode": episode},
            "_final_info": torch.ones(self.num_envs, dtype=torch.bool),
            "final_observation": torch.zeros(self.num_envs, 4),
        }
        done = torch.ones(self.num_envs, dtype=torch.bool)
        return (
            torch.zeros(self.num_envs, 4),
            torch.randn(self.num_envs),  # non-constant: PPO explained_variance needs var>0
            done,
            torch.zeros(self.num_envs, dtype=torch.bool),
            infos,
        )

    def close(self) -> None:
        return None


class _RecordingLogger:
    def __init__(self) -> None:
        self.scalars: dict[str, list[float]] = defaultdict(list)

    def add_scalar(self, key, value, step):
        self.scalars[key].append(value)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _sac(env) -> SAC:
    return SAC(
        env=env,
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        log_freq=1,
        std_log=True,
        net_arch={"pi": [16], "qf": [16]},
    )


def _ppo(env) -> PPO:
    return PPO(
        env=env,
        device="cpu",
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        eval_freq=0,
        log_freq=4,
        std_log=True,
        target_kl=None,
        net_arch=[16],
    )


def _train_keys(logger: _RecordingLogger) -> set[str]:
    return {k for k in logger.scalars if k.startswith("train/")}


@pytest.mark.parametrize("make_agent, total", [(_sac, 4), (_ppo, 4)])
def test_raw_recordepisodestatistics_keys_are_canonicalised(
    make_agent, total, capsys
):
    agent = make_agent(_FinalInfoVecEnv(RAW_EPISODE))
    agent.logger = _RecordingLogger()
    agent.learn(total_timesteps=total)

    assert _train_keys(agent.logger) == {"train/return", "train/episode_len"}
    assert agent.logger.scalars["train/return"][0] == pytest.approx(2.0)
    out = capsys.readouterr().out
    assert "return=2.0000" in out
    assert "return=nan" not in out


@pytest.mark.parametrize("make_agent, total", [(_sac, 4), (_ppo, 4)])
def test_canonical_keys_pass_through_unchanged(make_agent, total, capsys):
    agent = make_agent(_FinalInfoVecEnv(CANONICAL_EPISODE))
    agent.logger = _RecordingLogger()
    agent.learn(total_timesteps=total)

    assert _train_keys(agent.logger) == {"train/return", "train/success_at_end"}
    assert "return=2.0000" in capsys.readouterr().out


def test_sac_without_logger_still_canonicalises_window_metrics():
    agent = _sac(_FinalInfoVecEnv(RAW_EPISODE))
    agent.stats_window_size = 10
    agent._episode_metric_windows = defaultdict(list)
    agent.learn(total_timesteps=2)
    assert set(agent._episode_metric_windows) == {"return", "episode_len"}


class _EnvLogVecEnv(_FinalInfoVecEnv):
    """``_FinalInfoVecEnv`` that also reports ``infos["log"]`` scalar aggregates."""

    def step(self, actions):
        out = super().step(actions)
        out[4]["log"] = {
            "Episode_Reward/alive": torch.tensor(0.5),
            "Metrics/ratio": 0.25,
            "count": torch.tensor([3]),
            "per_env": torch.ones(self.num_envs),
            "label": "skipped",
        }
        return out


@pytest.mark.parametrize("make_agent, total", [(_sac, 4), (_ppo, 4)])
def test_env_log_scalars_are_logged_under_env_prefix(make_agent, total):
    agent = make_agent(_EnvLogVecEnv(CANONICAL_EPISODE))
    agent.logger = _RecordingLogger()
    agent.learn(total_timesteps=total)

    env_keys = {k for k in agent.logger.scalars if k.startswith("env/")}
    assert env_keys == {"env/Episode_Reward/alive", "env/Metrics/ratio", "env/count"}
    assert agent.logger.scalars["env/Episode_Reward/alive"][0] == pytest.approx(0.5)
    assert agent.logger.scalars["env/Metrics/ratio"][0] == pytest.approx(0.25)
    assert agent.logger.scalars["env/count"][0] == pytest.approx(3.0)
