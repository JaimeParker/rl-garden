"""ACRLPD, ACFQL, QAM and QGF execute an action chunk one action per env step
in every evaluation loop, and ACRLPD's and ACFQL's replay windows end with the
episode."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import ACFQL, ACRLPD, OfflineEnvSpec
from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.algorithms.offline import run_exact_episode_eval
from rl_garden.algorithms.qam import QAM
from rl_garden.algorithms.qgf import QGF
from rl_garden.training.evaluation import evaluate_agent
from rl_garden.training.off2on.runner import _offline_update_loop

ACTION_DIM = 2
HORIZON = 3
NUM_ENVS = 3
EVAL_EPISODE_LENS = (4, 7)


class _TraceEnv:
    """SAME_STEP autoreset; env ``i`` truncates after ``episode_lens[i]`` steps.
    Observations are ``(env, episode, step)`` and received actions are kept."""

    def __init__(self, episode_lens: tuple[int, ...]) -> None:
        self.num_envs = len(episode_lens)
        self._episode_lens = torch.tensor(episode_lens)
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, (3,), np.float32)}
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.received: list[torch.Tensor] = []
        self._episode = torch.zeros(self.num_envs, dtype=torch.long)
        self._step = torch.zeros(self.num_envs, dtype=torch.long)

    def _obs(self) -> dict[str, torch.Tensor]:
        columns = [torch.arange(self.num_envs), self._episode, self._step]
        return {"state": torch.stack(columns, dim=1).float()}

    def reset(self, seed=None):
        del seed
        self._episode.zero_()
        self._step.zero_()
        return self._obs(), {}

    def step(self, action):
        assert action.shape == (self.num_envs, ACTION_DIM)
        self.received.append(action.clone())
        self._step += 1
        truncated = self._step >= self._episode_lens
        infos = {}
        if truncated.any():
            infos = {
                "final_observation": self._obs(),
                "_final_observation": truncated.clone(),
                "final_info": {"episode": {"return": self._step.float()}},
                "_final_info": truncated.clone(),
            }
            self._episode[truncated] += 1
            self._step[truncated] = 0
        reward = torch.ones(self.num_envs)
        return self._obs(), reward, torch.zeros_like(truncated), truncated, infos

    def close(self) -> None:
        pass


def _make_agent(algorithm, horizon_length=HORIZON, **overrides):
    if algorithm in OFFLINE_ALGORITHMS:
        return _make_offline_agent(algorithm, horizon_length, **overrides)
    kwargs = {
        "env": _TraceEnv((6,) * NUM_ENVS),
        "eval_env": _TraceEnv(EVAL_EPISODE_LENS),
        "horizon_length": horizon_length,
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 300,
        "batch_size": 8,
        "learning_starts": 9,
        "training_freq": 3,
        "eval_freq": 0,
        "num_eval_steps": 8,
        "log_freq": 0,
        "std_log": False,
        "net_arch": [8],
    }
    kwargs.update(overrides)
    return algorithm(**kwargs)


OFFLINE_ALGORITHMS = {QAM: {"flow_steps": 2}, QGF: {"denoise_steps": 2}}


def _make_offline_agent(algorithm, horizon_length, **overrides):
    eval_env = _TraceEnv(EVAL_EPISODE_LENS)
    kwargs = {
        "env": OfflineEnvSpec(
            eval_env.single_observation_space, eval_env.single_action_space, num_envs=1
        ),
        "eval_env": eval_env,
        "horizon_length": horizon_length,
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 300,
        "batch_size": 8,
        "num_eval_steps": 8,
        "log_freq": 0,
        "std_log": False,
        "net_arch": [8],
        **OFFLINE_ALGORITHMS[algorithm],
    }
    kwargs.update(overrides)
    return algorithm(**kwargs)


def _script_chunks(agent, monkeypatch) -> list[int]:
    """Call ``k`` of ``predict`` returns the flat chunk ``k + 0.1 * position``
    (over 100, to stay in the action bounds) for every env."""
    calls: list[int] = []

    def predict(obs, deterministic=False):
        del deterministic
        calls.append(len(calls))
        positions = torch.arange(agent.horizon_length).repeat_interleave(ACTION_DIM)
        chunk = (calls[-1] + 0.1 * positions) / 100
        return chunk.expand(obs["state"].shape[0], -1).clone()

    monkeypatch.setattr(agent.policy, "predict", predict)
    return calls


EVALUATIONS = {
    "training": lambda agent: agent._evaluate(),
    "checkpoint": lambda agent: evaluate_agent(
        agent, agent.eval_env, seed=0, num_eval_episodes=3, max_eval_steps=8
    ),
    "exact_episodes": lambda agent: run_exact_episode_eval(
        agent, num_eval_episodes=3, num_eval_steps=8
    ),
}


@pytest.mark.parametrize("algorithm", [ACRLPD, ACFQL, QAM, QGF])
@pytest.mark.parametrize("evaluation", EVALUATIONS)
def test_evaluation_executes_the_chunk_step_by_step(algorithm, evaluation, monkeypatch):
    agent = _make_agent(algorithm)
    calls = _script_chunks(agent, monkeypatch)
    EVALUATIONS[evaluation](agent)
    received = 100 * torch.stack(agent.eval_env.received)[..., 0]
    # Env 0 ends its episode after 4 steps and env 1 after 7, in mid-chunk.
    expected = torch.tensor(
        [[0.0, 0.1, 0.2, 1.0, 2.0, 2.1, 2.2, 4.0], [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 3.0, 4.0]]
    )
    assert torch.allclose(received.T, expected, atol=1e-4)
    assert len(calls) == 5


@pytest.mark.parametrize("algorithm", [ACRLPD, ACFQL])
def test_evaluation_leaves_the_rollout_chunks_alone(algorithm, monkeypatch):
    agent = _make_agent(algorithm)
    _script_chunks(agent, monkeypatch)
    obs, _ = agent.env.reset()
    agent._on_env_reset(obs)
    executed = []
    for step in range(6):
        if step in (0, 1, 2, 5):
            agent._evaluate()
        action, _, _ = agent._rollout_action(obs, learning_has_started=True)
        obs, _, terminations, truncations, infos = agent.env.step(action)
        agent._post_rollout_step(None, terminations, truncations, infos)
        executed.append(100 * action[:, 0])
    values = torch.stack(executed)
    chunks = (values + 0.01).floor()
    positions = torch.tensor([0.0, 0.1, 0.2] * 2).unsqueeze(1).expand(-1, NUM_ENVS)
    assert torch.allclose(values - chunks, positions, atol=1e-4)
    assert (chunks[:3] == chunks[0]).all() and (chunks[3:] == chunks[3]).all()


@pytest.mark.parametrize("algorithm", [ACRLPD, ACFQL, QAM, QGF])
def test_horizon_of_one_evaluates_the_policy_on_every_step(algorithm, monkeypatch):
    agent = _make_agent(algorithm, horizon_length=1)

    def received_actions() -> torch.Tensor:
        agent.eval_env.received.clear()
        torch.manual_seed(0)
        agent._evaluate()
        return torch.stack(agent.eval_env.received)

    chunked = received_actions()
    monkeypatch.setattr(
        agent,
        "_eval_action_and_critic_action",
        lambda obs: BaseAlgorithm._eval_action_and_critic_action(agent, obs),
    )
    assert torch.equal(received_actions(), chunked)


def test_acrlpd_q_diagnostic_scores_each_chunk_where_it_was_chosen(monkeypatch):
    agent = _make_agent(ACRLPD, q_mc_diagnostics=True, num_eval_steps=7)
    _script_chunks(agent, monkeypatch)
    scored = []

    def eval_q_values(obs, actions):
        assert actions.shape[1] == HORIZON * ACTION_DIM
        scored.append((obs["state"].clone(), actions.clone()))
        return torch.zeros(actions.shape[0], 1)

    monkeypatch.setattr(agent, "_eval_q_values", eval_q_values)
    agent.gamma = 1.0
    metrics = agent._evaluate()
    # Env 0: steps 0 and 3 of a 4-step episode; env 1: steps 0, 3, 6 of 7. The
    # zero-valued critic bootstraps the truncations with 0.
    assert metrics["q_mc/num_steps"] == 5
    assert metrics["q_mc/mc_return_mean"] == pytest.approx((4 + 1 + 7 + 4 + 1) / 5)
    first_obs, first_actions = scored[0]
    assert torch.allclose(
        100 * first_actions[0], torch.tensor([0.0, 0.0, 0.1, 0.1, 0.2, 0.2]), atol=1e-5
    )
    assert (first_obs[:, 2] == 0).all()


def test_acfql_offline_stage_evaluates_single_actions():
    agent = _make_agent(ACFQL)
    agent.learn(total_timesteps=30)
    _offline_update_loop(agent, 2, None, log_freq=0, std_log=False, eval_freq=1)
    assert len(agent.eval_env.received) >= agent.num_eval_steps


@pytest.mark.parametrize("algorithm", [ACRLPD, ACFQL])
def test_chunk_windows_end_at_a_truncation(algorithm):
    agent = _make_agent(algorithm)
    agent.learn(total_timesteps=60)
    assert agent.replay_buffer.episode_ends.any()
    past_the_end = 0
    for _ in range(20):
        batch = agent.replay_buffer.sample(64)
        obs, next_obs = batch.obs["state"], batch.next_obs["state"]
        assert torch.equal(obs[:, :2], next_obs[:, :2])
        remaining = (6 - obs[:, 2]).clamp(max=HORIZON)
        assert torch.equal(batch.valid.sum(1).float(), remaining)
        assert torch.equal(next_obs[:, 2] - obs[:, 2], remaining)
        past_the_end += int((remaining < HORIZON).sum())
    assert past_the_end > 0
