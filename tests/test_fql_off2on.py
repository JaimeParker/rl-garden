"""Focused lifecycle tests for standard FQL offline-to-online continuation."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import Off2OnFQL


class _DoneEnv:
    def __init__(self, *, terminated: bool) -> None:
        self.terminated = terminated
        self.num_envs = 1
        self.single_observation_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = batch_space(self.single_action_space, 1)

    def reset(self, seed=None):
        del seed
        return torch.zeros(1, 3), {}

    def step(self, action):
        del action
        return (
            torch.zeros(1, 3),
            torch.ones(1),
            torch.tensor([self.terminated]),
            torch.tensor([not self.terminated]),
            {
                "final_observation": torch.full((1, 3), 7.0),
                "_final_observation": torch.tensor([True]),
            },
        )


class _NeverDoneEnv(_DoneEnv):
    def step(self, action):
        del action
        return torch.zeros(1, 3), torch.ones(1), torch.tensor([False]), torch.tensor([False]), {}


def _agent(
    *,
    terminated: bool = False,
    reference_compatible: bool = False,
    eval_env=None,
    num_eval_episodes=None,
    num_eval_steps: int = 50,
) -> Off2OnFQL:
    net_arch = [512, 512, 512, 512] if reference_compatible else [8]
    return Off2OnFQL(
        _DoneEnv(terminated=terminated),
        eval_env=eval_env,
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=2,
        net_arch=net_arch,
        reference_compatible=reference_compatible,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        log_freq=0,
        num_eval_episodes=num_eval_episodes,
        num_eval_steps=num_eval_steps,
    )


def _fill_offline(agent: Off2OnFQL) -> None:
    for _ in range(3):
        agent.replay_buffer.add(
            {"state": torch.zeros(1, 3)},
            {"state": torch.ones(1, 3)},
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.zeros(1),
        )


@pytest.mark.parametrize("reference_compatible", [False, True])
def test_shell_constructs_in_both_fql_modes(reference_compatible: bool) -> None:
    agent = _agent(reference_compatible=reference_compatible)
    assert agent.reference_compatible is reference_compatible


def test_reference_eval_fails_when_step_cap_prevents_requested_episodes() -> None:
    agent = _agent(
        reference_compatible=True,
        eval_env=_NeverDoneEnv(terminated=False),
        num_eval_episodes=1,
        num_eval_steps=1,
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        agent._evaluate()


def test_append_switch_preserves_replay_and_optimizer_state() -> None:
    agent = _agent()
    _fill_offline(agent)
    agent.train(1)
    actor_optimizer = agent.actor_optimizer
    critic_optimizer = agent.critic_optimizer
    actor_params_before = [param.detach().clone() for param in agent.policy.actor_parameters()]
    optimizer_state_before = {
        (index, key): value.detach().clone()
        for index, state in enumerate(agent.actor_optimizer.state.values())
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }

    agent.switch_to_online_mode(online_replay_mode="append")

    assert agent.offline_replay_buffer is None
    assert len(agent.replay_buffer) == 3
    assert agent.actor_optimizer is actor_optimizer
    assert agent.critic_optimizer is critic_optimizer
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_params_before, agent.policy.actor_parameters())
    )
    optimizer_state_after = {
        (index, key): value
        for index, state in enumerate(agent.actor_optimizer.state.values())
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    assert optimizer_state_before.keys() == optimizer_state_after.keys()
    assert all(torch.equal(optimizer_state_before[key], optimizer_state_after[key]) for key in optimizer_state_before)


@pytest.mark.parametrize(("terminated", "expected_done"), [(False, 0.0), (True, 1.0)])
def test_first_online_update_preserves_final_observation_and_done_mask(
    terminated: bool, expected_done: float
) -> None:
    agent = _agent(terminated=terminated)
    _fill_offline(agent)
    agent.switch_to_online_mode(online_replay_mode="append")

    agent.learn(total_timesteps=1)

    assert agent._global_update == 1
    assert len(agent.replay_buffer) == 4
    expected_next_obs = torch.zeros(3) if terminated else torch.full((3,), 7.0)
    assert torch.equal(agent.replay_buffer.next_obs["state"][3, 0], expected_next_obs)
    assert agent.replay_buffer.dones[3, 0].item() == expected_done


def test_fixed_cadence_final_eval_runs_once_at_online_budget(monkeypatch) -> None:
    eval_env = _DoneEnv(terminated=True)
    agent = _agent(eval_env=eval_env, num_eval_episodes=1)
    agent.eval_freq = 2
    _fill_offline(agent)
    agent.switch_to_online_mode(online_replay_mode="append")
    agent._global_step = 2
    evaluations: list[int] = []
    monkeypatch.setattr(agent, "_evaluate", lambda: evaluations.append(agent._global_step) or {})

    agent.learn(total_timesteps=4)
    agent.learn(total_timesteps=4)

    assert evaluations == [4]


def test_fixed_cadence_keeps_intermediate_and_terminal_eval(monkeypatch) -> None:
    agent = _agent(eval_env=_DoneEnv(terminated=True), num_eval_episodes=1)
    agent.eval_freq = 2
    _fill_offline(agent)
    agent.switch_to_online_mode(online_replay_mode="append")
    evaluations: list[int] = []
    monkeypatch.setattr(agent, "_evaluate", lambda: evaluations.append(agent._global_step) or {})

    agent.learn(total_timesteps=4)

    assert evaluations == [2, 4]


@pytest.mark.parametrize(
    ("eval_env", "eval_freq", "target"),
    [(_DoneEnv(terminated=True), 0, 2), (None, 2, 2)],
)
def test_final_eval_is_noop_without_enabled_eval(monkeypatch, eval_env, eval_freq, target) -> None:
    agent = _agent(eval_env=eval_env)
    agent.eval_freq = eval_freq
    _fill_offline(agent)
    agent.switch_to_online_mode(online_replay_mode="append")
    evaluations: list[int] = []
    monkeypatch.setattr(agent, "_evaluate", lambda: evaluations.append(agent._global_step) or {})

    agent.learn(total_timesteps=target)

    assert evaluations == []


def test_final_eval_skips_nonboundary_budget(monkeypatch) -> None:
    agent = _agent(eval_env=_DoneEnv(terminated=True))
    agent.eval_freq = 2
    _fill_offline(agent)
    agent.switch_to_online_mode(online_replay_mode="append")
    evaluations: list[int] = []
    monkeypatch.setattr(agent, "_evaluate", lambda: evaluations.append(agent._global_step) or {})

    agent.learn(total_timesteps=3)

    assert evaluations == [2]
