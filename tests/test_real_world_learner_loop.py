"""Tests for LearnerLoop against a fake OffPolicyAlgorithm double.

LearnerLoop is meant to be algorithm-agnostic: it only ever touches
``replay_buffer.add``, ``train``, ``learning_starts``, ``utd``,
``buffer_device``, and ``policy.state_dict`` -- the fake below exposes
exactly that surface, nothing more, so these tests double as a check that
LearnerLoop doesn't reach for anything beyond the documented public
interface of ``OffPolicyAlgorithm``.
"""
from __future__ import annotations

import threading
import time

import torch

from rl_garden.real_world.learner_loop import LearnerLoop
from rl_garden.real_world.sync import ActorSyncClient


class _FakeReplayBuffer:
    def __init__(self):
        self.add_calls = []

    def add(self, obs, next_obs, action, reward, done, **kwargs):
        self.add_calls.append(
            dict(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, **kwargs)
        )


class _FakePolicy:
    def __init__(self):
        self.version = 0

    def state_dict(self):
        return {"version": torch.tensor([self.version])}


class _FakeAgent:
    def __init__(self, learning_starts: int = 2, utd: float = 1.0):
        self.buffer_device = "cpu"
        self.replay_buffer = _FakeReplayBuffer()
        self.learning_starts = learning_starts
        self.utd = utd
        self.policy = _FakePolicy()
        self.train_calls: list[int] = []

    def train(self, gradient_steps: int, compute_info: bool = False):
        del compute_info
        self.train_calls.append(gradient_steps)
        self.policy.version += 1
        return {}


def _transition(value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.full((1, 4), value),
        "next_obs": torch.full((1, 4), value + 1),
        "action": torch.zeros(1, 2),
        "reward": torch.tensor([1.0]),
        "done": torch.tensor([False]),
    }


def test_on_transition_adds_to_replay_buffer_with_device_cast():
    agent = _FakeAgent()
    loop = LearnerLoop(agent, "127.0.0.1", 0)
    loop._on_transition(_transition(1.0))
    assert len(agent.replay_buffer.add_calls) == 1
    call = agent.replay_buffer.add_calls[0]
    torch.testing.assert_close(call["obs"], torch.full((1, 4), 1.0))
    assert call["obs"].device == torch.device("cpu")
    assert loop.received_transitions == 1


def test_on_transition_passes_extra_kwargs_through():
    agent = _FakeAgent()
    loop = LearnerLoop(agent, "127.0.0.1", 0)
    transition = _transition(0.0)
    transition["discounts"] = torch.tensor([0.99])
    loop._on_transition(transition)
    assert "discounts" in agent.replay_buffer.add_calls[0]


def test_run_waits_for_learning_starts_before_training():
    agent = _FakeAgent(learning_starts=3, utd=1.0)
    loop = LearnerLoop(agent, "127.0.0.1", 0, train_freq=1, publish_freq=1, idle_poll_interval=0.02)
    loop._on_transition(_transition())
    loop._on_transition(_transition())  # only 2 of 3 required

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        assert agent.train_calls == [], "train() must not run before learning_starts"
        loop._on_transition(_transition())  # crosses learning_starts == 3
        deadline = time.time() + 5.0
        while not agent.train_calls and time.time() < deadline:
            time.sleep(0.02)
        assert agent.train_calls, "train() never ran after learning_starts was reached"
        assert agent.train_calls[0] == max(1, int(1 * agent.utd))
    finally:
        loop.stop()
        thread.join(timeout=5.0)


def test_refresh_offline_data_default_is_a_harmless_noop():
    agent = _FakeAgent()
    loop = LearnerLoop(agent, "127.0.0.1", 0)
    assert loop._refresh_offline_data() is None


def test_refresh_offline_data_hook_is_called_each_training_iteration():
    calls = []

    class _CountingLearnerLoop(LearnerLoop):
        def _refresh_offline_data(self) -> None:
            calls.append(1)

    agent = _FakeAgent(learning_starts=1, utd=1.0)
    loop = _CountingLearnerLoop(agent, "127.0.0.1", 0, train_freq=1, publish_freq=1, idle_poll_interval=0.02)
    loop._on_transition(_transition())

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 5.0
        while not agent.train_calls and time.time() < deadline:
            time.sleep(0.02)
        assert agent.train_calls, "learner never trained"
        assert len(calls) >= 1
        assert len(calls) == len(agent.train_calls)
    finally:
        loop.stop()
        thread.join(timeout=5.0)


def test_run_publishes_params_reachable_by_a_real_actor_client():
    agent = _FakeAgent(learning_starts=1, utd=1.0)
    loop = LearnerLoop(agent, "127.0.0.1", 0, train_freq=1, publish_freq=1, idle_poll_interval=0.02)
    loop._on_transition(_transition())

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 5.0
        while not agent.train_calls and time.time() < deadline:
            time.sleep(0.02)
        assert agent.train_calls, "learner never trained"

        host, port = loop._server.server_address
        client = ActorSyncClient(f"http://{host}:{port}", poll_interval=10.0)
        client._poll_once()
        params = client.latest_policy_params()
        assert params is not None
        assert params["version"].item() >= 1
    finally:
        loop.stop()
        thread.join(timeout=5.0)
