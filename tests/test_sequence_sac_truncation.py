"""RecurrentSAC/TransformerSAC targets around episode ends inside a sampled window.

R2D2 bootstraps a return that an episode end cut short from the last stored
observation of that episode, with the hidden state that led to it. The reference
here unrolls each sampled column by hand from the rows the buffer stores.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import RecurrentSAC, TransformerSAC

NUM_ENVS = 2
BURN_IN, LEARNING, FORWARD = 4, 4, 3
WINDOW = BURN_IN + LEARNING + FORWARD
EPISODE_LENGTHS = (5, 7)
CPU_OR_CUDA = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


class _Env:
    """Random observations; env ``i`` ends its episode every ``EPISODE_LENGTHS[i]`` steps."""

    num_envs = NUM_ENVS

    def __init__(self, terminate: bool, lengths, device: str) -> None:
        space = spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (3,), np.float32)})
        self.single_observation_space = space
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, (NUM_ENVS, 3), np.float32)}
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (NUM_ENVS, 2), np.float32)
        self.terminate, self.device = terminate, device
        self.lengths = torch.tensor(lengths, device=device)
        self.generator = torch.Generator().manual_seed(0)
        self.steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)

    def _obs(self):
        return {"state": torch.randn(NUM_ENVS, 3, generator=self.generator).to(self.device)}

    def reset(self, seed=None, options=None):
        self.steps.zero_()
        return self._obs(), {}

    def step(self, actions):
        self.steps += 1
        ended = self.steps >= self.lengths
        final, obs = self._obs(), self._obs()
        self.steps[ended] = 0
        none = torch.zeros_like(ended)
        terminated, truncated = (ended, none) if self.terminate else (none, ended)
        return obs, torch.ones(NUM_ENVS, device=self.device), terminated, truncated, {
            "final_observation": final
        }


def _agent(cls, *, terminate=False, lengths=EPISODE_LENGTHS, device="cpu", **kwargs):
    if device == "cuda" and torch.cuda.device_count() == 0:
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    agent = cls(
        _Env(terminate, lengths, device),
        burn_in_len=BURN_IN, learning_len=LEARNING, forward_len=FORWARD,
        buffer_size=NUM_ENVS * 60, buffer_device=device, device=device, gamma=0.9,
        learning_starts=10**6, training_freq=NUM_ENVS, batch_size=8, net_arch=[16],
        eval_freq=0, std_log=False, save_final_checkpoint=False,
        **({"rnn_hidden_size": 8, **kwargs} if cls is RecurrentSAC else
           {"embed_dim": 8, "head_dim": 4, "num_heads": 2, "num_transformer_layers": 2,
            "memory_len": 4, **kwargs}),
    )
    agent.learn(NUM_ENVS * 40)
    agent.policy.eval()
    return agent


def _all_valid(agent):
    buffer = agent.replay_buffer
    leaf = torch.arange(buffer.checkpoint_capacity * NUM_ENVS, device=buffer.storage_device)
    slot, env = leaf % buffer.checkpoint_capacity, leaf // buffer.checkpoint_capacity
    t0 = buffer._ckpt_slot_to_pos[slot, env]
    valid = (t0 >= 0) & buffer._valid_window_batch(t0.clamp(min=0), env)
    return t0[valid], env[valid], leaf[valid]


def _sample_all(agent):
    t0, env, leaf = _all_valid(agent)
    buffer = agent.replay_buffer
    buffer._sample_valid_window_starts = lambda n, generator=None: (
        t0, env, leaf, torch.ones(len(t0), device=t0.device))
    try:
        return t0, env, buffer.sample(len(t0))
    finally:
        del buffer._sample_valid_window_starts


def _target_features(agent, data):
    """The features ``_critic_loss`` evaluates the target critic on, ``(learning_len, B, H)``."""
    captured = []
    min_q_value = agent.policy.min_q_value

    def capture(features, action, **kwargs):
        if kwargs.get("target"):
            captured.append(features.detach().clone())
        return min_q_value(features, action, **kwargs)

    agent.policy.min_q_value = capture
    try:
        agent._critic_loss(data)
    finally:
        del agent.policy.min_q_value
    return captured[-1].reshape(LEARNING, data.episode_starts.shape[1], -1)


def _reference(agent, t0: int, env: int, initial):
    """Per learning row, the bootstrap feature (``None``: terminated)."""
    buffer, encoder = agent.replay_buffer, agent.policy.recurrent_encoder

    def latent(obs):
        return agent.policy.extract_features({"state": obs.unsqueeze(0)})

    state, rows, after_end = initial, [], {}
    with torch.no_grad():
        for j in range(WINDOW):
            row = t0 + j
            if row >= buffer.pos:
                break  # past the newest row
            starts = buffer._ep_relative_step[row, env].eq(0).float().reshape(1)
            output, state = encoder.step(latent(buffer.obs["state"][row, env]), state, starts)
            rows.append(output[0])
            if bool(buffer.episode_ends[row, env]):
                final = buffer._final_obs["state"][buffer._final_slot_ids[row, env]]
                after_end[j] = encoder.step(latent(final), state)[0][0]
    features = []
    for first in range(BURN_IN, BURN_IN + LEARNING):
        ends = [j for j in range(first, first + FORWARD) if j in after_end]
        if not ends:
            features.append(rows[first + FORWARD])
        elif bool(buffer.dones[t0 + ends[0], env]):
            features.append(None)
        else:
            features.append(after_end[ends[0]])
    return features


@pytest.mark.parametrize("device", CPU_OR_CUDA)
@pytest.mark.parametrize(
    "cls,kwargs",
    [(RecurrentSAC, {}), (RecurrentSAC, {"rnn_type": "gru", "rnn_num_layers": 2}), (TransformerSAC, {})],
    ids=["lstm", "gru2", "gtrxl"],
)
def test_truncated_returns_bootstrap_from_their_own_final_observation(cls, kwargs, device):
    agent = _agent(cls, device=device, **kwargs)
    t0, env, data = _sample_all(agent)
    features = _target_features(agent, data)
    assert (data.final_index >= 0).any() and (data.final_index < 0).any()
    state = agent._initial_state_from_sample(data)
    for i, (start, column) in enumerate(zip(t0.tolist(), env.tolist())):
        initial = agent.policy.recurrent_encoder.index_state(state, torch.tensor([i], device=device))
        for j, expected in enumerate(_reference(agent, start, column, initial)):
            torch.testing.assert_close(features[j, i], expected, atol=1e-4, rtol=1e-3)
    agent.policy.train()
    loss, _ = agent._critic_loss(data)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("cls", [RecurrentSAC, TransformerSAC])
def test_terminated_returns_do_not_bootstrap(cls):
    agent = _agent(cls, terminate=True)
    t0, env, data = _sample_all(agent)
    assert data.final_steps.numel() == 0
    state = agent._initial_state_from_sample(data)
    features = _target_features(agent, data)
    for i, (start, column) in enumerate(zip(t0.tolist(), env.tolist())):
        initial = agent.policy.recurrent_encoder.index_state(state, torch.tensor([i]))
        for j, expected in enumerate(_reference(agent, start, column, initial)):
            if expected is None:
                assert float(data.discounts[j, i]) == 0.0
            else:
                torch.testing.assert_close(features[j, i], expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("cls", [RecurrentSAC, TransformerSAC])
def test_learning_rows_after_an_end_read_the_observation_the_policy_acted_on(cls):
    agent = _agent(cls)
    t0, env, data = _sample_all(agent)
    buffer = agent.replay_buffer
    for i, (start, column) in enumerate(zip(t0.tolist(), env.tolist())):
        rows = torch.arange(start, start + BURN_IN + LEARNING)
        assert torch.equal(data.obs["state"][: BURN_IN + LEARNING, i], buffer.obs["state"][rows, column])


@pytest.mark.parametrize("cls", [RecurrentSAC, TransformerSAC])
def test_unread_rows_do_not_reach_the_losses(cls):
    agent = _agent(cls)
    t0, _, data = _sample_all(agent)
    buffer = agent.replay_buffer
    past_newest = (t0 + WINDOW - 1 >= buffer.pos).nonzero().flatten()
    assert past_newest.numel() > 0  # the episode ended, and its last returns stop there

    def losses(batch):
        torch.manual_seed(0)
        critic, _ = agent._critic_loss(batch)
        torch.manual_seed(0)
        actor, _ = agent._actor_loss_from_batch(batch)
        return critic.detach(), actor.detach()

    before = losses(data)
    for i in past_newest.tolist():
        data.obs["state"][buffer.pos - int(t0[i]) :, i] = 100.0
        data.episode_starts[buffer.pos - int(t0[i]) :, i] = 1.0
    assert all(map(torch.equal, before, losses(data)))


@pytest.mark.parametrize("cls", [RecurrentSAC, TransformerSAC])
def test_windows_without_an_end_take_the_plain_unroll(cls, monkeypatch):
    agent = _agent(cls, lengths=(1000, 1000))
    _, _, data = _sample_all(agent)
    assert data.final_steps.numel() == 0
    monkeypatch.setattr(agent.policy, "windowed_and_final_features", None, raising=False)
    features = _target_features(agent, data)
    state = agent._initial_state_from_sample(data)
    tail = agent.policy.windowed_features(data.obs, state, data.episode_starts, BURN_IN)
    assert torch.equal(features, tail[FORWARD:])


def test_sampling_fails_instead_of_reading_missing_rows():
    """Rows enough for a window only past the newest row: there is no data to train on yet."""
    agent = _agent(RecurrentSAC, lengths=(5, 5))
    buffer = agent._build_replay_buffer()
    hidden = agent.policy.recurrent_encoder.get_initial_state(NUM_ENVS, torch.device("cpu"))
    for step in range(6):
        end = torch.full((NUM_ENVS,), step == 4)
        obs = {"state": torch.zeros(NUM_ENVS, 3)}
        buffer.add(obs, obs, torch.zeros(NUM_ENVS, 2), torch.zeros(NUM_ENVS), end & False, end, hidden)
    with pytest.raises(RuntimeError, match="enough valid recurrent windows.*=11 consecutive"):
        buffer.sample(2)
