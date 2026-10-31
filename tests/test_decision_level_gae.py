"""Decision-level (semi-Markov) rollouts: SMDP GAE, decision storage in the
rollout buffers, the consumer-side decision key reader, and the env-side
handling of ``infos["decision"]`` / ``infos["decision_components"]``
(episode boundary contract rules 9-10).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from _decision_fakes import (
    NUM_ENVS,
    NUM_STEPS,
    _FixedHoldVecEnv,
    _scripted_decisions,
    _scripted_flags,
)
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from test_frame_stack_wrapper import _FakeBatchedEnv
from test_vector_env_contract import _TinyEnv

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.algorithms._decision_mask import DecisionMaskReader
from rl_garden.algorithms.gail import GAIL
from rl_garden.buffers import RolloutBuffer
from rl_garden.buffers.gae import compute_gae
from rl_garden.buffers.recurrent_rollout_buffer import RecurrentRolloutBuffer
from rl_garden.envs.vector_env import TorchVectorEnvAdapter
from rl_garden.envs.wrappers import ActionRepeatWrapper, ImageFrameStackWrapper
from rl_garden.envs.wrappers._batch_utils import ChunkAccumulator
from rl_garden.envs.wrappers.action_chunk import ActionChunkWrapper

GAMMA = 0.9
GAE_LAMBDA = 0.8


def _smdp_reference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    final_values: torch.Tensor,
    dones: torch.Tensor,
    decisions: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SMDP returns written directly from their definition (no recursion).

    ``dones`` are after-step flags. For every step ``t`` the return is the
    discounted reward sum up to the end of the current hold plus the
    discounted tail: the next decision's ``V + lambda * A``, the terminal
    bootstrap ``final_values`` when the episode ends inside the hold, or
    ``last_values`` when the hold runs past the window.
    """
    rewards, values, final_values, last_values = (
        x.double() for x in (rewards, values, final_values, last_values)
    )
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros(num_steps, num_envs, dtype=torch.float64)
    returns = torch.zeros(num_steps, num_envs, dtype=torch.float64)
    for env in range(num_envs):
        for step in reversed(range(num_steps)):
            reward_sum, discount, j = 0.0, 1.0, step
            while True:
                reward_sum += discount * rewards[j, env]
                discount *= gamma
                if dones[j, env]:
                    tail = final_values[j, env]
                    break
                if j == num_steps - 1:
                    tail = last_values[env]
                    break
                if decisions[j + 1, env]:
                    tail = values[j + 1, env] + gae_lambda * advantages[j + 1, env]
                    break
                j += 1
            returns[step, env] = reward_sum + discount * tail
            advantages[step, env] = returns[step, env] - values[step, env]
    return advantages, returns


def _step_lambda(decisions: torch.Tensor, gae_lambda: float) -> torch.Tensor:
    step_lambda = torch.full(decisions.shape, gae_lambda, dtype=torch.float32)
    step_lambda[:-1] = torch.where(decisions[1:], gae_lambda, 1.0)
    return step_lambda


def _episode_starts(dones: torch.Tensor) -> torch.Tensor:
    starts = torch.zeros_like(dones, dtype=torch.float32)
    starts[1:] = dones[:-1].float()
    return starts


# ---------------------------------------------------------------------------
# (a) compute_gae(step_lambda=...)
# ---------------------------------------------------------------------------


def test_compute_gae_step_lambda_matches_smdp_reference():
    generator = torch.Generator().manual_seed(0)
    rewards = torch.randn(NUM_STEPS, NUM_ENVS, generator=generator)
    values = torch.randn(NUM_STEPS, NUM_ENVS, generator=generator)
    last_values = torch.randn(NUM_ENVS, generator=generator)
    final_values = torch.zeros(NUM_STEPS, NUM_ENVS)
    final_values[4, 1] = 1.7  # truncation inside a hold bootstraps
    dones = _scripted_flags("terminated") | _scripted_flags("truncated")
    decisions = _scripted_decisions()

    advantages, returns = compute_gae(
        rewards,
        _episode_starts(dones),
        values,
        final_values,
        last_values,
        dones[-1],
        GAMMA,
        GAE_LAMBDA,
        step_lambda=_step_lambda(decisions, GAE_LAMBDA),
    )

    ref_adv, ref_ret = _smdp_reference(
        rewards, values, final_values, dones, decisions, last_values, GAMMA, GAE_LAMBDA
    )
    assert torch.allclose(advantages.double(), ref_adv, atol=1e-5)
    assert torch.allclose(returns.double(), ref_ret, atol=1e-5)

    # Non-decision step 1 of env 3 (hold 0-4): remaining hold rewards plus
    # gamma^n ((1 - lambda) V_d + lambda (V_d + A_d)) of the next decision d=5.
    d = 5
    hold_rewards = sum(GAMMA**k * rewards[1 + k, 3] for k in range(d - 1))
    tail = (1 - GAE_LAMBDA) * values[d, 3] + GAE_LAMBDA * (values[d, 3] + advantages[d, 3])
    assert torch.allclose(returns[1, 3], hold_rewards + GAMMA ** (d - 1) * tail, atol=1e-5)
    # Hold 10-14 of env 3 runs past the window: bootstrap from last_values.
    expected = rewards[10, 3] + GAMMA * rewards[11, 3] + GAMMA**2 * last_values[3]
    assert torch.allclose(returns[10, 3], expected, atol=1e-5)


def test_compute_gae_step_lambda_all_lambda_matches_scalar_path():
    generator = torch.Generator().manual_seed(1)
    rewards = torch.randn(7, 3, generator=generator)
    values = torch.randn(7, 3, generator=generator)
    final_values = torch.randn(7, 3, generator=generator)
    starts = (torch.rand(7, 3, generator=generator) < 0.3).float()
    last_values = torch.randn(3, generator=generator)
    last_dones = torch.tensor([0.0, 1.0, 0.0])
    args = (rewards, starts, values, final_values, last_values, last_dones, GAMMA, GAE_LAMBDA)

    adv_scalar, ret_scalar = compute_gae(*args)
    adv_tensor, ret_tensor = compute_gae(*args, step_lambda=torch.full_like(rewards, GAE_LAMBDA))

    assert torch.allclose(adv_tensor, adv_scalar, atol=1e-6)
    assert torch.allclose(ret_tensor, ret_scalar, atol=1e-6)


def _scalar_gae_loop(rewards, starts, values, final_values, last_values, last_dones, gamma, gae_lambda):
    """The scalar-lambda recurrence, expression by expression."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros_like(last_values)
    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            next_not_done = 1.0 - last_dones
            next_values = last_values
        else:
            next_not_done = 1.0 - starts[step + 1]
            next_values = values[step + 1]
        real_next_values = next_not_done * next_values + final_values[step]
        delta = rewards[step] + gamma * real_next_values - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_not_done * last_gae_lam
        advantages[step] = last_gae_lam
    return advantages, advantages + values


def test_compute_gae_without_step_lambda_is_bitwise_unchanged():
    generator = torch.Generator().manual_seed(2)
    rewards = torch.randn(9, 5, generator=generator)
    values = torch.randn(9, 5, generator=generator)
    final_values = torch.randn(9, 5, generator=generator)
    starts = (torch.rand(9, 5, generator=generator) < 0.3).float()
    last_values = torch.randn(5, generator=generator)
    last_dones = (torch.rand(5, generator=generator) < 0.5).float()
    args = (rewards, starts, values, final_values, last_values, last_dones, 0.99, 0.95)

    advantages, returns = compute_gae(*args)
    ref_adv, ref_ret = _scalar_gae_loop(*args)

    assert torch.equal(advantages, ref_adv)
    assert torch.equal(returns, ref_ret)


def test_compute_gae_without_step_lambda_matches_hardcoded_values():
    # Dyadic inputs: every intermediate value is exact in float32.
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    values = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    final_values = torch.tensor([[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    starts = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

    advantages, returns = compute_gae(
        rewards, starts, values, final_values, torch.tensor([2.0, 4.0]), torch.zeros(2), 0.5, 0.5
    )

    assert torch.equal(advantages, torch.tensor([[1.625, 3.75], [2.5, 5.0], [6.0, 6.0]]))
    assert torch.equal(returns, torch.tensor([[2.625, 3.75], [4.5, 6.0], [6.0, 8.0]]))


def test_compute_gae_rejects_step_lambda_of_wrong_shape():
    rewards = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="step_lambda"):
        compute_gae(
            rewards, rewards, rewards, rewards, torch.zeros(2), torch.zeros(2), 0.9, 0.9,
            step_lambda=torch.zeros(3),
        )


# ---------------------------------------------------------------------------
# Rollout buffers
# ---------------------------------------------------------------------------


def _buffer(buffer_class=RolloutBuffer, *, store_decisions: bool = True, num_steps: int = NUM_STEPS):
    return buffer_class(
        spaces.Box(-1e6, 1e6, shape=(2,)),
        spaces.Box(-1.0, 1.0, shape=(2,)),
        num_steps=num_steps,
        num_envs=NUM_ENVS,
        device="cpu",
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        store_decisions=store_decisions,
    )


def _fill(buffer, rewards, dones, values, final_values, decisions=None) -> None:
    """Obs row ``(t, n)`` holds ``(t * N + n, n)`` so samples reveal their indices."""
    for step in range(buffer.num_steps):
        flat_index = step * NUM_ENVS + torch.arange(NUM_ENVS, dtype=torch.float32)
        obs = torch.stack([flat_index, torch.arange(NUM_ENVS, dtype=torch.float32)], dim=-1)
        buffer.add(
            obs,
            torch.zeros(NUM_ENVS, 2),
            rewards[step],
            dones[step],
            values[step],
            torch.zeros(NUM_ENVS),
            final_values=final_values[step],
            decision=None if decisions is None else decisions[step],
        )


def _random_window(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(NUM_STEPS, NUM_ENVS, generator=generator)
    values = torch.randn(NUM_STEPS, NUM_ENVS, generator=generator)
    final_values = torch.zeros(NUM_STEPS, NUM_ENVS)
    final_values[4, 1] = 0.5
    dones = (_scripted_flags("terminated") | _scripted_flags("truncated")).float()
    last_values = torch.randn(NUM_ENVS, generator=generator)
    return rewards, values, final_values, dones, last_values


def test_buffer_decisions_use_step_lambda_gae():
    rewards, values, final_values, dones, last_values = _random_window()
    decisions = _scripted_decisions()
    buffer = _buffer()
    _fill(buffer, rewards, dones, values, final_values, decisions)

    assert buffer.window_has_decisions
    assert torch.equal(buffer.decisions, decisions)
    buffer.compute_returns_and_advantage(last_values, dones[-1])

    expected_adv, expected_ret = compute_gae(
        rewards,
        _episode_starts(dones),
        values,
        final_values,
        last_values,
        dones[-1],
        GAMMA,
        GAE_LAMBDA,
        step_lambda=_step_lambda(decisions, GAE_LAMBDA),
    )
    assert torch.equal(buffer.advantages, expected_adv)
    assert torch.equal(buffer.returns, expected_ret)


def test_buffer_rows_without_decision_count_as_decisions_and_keep_plain_gae():
    rewards, values, final_values, dones, last_values = _random_window()
    with_store = _buffer(store_decisions=True)
    without_store = _buffer(store_decisions=False)
    _fill(with_store, rewards, dones, values, final_values)
    _fill(without_store, rewards, dones, values, final_values)

    assert bool(with_store.decisions.all())
    assert not with_store.window_has_decisions
    with_store.compute_returns_and_advantage(last_values, dones[-1])
    without_store.compute_returns_and_advantage(last_values, dones[-1])
    assert torch.equal(with_store.advantages, without_store.advantages)
    assert torch.equal(with_store.returns, without_store.returns)
    assert all(sample.decision_mask is None for sample in with_store.get(8))


def test_buffer_reset_restores_all_true_decisions():
    rewards, values, final_values, dones, _ = _random_window()
    buffer = _buffer()
    _fill(buffer, rewards, dones, values, final_values, _scripted_decisions())
    buffer.reset()
    assert bool(buffer.decisions.all())
    assert not buffer.window_has_decisions


def test_buffer_rejects_decisions_on_only_some_steps():
    rewards, values, final_values, dones, last_values = _random_window()
    decisions = _scripted_decisions()
    buffer = _buffer()
    for step in range(NUM_STEPS):
        buffer.add(
            torch.zeros(NUM_ENVS, 2),
            torch.zeros(NUM_ENVS, 2),
            rewards[step],
            dones[step],
            values[step],
            torch.zeros(NUM_ENVS),
            final_values=final_values[step],
            decision=decisions[step] if step != 5 else None,
        )
    with pytest.raises(ValueError, match="every step"):
        buffer.compute_returns_and_advantage(last_values, dones[-1])


def test_buffer_rejects_decisions_with_finite_horizon_gae():
    rewards, values, final_values, dones, last_values = _random_window()
    buffer = _buffer()
    _fill(buffer, rewards, dones, values, final_values, _scripted_decisions())
    with pytest.raises(ValueError, match="finite_horizon_gae"):
        buffer.compute_returns_and_advantage(last_values, dones[-1], finite_horizon_gae=True)


def test_buffer_without_decision_storage_rejects_a_decision():
    buffer = _buffer(store_decisions=False)
    with pytest.raises(ValueError, match="store_decisions"):
        buffer.add(
            torch.zeros(NUM_ENVS, 2),
            torch.zeros(NUM_ENVS, 2),
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            decision=torch.ones(NUM_ENVS, dtype=torch.bool),
        )


def test_buffer_get_yields_decision_mask_of_sampled_rows():
    rewards, values, final_values, dones, last_values = _random_window()
    buffer = _buffer()
    _fill(buffer, rewards, dones, values, final_values, _scripted_decisions())
    buffer.compute_returns_and_advantage(last_values, dones[-1])

    flat_decisions = buffer.decisions.reshape(-1)
    seen = 0
    for sample in buffer.get(10):
        mb_inds = sample.obs[:, 0].long()
        assert sample.decision_mask.dtype == torch.bool
        assert torch.equal(sample.decision_mask, flat_decisions[mb_inds])
        seen += mb_inds.numel()
    assert seen == buffer.buffer_size


def test_recurrent_buffer_sequences_carry_decision_mask():
    rewards, values, final_values, dones, last_values = _random_window()
    buffer = _buffer(RecurrentRolloutBuffer)
    _fill(buffer, rewards, dones, values, final_values, _scripted_decisions())
    buffer.compute_returns_and_advantage(last_values, dones[-1])

    initial_hidden = torch.zeros(1, NUM_ENVS, 3)
    generator = torch.Generator().manual_seed(0)
    for sample in buffer.get_sequences(2, initial_hidden, generator=generator):
        idx = sample.obs[0, :, 1].long()
        assert sample.decision_mask.shape == (NUM_STEPS, 2)
        assert torch.equal(sample.decision_mask, buffer.decisions[:, idx])


def test_recurrent_buffer_without_decisions_yields_no_mask():
    rewards, values, final_values, dones, last_values = _random_window()
    buffer = _buffer(RecurrentRolloutBuffer, store_decisions=False)
    _fill(buffer, rewards, dones, values, final_values)
    buffer.compute_returns_and_advantage(last_values, dones[-1])
    samples = list(buffer.get_sequences(2, torch.zeros(1, NUM_ENVS, 3)))
    assert all(sample.decision_mask is None for sample in samples)


# ---------------------------------------------------------------------------
# (b) toy env through the real rollout loop
# ---------------------------------------------------------------------------


def _ppo_kwargs(**overrides) -> dict:
    kwargs = {
        "device": "cpu",
        "num_steps": NUM_STEPS,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "std_log": False,
        "target_kl": None,
        "net_arch": [8],
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
    }
    kwargs.update(overrides)
    return kwargs


def _skip_update() -> dict[str, float]:
    return {}


def _state_sum_values(obs, *args, **kwargs):
    del args, kwargs
    return obs["state"].sum(dim=-1, keepdim=True)


def _one_window(agent) -> None:
    agent.train = _skip_update
    agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)


def _assert_returns_match_reference(agent, env: _FixedHoldVecEnv) -> None:
    buffer = agent.rollout_buffer
    ref_adv, ref_ret = _smdp_reference(
        buffer.rewards,
        buffer.values,
        buffer.final_values,
        buffer.dones.bool(),
        buffer.decisions,
        env.last_state.sum(dim=-1),
        GAMMA,
        GAE_LAMBDA,
    )
    assert torch.allclose(buffer.advantages.double(), ref_adv, atol=1e-5)
    assert torch.allclose(buffer.returns.double(), ref_ret, atol=1e-5)


def test_ppo_rollout_stores_decisions_and_computes_smdp_returns():
    env = _FixedHoldVecEnv()
    agent = PPO(env, **_ppo_kwargs(decision_level=True))
    agent.policy.predict_values = _state_sum_values
    _one_window(agent)

    buffer = agent.rollout_buffer
    assert torch.equal(buffer.decisions, _scripted_decisions())
    # The truncation in the middle of env 1's hold bootstraps from the
    # terminal observation (state + 1.0 per column; 5 steps since the
    # episode start, env index 1, 5 env steps taken).
    assert buffer.final_values[4, 1] != 0
    expected_final = (torch.tensor([5.0, 1.0, 5.0, 1.0]) / 10.0 + 1.0).sum()
    assert torch.allclose(buffer.final_values[4, 1], expected_final)
    _assert_returns_match_reference(agent, env)


def test_recurrent_ppo_rollout_stores_decisions_and_computes_smdp_returns():
    env = _FixedHoldVecEnv()
    agent = RecurrentPPO(env, **_ppo_kwargs(decision_level=True), rnn_hidden_size=8)
    agent._predict_last_values = lambda obs, hidden: obs["state"].sum(dim=-1)
    _one_window(agent)

    assert torch.equal(agent.rollout_buffer.decisions, _scripted_decisions())
    _assert_returns_match_reference(agent, env)


# ---------------------------------------------------------------------------
# (e) default path unchanged
# ---------------------------------------------------------------------------


def _run_window(env: _FixedHoldVecEnv, **overrides) -> PPO:
    agent = PPO(env, **_ppo_kwargs(**overrides))
    _one_window(agent)
    return agent


def test_decision_level_without_env_key_matches_feature_off():
    on = _run_window(_FixedHoldVecEnv(emit_decisions=False), decision_level=True)
    off = _run_window(_FixedHoldVecEnv(emit_decisions=False), decision_level=False)

    assert not on.rollout_buffer.window_has_decisions
    assert torch.equal(on.rollout_buffer.returns, off.rollout_buffer.returns)
    assert torch.equal(on.rollout_buffer.advantages, off.rollout_buffer.advantages)
    assert all(sample.decision_mask is None for sample in on.rollout_buffer.get())


def _ignored_key_warnings(caught) -> list:
    return [w for w in caught if "decision-level training is off" in str(w.message)]


def test_decision_keys_with_feature_off_warn_once_and_are_ignored():
    silent = _run_window(_FixedHoldVecEnv(emit_decisions=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitting = _run_window(_FixedHoldVecEnv(emit_decisions=True))

    ignored = _ignored_key_warnings(caught)
    assert len(ignored) == 1
    assert issubclass(ignored[0].category, UserWarning)
    assert torch.equal(emitting.rollout_buffer.returns, silent.rollout_buffer.returns)
    assert torch.equal(emitting.rollout_buffer.advantages, silent.rollout_buffer.advantages)


def test_ignored_decision_key_warning_is_per_agent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = _run_window(_FixedHoldVecEnv(emit_decisions=True))
        _run_window(_FixedHoldVecEnv(emit_decisions=True))
        first.train = _skip_update
        first.learn(total_timesteps=2 * NUM_ENVS * NUM_STEPS)

    assert len(_ignored_key_warnings(caught)) == 2


def test_decision_level_metadata_and_adaptive_kl_kwargs_without_infos():
    agent = PPO(_FixedHoldVecEnv(), **_ppo_kwargs(decision_level=True))
    assert agent._checkpoint_metadata()["decision_level"] is True
    assert agent._extra_rollout_buffer_kwargs() == {}
    assert PPO(_FixedHoldVecEnv(), **_ppo_kwargs())._checkpoint_metadata()["decision_level"] is False


# ---------------------------------------------------------------------------
# (h) contract
# ---------------------------------------------------------------------------


def test_reader_rejects_key_appearing_on_only_some_steps():
    reader = DecisionMaskReader(2, (3,), "cpu")
    reader.read({"decision": torch.ones(2, dtype=torch.bool)})
    with pytest.raises(ValueError, match="key set changed at step 1"):
        reader.read({})

    reader = DecisionMaskReader(2, (3,), "cpu")
    assert reader.read({}) is None
    with pytest.raises(ValueError, match="key set changed"):
        reader.read({"decision": torch.ones(2, dtype=torch.bool)})


def test_learn_rejects_key_missing_on_one_step():
    agent = PPO(_FixedHoldVecEnv(drop_key_at_step=3), **_ppo_kwargs(decision_level=True))
    agent.train = _skip_update
    with pytest.raises(ValueError, match="key set changed at step 3"):
        agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)


def test_reader_rejects_both_keys_on_one_step():
    reader = DecisionMaskReader(2, (3,), "cpu")
    infos = {
        "decision": torch.ones(2, dtype=torch.bool),
        "decision_components": torch.ones(2, 3, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="both"):
        reader.read(infos)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("decision", torch.ones(2, dtype=torch.float32), "dtype"),
        ("decision", torch.ones(2, 1, dtype=torch.bool), "shape"),
        ("decision", torch.ones(3, dtype=torch.bool), "shape"),
        ("decision", np.ones(2, dtype=bool), "torch.Tensor"),
        ("decision_components", torch.ones(2, dtype=torch.bool), "shape"),
        ("decision_components", torch.ones(2, 3, dtype=torch.int64), "dtype"),
    ],
)
def test_reader_rejects_bad_dtype_or_shape(key, value, match):
    reader = DecisionMaskReader(2, (3,), "cpu")
    with pytest.raises(ValueError, match=match):
        reader.read({key: value})


def test_reader_derives_decision_from_components():
    reader = DecisionMaskReader(3, (2,), "cpu")
    components = torch.tensor([[True, False], [False, False], [False, True]])
    decision, component_mask = reader.read({"decision_components": components})
    assert decision.dtype == torch.bool
    assert torch.equal(decision, torch.tensor([True, False, True]))
    assert torch.equal(component_mask, components)


@pytest.mark.parametrize("key", ["decision", "decision_components"])
def test_chunk_accumulator_rejects_decision_keys(key):
    accumulator = ChunkAccumulator()
    zeros = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="no defined decision flag"):
        accumulator.add(torch.zeros(2), zeros, zeros, {key: torch.ones(2, dtype=torch.bool)})


def test_action_repeat_wrapper_rejects_decision_key():
    env = ActionRepeatWrapper(_FixedHoldVecEnv(), repeat=2)
    env.reset()
    with pytest.raises(ValueError, match="no defined decision flag"):
        env.step(torch.zeros(NUM_ENVS, 2))


def test_action_chunk_wrapper_rejects_decision_key():
    env = ActionChunkWrapper(_FixedHoldVecEnv(), act_steps=2)
    env.reset()
    with pytest.raises(ValueError, match="no defined decision flag"):
        env.step(torch.zeros(NUM_ENVS, 2, 2))


class _DecisionFrameEnv(_FakeBatchedEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_decision = torch.tensor([True, False])
        return obs, reward, terminated, truncated, {**info, "decision": self.last_decision}


@pytest.mark.parametrize("autoreset_in_step", [False, True])
def test_image_frame_stack_forwards_decision_key(autoreset_in_step):
    inner = _DecisionFrameEnv()
    env = ImageFrameStackWrapper(inner, frame_stack=3, autoreset_in_step=autoreset_in_step)
    env.reset()
    *_, info = env.step(torch.zeros(2, 1))
    assert info["decision"] is inner.last_decision


class _DecisionTinyEnv(_TinyEnv):
    """``_TinyEnv`` that reports ``decision`` (or ``decision_components``) on
    the steps listed in ``decision_steps``; ``None`` reports no key."""

    def __init__(self, terminate_at, decision_steps, *, components: bool = False):
        super().__init__(terminate_at=terminate_at)
        self.decision_steps = decision_steps
        self.components = components

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.decision_steps is not None:
            flag = self._t in self.decision_steps
            if self.components:
                info = {**info, "decision_components": np.array([flag])}
            else:
                info = {**info, "decision": flag}
        return obs, reward, terminated, truncated, info


def _decision_adapter(*envs) -> TorchVectorEnvAdapter:
    vec = SyncVectorEnv(
        [lambda env=env: env for env in envs], autoreset_mode=AutoresetMode.SAME_STEP
    )
    adapter = TorchVectorEnvAdapter(vec, device="cpu")
    adapter.reset(seed=0)
    return adapter


def test_vector_env_adapter_converts_decision_to_torch_bool():
    adapter = _decision_adapter(
        _DecisionTinyEnv(terminate_at=3, decision_steps={1, 3}),
        _DecisionTinyEnv(terminate_at=100, decision_steps={1, 2}),
    )
    action = torch.zeros(2, 1)
    expected = [[True, True], [False, True], [True, False]]
    for step_expected in expected:
        *_, terminated, _, infos = adapter.step(action)
        decision = infos["decision"]
        assert isinstance(decision, torch.Tensor)
        assert decision.dtype == torch.bool and decision.device.type == "cpu"
        assert decision.tolist() == step_expected
        assert "_decision" not in infos
        decision, component_mask = DecisionMaskReader(2, (1,), "cpu").read(infos)
        assert decision.tolist() == step_expected
        assert component_mask is None
    # Env 0 ended on the last step: its flag came from that step's info,
    # which gymnasium moved into final_info.
    assert terminated.tolist() == [True, False]


def test_vector_env_adapter_converts_decision_components():
    adapter = _decision_adapter(
        _DecisionTinyEnv(terminate_at=100, decision_steps={1}, components=True),
        _DecisionTinyEnv(terminate_at=100, decision_steps={2}, components=True),
    )
    *_, infos = adapter.step(torch.zeros(2, 1))
    components = infos["decision_components"]
    assert components.dtype == torch.bool
    assert components.tolist() == [[True], [False]]
    assert "_decision_components" not in infos


def test_vector_env_adapter_rejects_partially_reported_decision():
    adapter = _decision_adapter(
        _DecisionTinyEnv(terminate_at=100, decision_steps={1}),
        _DecisionTinyEnv(terminate_at=100, decision_steps=None),
    )
    with pytest.raises(ValueError, match="Env 1 did not report"):
        adapter.step(torch.zeros(2, 1))


def test_gail_rejects_decision_level():
    with pytest.raises(ValueError, match="GAIL does not support decision_level"):
        GAIL(
            _FixedHoldVecEnv(),
            demo_env_id="unused",
            **_ppo_kwargs(decision_level=True),
        )


def test_ppo_rejects_decision_level_with_finite_horizon_gae():
    with pytest.raises(ValueError, match="finite_horizon_gae"):
        PPO(_FixedHoldVecEnv(), **_ppo_kwargs(decision_level=True, finite_horizon_gae=True))
