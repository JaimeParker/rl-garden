"""``PPO._build_policy`` and the policy-level hidden-state interface used by
``SequencePPO``: pure seams, so construction and training stay bit-identical."""

from __future__ import annotations

import pytest
import torch
from _decision_fakes import NUM_ENVS, NUM_STEPS, _FixedHoldVecEnv

from rl_garden.algorithms import PPO, RecurrentPPO, TransformerPPO
from rl_garden.policies.ppo_policy import PPOPolicy
from rl_garden.policies.recurrent_ppo_policy import RecurrentPPOPolicy

TRANSFORMER_KWARGS = {
    "embed_dim": 8,
    "head_dim": 4,
    "num_heads": 2,
    "num_transformer_layers": 1,
    "mlp_num": 1,
    "memory_len": 2,
}
AGENT_CASES = [
    pytest.param(PPO, {}, PPOPolicy, id="ppo"),
    pytest.param(RecurrentPPO, {"rnn_hidden_size": 8}, RecurrentPPOPolicy, id="gru"),
    pytest.param(
        RecurrentPPO, {"rnn_hidden_size": 8, "rnn_type": "lstm"}, RecurrentPPOPolicy, id="lstm"
    ),
    pytest.param(TransformerPPO, TRANSFORMER_KWARGS, RecurrentPPOPolicy, id="gtrxl"),
]
STATEFUL_CASES = [case for case in AGENT_CASES if case.id != "ppo"]


def _make_agent(agent_cls, extra_kwargs, *, seed: int = 0):
    return agent_cls(
        _FixedHoldVecEnv(emit_decisions=False),
        device="cpu",
        num_steps=NUM_STEPS,
        num_minibatches=2,
        update_epochs=2,
        eval_freq=0,
        log_freq=0,
        std_log=False,
        target_kl=None,
        net_arch=[8],
        seed=seed,
        **extra_kwargs,
    )


def _assert_state_equal(actual, expected) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for actual_part, expected_part in zip(actual, expected):
            assert torch.equal(actual_part, expected_part)
    else:
        assert torch.equal(actual, expected)


def _assert_state_dicts_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name


@pytest.mark.parametrize("agent_cls,extra_kwargs,policy_cls", AGENT_CASES)
def test_build_policy_returns_the_expected_policy_class(agent_cls, extra_kwargs, policy_cls):
    agent = _make_agent(agent_cls, extra_kwargs)
    built = agent._build_policy(agent.env.single_observation_space)
    assert type(built) is policy_cls
    assert type(agent.policy) is policy_cls


@pytest.mark.parametrize("agent_cls,extra_kwargs,policy_cls", AGENT_CASES)
def test_is_stateful_matches_policy_class(agent_cls, extra_kwargs, policy_cls):
    agent = _make_agent(agent_cls, extra_kwargs)
    assert agent.policy.is_stateful is (policy_cls is RecurrentPPOPolicy)


@pytest.mark.parametrize("agent_cls,extra_kwargs,policy_cls", STATEFUL_CASES)
def test_hidden_state_methods_delegate_to_the_encoder(agent_cls, extra_kwargs, policy_cls):
    del policy_cls
    policy = _make_agent(agent_cls, extra_kwargs).policy
    encoder = policy.recurrent_encoder

    initial = policy.initial_hidden_state(NUM_ENVS, torch.device("cpu"))
    _assert_state_equal(initial, encoder.get_initial_state(NUM_ENVS, torch.device("cpu")))

    generator = torch.Generator().manual_seed(0)
    state = tuple(torch.randn(part.shape, generator=generator) for part in initial) if isinstance(
        initial, tuple
    ) else torch.randn(initial.shape, generator=generator)
    keep_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    _assert_state_equal(
        policy.mask_hidden_state(state, keep_mask), encoder.mask_state(state, keep_mask)
    )
    done_mask = torch.tensor([False, True, True, False])
    _assert_state_equal(
        policy.index_hidden_state(state, done_mask), encoder.index_state(state, done_mask)
    )
    indices = torch.tensor([3, 0])
    _assert_state_equal(
        policy.index_hidden_state(state, indices), encoder.index_state(state, indices)
    )


@pytest.mark.parametrize("agent_cls,extra_kwargs,policy_cls", AGENT_CASES)
def test_seeded_construction_is_deterministic(agent_cls, extra_kwargs, policy_cls):
    del policy_cls
    first = _make_agent(agent_cls, extra_kwargs).policy.state_dict()
    second = _make_agent(agent_cls, extra_kwargs).policy.state_dict()
    _assert_state_dicts_equal(second, first)


@pytest.mark.parametrize("agent_cls,extra_kwargs,policy_cls", AGENT_CASES)
def test_seeded_learn_window_is_deterministic(agent_cls, extra_kwargs, policy_cls):
    del policy_cls

    def trained_parameters() -> dict:
        agent = _make_agent(agent_cls, extra_kwargs)
        agent.learn(total_timesteps=agent.num_envs * agent.num_steps)
        return agent.policy.state_dict()

    _assert_state_dicts_equal(trained_parameters(), trained_parameters())
