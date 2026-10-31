"""Decision-level PPO with partially effective (multi-rate) action components:
``infos["decision_components"]`` in the rollout, the rollout buffers and the
PPO losses. A component that did not take effect on a step contributes
nothing to that step's log-prob, log-ratio, entropy or KL, so it gets no
policy gradient from it."""

from __future__ import annotations

import dataclasses
import math
import warnings

import pytest
import torch
from _decision_fakes import (
    COMPONENT_CYCLES_WITHOUT_LAST,
    NUM_COMPONENTS,
    NUM_ENVS,
    NUM_STEPS,
    _ComponentMaskVecEnv,
    _scripted_components,
    _scripted_flags,
)
from gymnasium import spaces
from test_frame_stack_wrapper import _FakeBatchedEnv
from test_rollout_episode_metrics import _RecordingLogger

import rl_garden.algorithms.ppo as ppo_module
from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.algorithms._decision_mask import DecisionMaskReader
from rl_garden.buffers import RolloutBuffer
from rl_garden.buffers.gae import compute_gae
from rl_garden.buffers.recurrent_rollout_buffer import RecurrentRolloutBuffer
from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.envs.wrappers import ImageFrameStackWrapper
from rl_garden.networks.actor_critic import gaussian_kl_divergence

GAMMA = 0.9
GAE_LAMBDA = 0.8
RECURRENT_KWARGS = {"rnn_hidden_size": 8}
AGENT_CLASSES = [(PPO, {}), (RecurrentPPO, RECURRENT_KWARGS)]
LR_SCHEDULES = ["constant", "adaptive_kl"]


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


def _collect_windows(agent, num_windows: int = 1) -> None:
    """Real rollouts, no update: the buffer keeps the last window."""
    agent.train = _skip_update
    agent.learn(total_timesteps=num_windows * agent.num_envs * agent.num_steps)


def _disable_optimizer_step(monkeypatch, agent) -> None:
    """Keep the parameters fixed; gradients stay readable after the update."""
    monkeypatch.setattr(agent.policy_optimizer, "step", lambda *args, **kwargs: None)


def _minibatch_update(agent, data) -> tuple[dict, dict[str, torch.Tensor | None], float]:
    """One ``_ppo_minibatch_update`` on ``data``; returns its result, the
    parameter gradients and the learning rate it set (restored afterwards)."""
    (
        values, log_prob, entropy, old_values, old_log_prob, advantages, returns,
        old_mean, old_log_std, new_mean, new_log_std,
    ) = agent._evaluate_minibatch(data)
    learning_rates = [group["lr"] for group in agent.policy_optimizer.param_groups]
    result = agent._ppo_minibatch_update(
        values=values,
        log_prob=log_prob,
        entropy=entropy,
        old_values=old_values,
        old_log_prob=old_log_prob,
        advantages=advantages,
        returns=returns,
        clip_coef=agent.clip_coef,
        old_mean=old_mean,
        old_log_std=old_log_std,
        new_mean=new_mean,
        new_log_std=new_log_std,
        **agent._minibatch_masks(data),
    )
    learning_rate = agent.policy_optimizer.param_groups[0]["lr"]
    for group, lr in zip(agent.policy_optimizer.param_groups, learning_rates):
        group["lr"] = lr
    grads = {
        name: None if p.grad is None else p.grad.clone()
        for name, p in agent.policy.named_parameters()
    }
    return result, grads, learning_rate


_LOSS_KEYS = (
    "loss", "policy_loss", "value_loss", "entropy_loss", "approx_kl", "old_approx_kl", "clipfrac",
)


def _assert_same_update(first, second) -> None:
    (result_a, grads_a, lr_a), (result_b, grads_b, lr_b) = first, second
    for key in _LOSS_KEYS:
        assert math.isfinite(result_a[key]), key
        assert result_a[key] == result_b[key], key
    assert torch.equal(result_a["decisions"], result_b["decisions"])
    assert lr_a == lr_b
    assert grads_a.keys() == grads_b.keys()
    nonzero = 0
    for name, grad in grads_a.items():
        assert (grad is None) == (grads_b[name] is None), name
        if grad is None:
            continue
        assert torch.isfinite(grad).all(), name
        assert torch.equal(grad, grads_b[name]), name
        nonzero += int(grad.abs().sum() > 0)
    assert nonzero > 0


def _assert_state_dicts_equal(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for name, tensor in first.items():
        assert torch.equal(second[name], tensor), name


# ---------------------------------------------------------------------------
# sum_dims on the policy and KL helpers
# ---------------------------------------------------------------------------


def test_gaussian_kl_divergence_per_component():
    generator = torch.Generator().manual_seed(0)
    old_mean, old_log_std, new_mean, new_log_std = (
        torch.randn(5, NUM_COMPONENTS, generator=generator) for _ in range(4)
    )
    summed = gaussian_kl_divergence(old_mean, old_log_std, new_mean, new_log_std)
    per_component = gaussian_kl_divergence(
        old_mean, old_log_std, new_mean, new_log_std, sum_dims=False
    )
    assert per_component.shape == (5, NUM_COMPONENTS)
    assert torch.equal(per_component.sum(-1), summed)


def test_policy_dist_param_methods_keep_components_with_sum_dims_false():
    agent = PPO(_ComponentMaskVecEnv(), **_ppo_kwargs(lr_schedule="adaptive_kl"))
    obs = {"state": torch.randn(NUM_ENVS, 4)}
    torch.manual_seed(1)
    summed = agent.policy.act_with_value_logprob_and_dist_params(obs)
    torch.manual_seed(1)
    per_component = agent.policy.act_with_value_logprob_and_dist_params(obs, sum_dims=False)
    assert per_component[2].shape == (NUM_ENVS, NUM_COMPONENTS)
    assert torch.equal(per_component[2].sum(-1, keepdim=True), summed[2])
    assert torch.equal(per_component[3].sum(-1, keepdim=True), summed[3])
    for index in (0, 1, 4, 5):
        assert torch.equal(per_component[index], summed[index])

    actions = summed[0]
    summed = agent.policy.evaluate_actions_with_dist_params(obs, actions)
    per_component = agent.policy.evaluate_actions_with_dist_params(obs, actions, sum_dims=False)
    assert per_component[1].shape == (NUM_ENVS, NUM_COMPONENTS)
    assert torch.equal(per_component[1].sum(-1, keepdim=True), summed[1])
    assert torch.equal(per_component[2].sum(-1, keepdim=True), summed[2])


def test_recurrent_policy_sequence_methods_keep_components_with_sum_dims_false():
    agent = RecurrentPPO(_ComponentMaskVecEnv("none"), **_ppo_kwargs(), **RECURRENT_KWARGS)
    _collect_windows(agent)
    data = next(agent.rollout_buffer.get_sequences(1, agent._rollout_initial_hidden, shuffle=False))
    args = (data.obs, data.actions, data.initial_hidden, data.episode_starts)
    policy = agent.policy
    methods = (
        policy.evaluate_actions_sequence,
        policy.evaluate_actions_sequence_with_dist_params,
    )
    for method in methods:
        summed = method(*args)
        per_component = method(*args, sum_dims=False)
        assert per_component[1].shape == (NUM_STEPS, NUM_ENVS, NUM_COMPONENTS)
        assert torch.equal(per_component[1].sum(-1, keepdim=True), summed[1])
        assert torch.equal(per_component[2].sum(-1, keepdim=True), summed[2])


# ---------------------------------------------------------------------------
# Rollout: per-component log-probs, summed exactly like the actor does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lr_schedule", LR_SCHEDULES)
@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_decision_level_rollout_sums_components_bitwise(agent_cls, extra_kwargs, lr_schedule):
    on = agent_cls(
        _ComponentMaskVecEnv(),
        **_ppo_kwargs(decision_level=True, lr_schedule=lr_schedule),
        **extra_kwargs,
    )
    off = agent_cls(_ComponentMaskVecEnv(), **_ppo_kwargs(lr_schedule=lr_schedule), **extra_kwargs)
    off.policy.load_state_dict(on.policy.state_dict())
    obs, _ = on.env.reset()
    hidden = on._initial_hidden_state(NUM_ENVS)
    episode_starts = torch.zeros(NUM_ENVS)

    torch.manual_seed(5)
    rollout_on = on._rollout_step(obs, hidden, episode_starts)
    torch.manual_seed(5)
    rollout_off = off._rollout_step(obs, hidden, episode_starts)

    for value_on, value_off in zip(rollout_on[:4], rollout_off[:4]):
        assert torch.equal(value_on, value_off)
    component_log_probs = on._rollout_component_log_probs
    assert component_log_probs.shape == (NUM_ENVS, NUM_COMPONENTS)
    assert torch.equal(component_log_probs.sum(-1, keepdim=True), rollout_on[2])
    if lr_schedule == "adaptive_kl":
        assert torch.equal(on._rollout_mean, off._rollout_mean)
        assert torch.equal(on._rollout_log_std, off._rollout_log_std)


def test_rollout_stores_component_mask_and_log_probs():
    agent = PPO(_ComponentMaskVecEnv(), **_ppo_kwargs(decision_level=True))
    _collect_windows(agent)
    buffer = agent.rollout_buffer
    components = _scripted_components()

    assert buffer.window_has_components
    assert torch.equal(buffer.component_mask, components)
    assert torch.equal(buffer.decisions, components.any(-1))
    with torch.no_grad():
        _, log_prob, _ = agent.policy.evaluate_actions(
            flatten_leading_dims(buffer.obs),
            buffer.actions.reshape(-1, NUM_COMPONENTS),
            sum_dims=False,
        )
    assert torch.allclose(
        buffer.component_log_probs, log_prob.reshape(NUM_STEPS, NUM_ENVS, -1), atol=1e-5
    )
    # The stored behaviour log-prob covers the components that took effect.
    assert (buffer.log_probs[~buffer.decisions] == 0).all()
    expected = torch.where(components, buffer.component_log_probs, 0.0).sum(-1)
    assert torch.allclose(buffer.log_probs, expected)


# ---------------------------------------------------------------------------
# Rollout buffers
# ---------------------------------------------------------------------------


def _component_buffer(buffer_class=RolloutBuffer, *, store_decisions: bool = True):
    return buffer_class(
        spaces.Box(-1e6, 1e6, shape=(2,)),
        spaces.Box(-1.0, 1.0, shape=(NUM_COMPONENTS,)),
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
        device="cpu",
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        store_decisions=store_decisions,
    )


def _random_window(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(NUM_STEPS, NUM_ENVS, generator=generator)
    values = torch.randn(NUM_STEPS, NUM_ENVS, generator=generator)
    final_values = torch.zeros(NUM_STEPS, NUM_ENVS)
    final_values[4, 1] = 0.5
    dones = (_scripted_flags("terminated") | _scripted_flags("truncated")).float()
    last_values = torch.randn(NUM_ENVS, generator=generator)
    component_log_probs = torch.randn(NUM_STEPS, NUM_ENVS, NUM_COMPONENTS, generator=generator)
    return rewards, values, final_values, dones, last_values, component_log_probs


def _fill(buffer, window, *, components=None, decisions=None, component_steps=None) -> None:
    """Obs row ``(t, n)`` holds ``(t * N + n, n)`` so samples reveal their
    indices. ``components`` passes component masks (on ``component_steps``,
    default every step); ``decisions`` passes whole-step decisions only."""
    rewards, values, final_values, dones, _, component_log_probs = window
    for step in range(NUM_STEPS):
        flat_index = step * NUM_ENVS + torch.arange(NUM_ENVS, dtype=torch.float32)
        obs = torch.stack([flat_index, torch.arange(NUM_ENVS, dtype=torch.float32)], dim=-1)
        kwargs = {}
        if components is not None and (component_steps is None or step in component_steps):
            kwargs = {
                "decision": components[step].any(-1),
                "component_mask": components[step],
                "component_log_probs": component_log_probs[step],
            }
        elif decisions is not None:
            kwargs = {"decision": decisions[step]}
        buffer.add(
            obs,
            torch.zeros(NUM_ENVS, NUM_COMPONENTS),
            rewards[step],
            dones[step],
            values[step],
            component_log_probs[step].sum(-1),
            final_values=final_values[step],
            **kwargs,
        )


def test_buffer_old_log_prob_sums_components_that_took_effect():
    window = _random_window()
    components = _scripted_components()
    buffer = _component_buffer()
    _fill(buffer, window, components=components)
    assert buffer.window_has_components and buffer.window_has_decisions
    buffer.compute_returns_and_advantage(window[4], window[3][-1])

    component_log_probs = window[5].double()
    expected = torch.zeros(NUM_STEPS, NUM_ENVS, dtype=torch.float64)
    for component in range(NUM_COMPONENTS):
        used = components[..., component]
        expected[used] += component_log_probs[..., component][used]
    assert torch.allclose(buffer.log_probs.double(), expected, atol=1e-6)
    no_decision = ~components.any(-1)
    assert no_decision.any()
    assert (buffer.log_probs[no_decision] == 0).all()


def test_buffer_ignores_log_probs_of_components_that_did_not_take_effect():
    window = _random_window()
    components = _scripted_components()
    clean = _component_buffer()
    _fill(clean, window, components=components)
    unused_values = torch.tensor([float("inf"), -float("inf"), float("nan")])
    noise = unused_values[torch.arange(components.numel()) % 3].reshape(components.shape)
    perturbed_window = (*window[:5], torch.where(components, window[5], noise))
    perturbed = _component_buffer()
    _fill(perturbed, perturbed_window, components=components)

    for buffer in (clean, perturbed):
        buffer.compute_returns_and_advantage(window[4], window[3][-1])
    assert torch.isfinite(perturbed.log_probs).all()
    assert torch.equal(perturbed.log_probs, clean.log_probs)
    assert torch.equal(perturbed.advantages, clean.advantages)
    assert torch.equal(perturbed.returns, clean.returns)


def test_buffer_component_gae_equals_whole_step_gae_of_any_component():
    window = _random_window()
    rewards, values, final_values, dones, last_values, _ = window
    components = _scripted_components()
    decisions = components.any(-1)
    with_components = _component_buffer()
    with_decisions = _component_buffer()
    _fill(with_components, window, components=components)
    _fill(with_decisions, window, decisions=decisions)
    for buffer in (with_components, with_decisions):
        buffer.compute_returns_and_advantage(last_values, dones[-1])

    assert torch.equal(with_components.decisions, decisions)
    assert torch.equal(with_components.advantages, with_decisions.advantages)
    assert torch.equal(with_components.returns, with_decisions.returns)
    step_lambda = torch.full(decisions.shape, GAE_LAMBDA)
    step_lambda[:-1] = torch.where(decisions[1:], GAE_LAMBDA, 1.0)
    starts = torch.zeros_like(dones)
    starts[1:] = dones[:-1]
    expected_adv, expected_ret = compute_gae(
        rewards, starts, values, final_values, last_values, dones[-1], GAMMA, GAE_LAMBDA,
        step_lambda=step_lambda,
    )
    assert torch.equal(with_components.advantages, expected_adv)
    assert torch.equal(with_components.returns, expected_ret)


def test_buffer_components_on_every_step_match_plain_gae():
    window = _random_window()
    components = _scripted_components()
    components[..., 0] |= ~components.any(-1)  # every step is a decision
    with_components = _component_buffer()
    plain = _component_buffer(store_decisions=False)
    _fill(with_components, window, components=components)
    _fill(plain, window)
    for buffer in (with_components, plain):
        buffer.compute_returns_and_advantage(window[4], window[3][-1])

    assert torch.allclose(with_components.advantages, plain.advantages, atol=1e-6)
    assert torch.allclose(with_components.returns, plain.returns, atol=1e-6)


def test_buffer_samples_carry_component_mask():
    window = _random_window()
    components = _scripted_components()
    buffer = _component_buffer()
    _fill(buffer, window, components=components)
    buffer.compute_returns_and_advantage(window[4], window[3][-1])

    flat_components = components.reshape(-1, NUM_COMPONENTS)
    seen = 0
    for sample in buffer.get(10):
        mb_inds = sample.obs[:, 0].long()
        assert sample.component_mask.dtype == torch.bool
        assert torch.equal(sample.component_mask, flat_components[mb_inds])
        seen += mb_inds.numel()
    assert seen == buffer.buffer_size


def test_recurrent_buffer_sequences_carry_component_mask():
    window = _random_window()
    components = _scripted_components()
    buffer = _component_buffer(RecurrentRolloutBuffer)
    _fill(buffer, window, components=components)
    buffer.compute_returns_and_advantage(window[4], window[3][-1])

    generator = torch.Generator().manual_seed(0)
    initial_hidden = torch.zeros(1, NUM_ENVS, 3)
    for sample in buffer.get_sequences(2, initial_hidden, generator=generator):
        idx = sample.obs[0, :, 1].long()
        assert sample.component_mask.shape == (NUM_STEPS, 2, NUM_COMPONENTS)
        assert torch.equal(sample.component_mask, components[:, idx])


def test_buffer_without_component_mask_yields_none_and_keeps_log_probs():
    window = _random_window()
    buffer = _component_buffer()
    _fill(buffer, window, decisions=_scripted_components().any(-1))
    buffer.compute_returns_and_advantage(window[4], window[3][-1])

    assert buffer.window_has_decisions and not buffer.window_has_components
    assert torch.equal(buffer.log_probs, window[5].sum(-1))
    assert all(sample.component_mask is None for sample in buffer.get(8))
    sequences = _component_buffer(RecurrentRolloutBuffer)
    _fill(sequences, window, decisions=_scripted_components().any(-1))
    sequences.compute_returns_and_advantage(window[4], window[3][-1])
    hidden = torch.zeros(1, NUM_ENVS, 3)
    assert all(sample.component_mask is None for sample in sequences.get_sequences(2, hidden))


def test_buffer_reset_restores_all_true_component_mask():
    window = _random_window()
    buffer = _component_buffer()
    _fill(buffer, window, components=_scripted_components())
    buffer.reset()
    assert bool(buffer.component_mask.all())
    assert not buffer.window_has_components


def _add_kwargs(**overrides) -> dict:
    kwargs = {
        "obs": torch.zeros(NUM_ENVS, 2),
        "actions": torch.zeros(NUM_ENVS, NUM_COMPONENTS),
        "rewards": torch.zeros(NUM_ENVS),
        "dones": torch.zeros(NUM_ENVS),
        "values": torch.zeros(NUM_ENVS),
        "log_probs": torch.zeros(NUM_ENVS),
        "decision": torch.ones(NUM_ENVS, dtype=torch.bool),
        "component_mask": torch.ones(NUM_ENVS, NUM_COMPONENTS, dtype=torch.bool),
        "component_log_probs": torch.zeros(NUM_ENVS, NUM_COMPONENTS),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"component_log_probs": None}, "together"),
        ({"component_mask": None}, "together"),
        ({"decision": None}, "without the decision"),
    ],
)
def test_buffer_add_rejects_incomplete_component_rows(overrides, match):
    with pytest.raises(ValueError, match=match):
        _component_buffer().add(**_add_kwargs(**overrides))


def test_buffer_without_decision_storage_rejects_a_component_mask():
    with pytest.raises(ValueError, match="store_decisions"):
        _component_buffer(store_decisions=False).add(**_add_kwargs(decision=None))


def test_buffer_rejects_component_mask_on_only_some_steps():
    window = _random_window()
    buffer = _component_buffer()
    _fill(
        buffer,
        window,
        components=_scripted_components(),
        decisions=_scripted_components().any(-1),
        component_steps=set(range(NUM_STEPS)) - {5},
    )
    with pytest.raises(ValueError, match="decision_components'\\] on every step"):
        buffer.compute_returns_and_advantage(window[4], window[3][-1])


# ---------------------------------------------------------------------------
# (j) components that did not take effect get no policy gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lr_schedule", LR_SCHEDULES)
@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_unused_components_do_not_change_loss_or_gradients(
    monkeypatch, agent_cls, extra_kwargs, lr_schedule
):
    agent = agent_cls(
        _ComponentMaskVecEnv(),
        **_ppo_kwargs(decision_level=True, lr_schedule=lr_schedule, ent_coef=0.01),
        **extra_kwargs,
    )
    _collect_windows(agent)
    _disable_optimizer_step(monkeypatch, agent)
    data = next(agent._iter_minibatches())
    unused = ~data.component_mask
    decision = data.component_mask.any(-1)
    # Decision steps on which some component did not take effect.
    assert (decision & unused.any(-1)).any()

    # Huge actions overflow the squared error of the Gaussian log-prob.
    perturbed = dataclasses.replace(data, actions=torch.where(unused, 1e30, data.actions))
    if lr_schedule == "adaptive_kl":
        perturbed = dataclasses.replace(
            perturbed,
            old_mean=torch.where(unused, 1e30, data.old_mean),
            old_log_std=torch.where(unused, 1e4, data.old_log_std),  # exp overflows
        )

    _assert_same_update(_minibatch_update(agent, data), _minibatch_update(agent, perturbed))


@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_unused_stored_component_log_probs_do_not_change_the_window(agent_cls, extra_kwargs):
    def collected_buffer(perturb: bool):
        agent = agent_cls(
            _ComponentMaskVecEnv(), **_ppo_kwargs(decision_level=True), **extra_kwargs
        )
        buffer = agent.rollout_buffer
        compute = buffer.compute_returns_and_advantage

        def compute_after_perturbing(*args, **kwargs):
            if perturb:
                buffer.component_log_probs.masked_fill_(~buffer.component_mask, -1e30)
            return compute(*args, **kwargs)

        buffer.compute_returns_and_advantage = compute_after_perturbing
        _collect_windows(agent)
        return buffer

    clean, perturbed = collected_buffer(False), collected_buffer(True)
    assert (perturbed.component_log_probs == -1e30).any()
    for name in ("log_probs", "advantages", "returns"):
        assert torch.equal(getattr(perturbed, name), getattr(clean, name)), name


@pytest.mark.parametrize("lr_schedule", LR_SCHEDULES)
@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_never_used_component_gets_exactly_zero_gradient(
    monkeypatch, agent_cls, extra_kwargs, lr_schedule
):
    agent = agent_cls(
        _ComponentMaskVecEnv(cycles=COMPONENT_CYCLES_WITHOUT_LAST),
        **_ppo_kwargs(decision_level=True, lr_schedule=lr_schedule, ent_coef=0.01),
        **extra_kwargs,
    )
    _collect_windows(agent)
    assert not agent.rollout_buffer.component_mask[..., 2].any()
    _disable_optimizer_step(monkeypatch, agent)
    _minibatch_update(agent, next(agent._iter_minibatches()))

    actor = agent.policy.actor
    assert (actor.mean.weight.grad[2] == 0).all()
    assert actor.mean.bias.grad[2] == 0
    assert actor.log_std.grad[0, 2] == 0
    assert actor.mean.weight.grad[:2].abs().sum() > 0
    assert (actor.log_std.grad[0, :2] != 0).all()


@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_never_used_component_parameters_stay_at_init_through_training(agent_cls, extra_kwargs):
    agent = agent_cls(
        _ComponentMaskVecEnv(cycles=COMPONENT_CYCLES_WITHOUT_LAST),
        **_ppo_kwargs(decision_level=True, ent_coef=0.01, num_minibatches=2, update_epochs=2),
        **extra_kwargs,
    )
    actor = agent.policy.actor
    initial = {name: p.detach().clone() for name, p in actor.named_parameters()}
    agent.learn(total_timesteps=2 * NUM_ENVS * NUM_STEPS)

    assert torch.equal(actor.mean.weight[2], initial["mean.weight"][2])
    assert torch.equal(actor.mean.bias[2], initial["mean.bias"][2])
    assert torch.equal(actor.log_std[0, 2], initial["log_std"][0, 2])
    assert not torch.equal(actor.log_std[0, :2], initial["log_std"][0, :2])


def test_component_mask_requires_decision_mask():
    agent = PPO(_ComponentMaskVecEnv(), **_ppo_kwargs(decision_level=True))
    batch = torch.zeros(4)
    with pytest.raises(ValueError, match="requires decision_mask"):
        agent._ppo_minibatch_update(
            values=batch, log_prob=torch.zeros(4, NUM_COMPONENTS),
            entropy=torch.zeros(4, NUM_COMPONENTS), old_values=batch, old_log_prob=batch,
            advantages=batch, returns=batch, clip_coef=0.2,
            component_mask=torch.ones(4, NUM_COMPONENTS, dtype=torch.bool),
        )


# ---------------------------------------------------------------------------
# (k) per-step terms are sums over the components that took effect
# ---------------------------------------------------------------------------


def _used_sum(per_component: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, per_component, 0.0).sum(-1)


def test_per_step_terms_are_sums_over_components_that_took_effect(monkeypatch):
    agent = PPO(
        _ComponentMaskVecEnv(),
        **_ppo_kwargs(decision_level=True, lr_schedule="adaptive_kl", ent_coef=0.01),
    )
    _collect_windows(agent)
    _disable_optimizer_step(monkeypatch, agent)
    with torch.no_grad():
        # Move the policy away from the rollout policy: non-trivial ratios and KL.
        for parameter in agent.policy.actor.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
    data = next(agent._iter_minibatches())

    captured: dict[str, torch.Tensor] = {}
    policy_loss = agent._policy_loss

    def recording_policy_loss(**kwargs):
        captured.update(advantages=kwargs["advantages"].detach(), ratio=kwargs["ratio"].detach())
        return policy_loss(**kwargs)

    def recording_allreduce_mean(kl_mean):
        captured["kl_mean"] = kl_mean
        return kl_mean

    monkeypatch.setattr(agent, "_policy_loss", recording_policy_loss)
    monkeypatch.setattr(ppo_module, "allreduce_mean", recording_allreduce_mean)
    evaluated = agent._evaluate_minibatch(data)
    assert evaluated[1].shape == evaluated[2].shape == (NUM_ENVS * NUM_STEPS, NUM_COMPONENTS)
    result, _, _ = _minibatch_update(agent, data)

    # Hand-computed Gaussian terms, per component, in float64.
    with torch.no_grad():
        _, _, _, new_mean, new_log_std = agent.policy.evaluate_actions_with_dist_params(
            data.obs, data.actions
        )
    mask = data.component_mask
    decision = mask.any(-1)
    count = int(decision.sum())
    assert 0 < count < decision.numel()
    mean, log_std, actions = new_mean.double(), new_log_std.double(), data.actions.double()
    old_mean, old_log_std = data.old_mean.double(), data.old_log_std.double()
    log_prob = -((actions - mean) ** 2) / (2 * torch.exp(2 * log_std)) - log_std
    log_prob = log_prob - 0.5 * math.log(2 * math.pi)
    entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
    kl = (log_std - old_log_std) - 0.5
    kl = kl + (torch.exp(2 * old_log_std) + (old_mean - mean) ** 2) / (2 * torch.exp(2 * log_std))
    logratio = _used_sum(log_prob, mask) - data.old_log_prob.double()
    logratio = logratio[decision]

    ratio = captured["ratio"]
    assert torch.allclose(ratio[decision].double(), logratio.exp(), rtol=1e-4)
    assert (ratio[~decision] == 1).all()
    assert result["decisions"] == count
    assert result["old_approx_kl"] == pytest.approx(float(-logratio.mean()), rel=1e-4, abs=1e-6)
    expected_approx_kl = float(((logratio.exp() - 1) - logratio).mean())
    assert result["approx_kl"] == pytest.approx(expected_approx_kl, rel=1e-3, abs=1e-6)
    clipped = ((ratio[decision] - 1).abs() > agent.clip_coef).float()
    assert result["clipfrac"] == pytest.approx(float(clipped.sum()) / count)
    expected_entropy = float(_used_sum(entropy, mask)[decision].mean())
    assert result["entropy_loss"] == pytest.approx(expected_entropy, rel=1e-5)
    expected_kl = float(_used_sum(kl, mask)[decision].mean())
    assert float(captured["kl_mean"]) == pytest.approx(expected_kl, rel=1e-4, abs=1e-7)

    advantages = data.advantages.double()[decision]
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    assert torch.allclose(captured["advantages"][decision].double(), normalized, atol=1e-5)
    ratio_ref = logratio.exp()
    surrogate = torch.max(
        -normalized * ratio_ref,
        -normalized * ratio_ref.clamp(1 - agent.clip_coef, 1 + agent.clip_coef),
    )
    assert result["policy_loss"] == pytest.approx(float(surrogate.mean()), rel=1e-4, abs=1e-6)


def test_sequence_evaluation_keeps_component_axis_t_major():
    agent = RecurrentPPO(
        _ComponentMaskVecEnv(), **_ppo_kwargs(decision_level=True), **RECURRENT_KWARGS
    )
    _collect_windows(agent)
    data = next(agent.rollout_buffer.get_sequences(1, agent._rollout_initial_hidden, shuffle=False))

    _, log_prob, entropy, _, old_log_prob, *_ = agent._evaluate_minibatch(data)
    masks = agent._minibatch_masks(data)
    _, expected_log_prob, expected_entropy = agent.policy.evaluate_actions_sequence(
        data.obs,
        torch.where(data.component_mask, data.actions, 0.0),
        data.initial_hidden,
        data.episode_starts,
        stop_gradient_actor=agent.policy.actor_features_detached,
        sum_dims=False,
    )
    rows = NUM_STEPS * NUM_ENVS
    components = _scripted_components()
    component_mask = masks["component_mask"]
    assert log_prob.shape == entropy.shape == component_mask.shape == (rows, NUM_COMPONENTS)
    assert torch.equal(log_prob.reshape(NUM_STEPS, NUM_ENVS, -1), expected_log_prob)
    assert torch.equal(entropy.reshape(NUM_STEPS, NUM_ENVS, -1), expected_entropy)
    assert torch.equal(component_mask.reshape(NUM_STEPS, NUM_ENVS, -1), components)
    assert torch.equal(masks["decision_mask"].reshape(NUM_STEPS, NUM_ENVS), components.any(-1))
    assert torch.equal(old_log_prob, agent.rollout_buffer.log_probs.reshape(-1))


# ---------------------------------------------------------------------------
# (l) SMDP GAE over the decisions of a component mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_component_window_gae_matches_whole_step_decisions(agent_cls, extra_kwargs):
    def collected_buffer(report: str):
        agent = agent_cls(
            _ComponentMaskVecEnv(report), **_ppo_kwargs(decision_level=True), **extra_kwargs
        )
        _collect_windows(agent)
        return agent.rollout_buffer

    with_components, with_decisions = collected_buffer("components"), collected_buffer("decision")

    assert with_components.window_has_components and not with_decisions.window_has_components
    assert torch.equal(with_components.decisions, _scripted_components().any(-1))
    assert torch.equal(with_components.decisions, with_decisions.decisions)
    assert torch.equal(with_components.advantages, with_decisions.advantages)
    assert torch.equal(with_components.returns, with_decisions.returns)


# ---------------------------------------------------------------------------
# (m) a component mask broadcast from whole-step decisions trains the same
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lr_schedule", LR_SCHEDULES)
@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_broadcast_component_mask_trains_like_whole_step_decisions(
    agent_cls, extra_kwargs, lr_schedule
):
    def trained(report: str):
        agent = agent_cls(
            _ComponentMaskVecEnv(report),
            **_ppo_kwargs(
                decision_level=True,
                lr_schedule=lr_schedule,
                num_minibatches=2,
                update_epochs=2,
                ent_coef=0.01,
            ),
            **extra_kwargs,
        )
        initial = {name: t.clone() for name, t in agent.policy.state_dict().items()}
        agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)
        return agent, initial

    broadcast, initial = trained("broadcast")
    whole_step, _ = trained("decision")

    assert broadcast.rollout_buffer.window_has_components
    assert not whole_step.rollout_buffer.window_has_components
    reference = whole_step.policy.state_dict()
    for name, tensor in broadcast.policy.state_dict().items():
        assert torch.allclose(tensor, reference[name], rtol=1e-4, atol=1e-5), name
    assert any(not torch.equal(reference[name], t) for name, t in initial.items())


# ---------------------------------------------------------------------------
# (n) bit-identity without a component mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lr_schedule", LR_SCHEDULES)
@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_without_a_component_mask_training_is_bit_identical(agent_cls, extra_kwargs, lr_schedule):
    def trained_parameters(report: str, decision_level: bool) -> dict:
        agent = agent_cls(
            _ComponentMaskVecEnv(report),
            **_ppo_kwargs(
                decision_level=decision_level,
                lr_schedule=lr_schedule,
                num_minibatches=2,
                update_epochs=2,
            ),
            **extra_kwargs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.learn(total_timesteps=agent.num_envs * agent.num_steps)
        return agent.policy.state_dict()

    reference = trained_parameters("none", decision_level=False)
    _assert_state_dicts_equal(reference, trained_parameters("components", decision_level=False))
    _assert_state_dicts_equal(reference, trained_parameters("none", decision_level=True))
    with_components = trained_parameters("components", decision_level=True)
    assert any(not torch.equal(with_components[name], t) for name, t in reference.items())


# ---------------------------------------------------------------------------
# (o) contract
# ---------------------------------------------------------------------------


def test_reader_returns_decision_and_component_mask():
    reader = DecisionMaskReader(NUM_ENVS, (NUM_COMPONENTS,), "cpu")
    components = _scripted_components()[1]
    decision, component_mask = reader.read({"decision_components": components})
    assert component_mask.dtype == torch.bool
    assert torch.equal(component_mask, components)
    assert torch.equal(decision, components.any(-1))

    decision, component_mask = DecisionMaskReader(NUM_ENVS, (NUM_COMPONENTS,), "cpu").read(
        {"decision": components.any(-1)}
    )
    assert component_mask is None
    assert torch.equal(decision, components.any(-1))


@pytest.mark.parametrize(
    "first,second",
    [
        ("decision_components", "decision"),
        ("decision", "decision_components"),
        ("decision_components", None),
        (None, "decision_components"),
    ],
)
def test_reader_rejects_component_key_set_changing_within_a_run(first, second):
    values = {
        "decision": torch.ones(NUM_ENVS, dtype=torch.bool),
        "decision_components": torch.ones(NUM_ENVS, NUM_COMPONENTS, dtype=torch.bool),
    }
    reader = DecisionMaskReader(NUM_ENVS, (NUM_COMPONENTS,), "cpu")
    reader.read({} if first is None else {first: values[first]})
    with pytest.raises(ValueError, match="key set changed at step 1"):
        reader.read({} if second is None else {second: values[second]})


def test_reader_rejects_both_keys_with_components():
    reader = DecisionMaskReader(NUM_ENVS, (NUM_COMPONENTS,), "cpu")
    infos = {
        "decision": torch.ones(NUM_ENVS, dtype=torch.bool),
        "decision_components": torch.ones(NUM_ENVS, NUM_COMPONENTS, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="both"):
        reader.read(infos)


_MALFORMED_COMPONENTS = [
    (lambda mask: mask.float(), "dtype"),
    (lambda mask: mask.to(torch.uint8), "dtype"),
    (lambda mask: mask[:, :2], "shape"),
    (lambda mask: mask.any(-1), "shape"),
    (lambda mask: mask[:, :, None], "shape"),
    (lambda mask: mask.numpy(), "torch.Tensor"),
]


@pytest.mark.parametrize("malform,match", _MALFORMED_COMPONENTS)
def test_reader_rejects_malformed_component_mask(malform, match):
    reader = DecisionMaskReader(NUM_ENVS, (NUM_COMPONENTS,), "cpu")
    with pytest.raises(ValueError, match=match):
        reader.read({"decision_components": malform(_scripted_components()[0])})


class _MalformedComponentsEnv(_ComponentMaskVecEnv):
    def __init__(self, malform) -> None:
        super().__init__()
        self.malform = malform

    def step(self, action):
        *outputs, infos = super().step(action)
        infos["decision_components"] = self.malform(infos["decision_components"])
        return (*outputs, infos)


@pytest.mark.parametrize("malform,match", _MALFORMED_COMPONENTS)
def test_learn_rejects_malformed_component_mask(malform, match):
    agent = PPO(_MalformedComponentsEnv(malform), **_ppo_kwargs(decision_level=True))
    agent.train = _skip_update
    with pytest.raises(ValueError, match=match):
        agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)


@pytest.mark.parametrize(
    "env_kwargs,match",
    [
        ({"drop_key_at_step": 3}, "key set changed at step 3"),
        ({"switch_key_at_step": 4}, "key set changed at step 4"),
    ],
)
def test_learn_rejects_component_key_missing_or_switched(env_kwargs, match):
    agent = PPO(_ComponentMaskVecEnv(**env_kwargs), **_ppo_kwargs(decision_level=True))
    agent.train = _skip_update
    with pytest.raises(ValueError, match=match):
        agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)


class _ComponentFrameEnv(_FakeBatchedEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_components = torch.tensor([[True], [False]])
        info = {**info, "decision_components": self.last_components}
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("autoreset_in_step", [False, True])
def test_image_frame_stack_forwards_decision_components(autoreset_in_step):
    inner = _ComponentFrameEnv()
    env = ImageFrameStackWrapper(inner, frame_stack=3, autoreset_in_step=autoreset_in_step)
    env.reset()
    *_, info = env.step(torch.zeros(2, 1))
    assert info["decision_components"] is inner.last_components


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

COMPONENT_METRIC_KEYS = {
    "train/component_usage/0",
    "train/component_usage/1",
    "train/component_usage/2",
    "train/components_per_decision",
}


def _train_one_window_with_recording_logger(env, **overrides) -> PPO:
    agent = PPO(env, **_ppo_kwargs(log_freq=1, num_minibatches=2, **overrides))
    agent.logger = _RecordingLogger()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)
    return agent


def test_component_metrics_are_logged():
    agent = _train_one_window_with_recording_logger(_ComponentMaskVecEnv(), decision_level=True)
    scalars = agent.logger.scalars
    components = _scripted_components()
    decision = components.any(-1)

    assert COMPONENT_METRIC_KEYS <= set(scalars)
    for component in range(NUM_COMPONENTS):
        usage = float(components[..., component].float().mean())
        assert scalars[f"train/component_usage/{component}"] == [pytest.approx(usage)]
    per_decision = float(components.sum(-1)[decision].float().mean())
    assert scalars["train/components_per_decision"] == [pytest.approx(per_decision)]
    assert scalars["train/decisions"] == [int(decision.sum())]


@pytest.mark.parametrize(
    "report,decision_level", [("decision", True), ("components", False), ("none", True)]
)
def test_component_metrics_are_absent_without_a_component_window(report, decision_level):
    agent = _train_one_window_with_recording_logger(
        _ComponentMaskVecEnv(report), decision_level=decision_level
    )
    assert "losses/loss" in agent.logger.scalars
    assert not set(agent.logger.scalars) & COMPONENT_METRIC_KEYS


# ---------------------------------------------------------------------------
# (i) no extra host synchronization in the rollout
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_component_rollout_adds_no_host_synchronization():
    def synchronizing_calls(decision_level: bool) -> int:
        agent = PPO(
            _ComponentMaskVecEnv(device="cuda"),
            **_ppo_kwargs(device="cuda", decision_level=decision_level),
        )
        agent.train = _skip_update
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        assert agent.rollout_buffer.window_has_components == decision_level
        return sum("synchronizing" in str(w.message) for w in caught)

    synchronizing_calls(decision_level=False)  # warm-up: one-time CUDA lazy init syncs
    feature_off = synchronizing_calls(decision_level=False)
    assert feature_off > 0  # the debug mode is active (per-step reward logging syncs)
    assert synchronizing_calls(decision_level=True) == feature_off
