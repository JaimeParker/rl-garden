"""Decision-level (semi-Markov) PPO losses: the decision mask in the policy,
entropy, KL and value terms, ``per_step_critic``, the CLI flags and the
decision metrics."""

from __future__ import annotations

import dataclasses
import importlib
import math
import warnings

import pytest
import torch
from _decision_fakes import NUM_ENVS, NUM_STEPS, _FixedHoldVecEnv, _scripted_decisions
from test_rollout_episode_metrics import _RecordingLogger

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.algorithms.ppo import ppo_clip_policy_loss

GAMMA = 0.9
GAE_LAMBDA = 0.8
RECURRENT_KWARGS = {"rnn_hidden_size": 8}
AGENT_CLASSES = [(PPO, {}), (RecurrentPPO, RECURRENT_KWARGS)]


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


def _state_sum_values(obs, *args, **kwargs):
    del args, kwargs
    return obs["state"].sum(dim=-1, keepdim=True)


def _disable_optimizer_step(monkeypatch, agent) -> None:
    """Keep the parameters fixed so repeated updates see the same policy;
    gradients stay readable after ``_ppo_minibatch_update`` returns."""
    monkeypatch.setattr(agent.policy_optimizer, "step", lambda *args, **kwargs: None)


def _update_and_grads(agent, data) -> tuple[dict, dict[str, torch.Tensor | None]]:
    (
        values, log_prob, entropy, old_values, old_log_prob, advantages, returns, *_,
    ) = agent._evaluate_minibatch(data)
    result = agent._ppo_minibatch_update(
        values=values,
        log_prob=log_prob,
        entropy=entropy,
        old_values=old_values,
        old_log_prob=old_log_prob,
        advantages=advantages,
        returns=returns,
        clip_coef=agent.clip_coef,
        **agent._minibatch_masks(data),
    )
    grads = {
        name: None if p.grad is None else p.grad.clone()
        for name, p in agent.policy.named_parameters()
    }
    return result, grads


_LOSS_KEYS = (
    "loss", "policy_loss", "value_loss", "entropy_loss", "approx_kl", "old_approx_kl", "clipfrac",
)


def _assert_same_update(first, second) -> None:
    (result_a, grads_a), (result_b, grads_b) = first, second
    for key in _LOSS_KEYS:
        assert math.isfinite(result_a[key]), key
        assert result_a[key] == result_b[key], key
    assert torch.equal(result_a["decisions"], result_b["decisions"])
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


def _perturb_policy_inputs(data, rows: torch.Tensor):
    """Actions, old log-probs (``exp`` of the log-ratio overflows) and
    advantages of ``rows``."""
    return dataclasses.replace(
        data,
        actions=torch.where(rows.unsqueeze(-1), data.actions + 0.7, data.actions),
        old_log_prob=torch.where(rows, -1e4, data.old_log_prob),
        advantages=torch.where(rows, 1e6, data.advantages),
    )


def _perturb_value_targets(data, rows: torch.Tensor):
    return dataclasses.replace(
        data,
        returns=torch.where(rows, data.returns + 1e3, data.returns),
        old_values=torch.where(rows, data.old_values - 1e3, data.old_values),
    )


# ---------------------------------------------------------------------------
# Masked surrogate
# ---------------------------------------------------------------------------


def test_masked_clip_policy_loss_averages_true_rows():
    generator = torch.Generator().manual_seed(0)
    advantages = torch.randn(10, generator=generator)
    ratio = torch.rand(10, generator=generator) + 0.5
    mask = torch.tensor([True, False] * 5)

    masked = ppo_clip_policy_loss(advantages, ratio, 0.2, mask=mask)
    subset = ppo_clip_policy_loss(advantages[mask], ratio[mask], 0.2)

    assert torch.allclose(masked, subset)
    assert ppo_clip_policy_loss(advantages, ratio, 0.2, mask=torch.zeros(10, dtype=torch.bool)) == 0


# ---------------------------------------------------------------------------
# (c) non-decision rows carry no policy gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clip_vloss", [False, True])
@pytest.mark.parametrize("per_step_critic", [True, False])
def test_non_decision_rows_do_not_change_loss_or_gradients(monkeypatch, per_step_critic, clip_vloss):
    agent = PPO(
        _FixedHoldVecEnv(),
        **_ppo_kwargs(decision_level=True, per_step_critic=per_step_critic, clip_vloss=clip_vloss),
    )
    _collect_windows(agent)
    _disable_optimizer_step(monkeypatch, agent)
    data = next(agent._iter_minibatches())
    non_decision = ~data.decision_mask
    assert non_decision.any() and data.decision_mask.any()

    perturbed = _perturb_policy_inputs(data, non_decision)
    if not per_step_critic:
        perturbed = _perturb_value_targets(perturbed, non_decision)

    _assert_same_update(_update_and_grads(agent, data), _update_and_grads(agent, perturbed))


# ---------------------------------------------------------------------------
# (d) BPTT: the policy loss of a decision reaches the earlier steps of its hold
# ---------------------------------------------------------------------------


def test_bptt_decision_loss_reaches_earlier_non_decision_obs():
    # encoder_sharing="shared": under the default "shared_critic_grad" the
    # actor's features are detached, so no policy loss reaches the encoder.
    agent = RecurrentPPO(
        _FixedHoldVecEnv(),
        **_ppo_kwargs(decision_level=True, encoder_sharing="shared"),
        **RECURRENT_KWARGS,
    )
    assert not agent.policy.actor_features_detached
    _collect_windows(agent)
    data = next(
        agent.rollout_buffer.get_sequences(1, agent._rollout_initial_hidden, shuffle=False)
    )
    obs = {"state": data.obs["state"].clone().requires_grad_()}

    _, log_prob, _ = agent.policy.evaluate_actions_sequence(
        obs,
        data.actions,
        data.initial_hidden,
        data.episode_starts,
        stop_gradient_actor=agent.policy.actor_features_detached,
    )
    mask = data.decision_mask.reshape(-1)
    logratio = torch.where(mask, log_prob.flatten() - data.old_log_prob.reshape(-1), 0.0)
    advantages = torch.where(mask, data.advantages.reshape(-1), 0.0)
    ppo_clip_policy_loss(advantages, logratio.exp(), 0.2, mask=mask).backward()

    # Env 3 holds for 5 steps: decisions at steps 0, 5 and 10, no episode end.
    env = 3
    assert data.decision_mask[:, env].nonzero().flatten().tolist() == [0, 5, 10]
    obs_grad = obs["state"].grad[:, env].abs().sum(dim=-1)
    assert obs_grad[3] > 0  # inside hold 0-4, feeds decision 5 through the RNN
    assert obs_grad[11] == 0  # after the last decision of the window


# ---------------------------------------------------------------------------
# (e) default paths are bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_cls,extra_kwargs", AGENT_CLASSES)
def test_without_a_decision_mask_training_is_bit_identical(agent_cls, extra_kwargs):
    def trained_parameters(emit_decisions: bool, decision_level: bool) -> dict:
        agent = agent_cls(
            _FixedHoldVecEnv(emit_decisions=emit_decisions),
            **_ppo_kwargs(decision_level=decision_level, num_minibatches=2, update_epochs=2),
            **extra_kwargs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.learn(total_timesteps=agent.num_envs * agent.num_steps)
        return agent.policy.state_dict()

    reference = trained_parameters(emit_decisions=False, decision_level=False)
    keys_ignored = trained_parameters(emit_decisions=True, decision_level=False)
    no_keys = trained_parameters(emit_decisions=False, decision_level=True)
    with_decisions = trained_parameters(emit_decisions=True, decision_level=True)

    for other in (keys_ignored, no_keys):
        assert other.keys() == reference.keys()
        for name, tensor in reference.items():
            assert torch.equal(other[name], tensor), name
    assert any(not torch.equal(with_decisions[name], t) for name, t in reference.items())


# ---------------------------------------------------------------------------
# (f) a hold spanning two windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_step_critic", [True, False])
def test_hold_spanning_windows(monkeypatch, per_step_critic):
    num_steps = 5
    env = _FixedHoldVecEnv(holds=(7, 1), episode_ends={})
    agent = PPO(
        env,
        **_ppo_kwargs(decision_level=True, per_step_critic=per_step_critic, num_steps=num_steps),
    )
    agent.policy.predict_values = _state_sum_values
    windows: list[dict[str, torch.Tensor]] = []

    def record_window() -> dict[str, float]:
        buffer = agent.rollout_buffer
        windows.append(
            {
                "rewards": buffer.rewards.clone(),
                "returns": buffer.returns.clone(),
                "decisions": buffer.decisions.clone(),
                "last_values": env.last_state.sum(dim=-1),
            }
        )
        return {}

    agent.train = record_window
    agent.learn(total_timesteps=2 * env.num_envs * num_steps)

    # Window 1: the decision at step 0 holds until step 7, past the window,
    # so its return is the partial reward sum plus gamma^5 * last_values.
    first = windows[0]
    assert first["decisions"][:, 0].tolist() == [True, False, False, False, False]
    partial = sum(GAMMA**k * first["rewards"][k, 0] for k in range(num_steps))
    expected = partial + GAMMA**num_steps * first["last_values"][0]
    assert torch.allclose(first["returns"][0, 0], expected, atol=1e-5)

    # Window 2 opens inside that hold: steps 5 and 6 are not decisions.
    assert windows[1]["decisions"][:, 0].tolist() == [False, False, True, False, False]
    _disable_optimizer_step(monkeypatch, agent)
    data = next(agent._iter_minibatches())
    state = data.obs["state"]
    env_steps = torch.round(state[:, 2] * 10)
    window_start = (state[:, 1] == 0) & (env_steps < 7)
    assert int(window_start.sum()) == 2
    assert not data.decision_mask[window_start].any()

    clean = _update_and_grads(agent, data)
    _assert_same_update(clean, _update_and_grads(agent, _perturb_policy_inputs(data, window_start)))
    shifted_returns = _update_and_grads(agent, _perturb_value_targets(data, window_start))
    if per_step_critic:
        assert shifted_returns[0]["value_loss"] != clean[0]["value_loss"]
    else:
        assert shifted_returns[0]["value_loss"] == clean[0]["value_loss"]


# ---------------------------------------------------------------------------
# (g) a minibatch without decisions
# ---------------------------------------------------------------------------


def test_minibatch_without_decisions_is_finite_and_never_stops_early():
    num_steps, update_epochs = 5, 3
    # Env 1 decides at steps 0 and 12: the second window (steps 5-9) has none.
    env = _FixedHoldVecEnv(holds=(1, 12), episode_ends={})
    agent = RecurrentPPO(
        env,
        **_ppo_kwargs(
            decision_level=True,
            num_steps=num_steps,
            num_minibatches=2,
            update_epochs=update_epochs,
            target_kl=1e-9,
        ),
        **RECURRENT_KWARGS,
    )
    _collect_windows(agent, num_windows=2)
    buffer = agent.rollout_buffer
    assert buffer.decisions[:, 0].all() and not buffer.decisions[:, 1].any()

    # Extreme old log-probs on the rows without a decision; the decision
    # rows are evaluated at exactly their rollout log-probs (ratio 1), so
    # only the decision-free minibatch could trip target_kl.
    rollout_log_probs = buffer.log_probs.clone()
    buffer.log_probs[:, 1] = -1e4
    evaluate = agent._evaluate_minibatch
    update = agent._ppo_minibatch_update
    results: list[dict] = []

    def evaluate_at_rollout_log_probs(data):
        outputs = list(evaluate(data))
        env_ids = torch.round(data.obs["state"][0, :, 1] * 10).long()
        log_prob = outputs[1]
        outputs[1] = rollout_log_probs[:, env_ids].reshape(-1) + (log_prob - log_prob.detach())
        return tuple(outputs)

    def recording_update(**kwargs):
        result = update(**kwargs)
        results.append(dict(result))
        return result

    agent._evaluate_minibatch = evaluate_at_rollout_log_probs
    agent._ppo_minibatch_update = recording_update
    metrics = type(agent).train(agent)

    assert len(results) == update_epochs * 2
    assert not any(result["stop"] for result in results)
    empty = [result for result in results if int(result["decisions"]) == 0]
    assert len(empty) == update_epochs
    for result in empty:
        assert result["policy_loss"] == 0.0
        assert result["entropy_loss"] == 0.0
        assert result["approx_kl"] == 0.0
        assert result["clipfrac"] == 0.0
        assert math.isfinite(result["value_loss"]) and math.isfinite(result["loss"])
    assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
    assert metrics["decisions_per_minibatch_min"] == 0.0
    assert metrics["decisions_per_minibatch_mean"] == pytest.approx(num_steps / 2)


# ---------------------------------------------------------------------------
# per_step_critic and the all-true mask
# ---------------------------------------------------------------------------


def _synthetic_minibatch(size: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return {
        name: torch.randn(size, generator=generator)
        for name in (
            "values", "log_prob", "entropy", "old_values", "old_log_prob", "advantages", "returns",
        )
    }


def _synthetic_update(agent, batch, **mask_kwargs) -> tuple[dict, torch.Tensor]:
    values = batch["values"].clone().requires_grad_()
    result = agent._ppo_minibatch_update(
        values=values,
        log_prob=batch["log_prob"].clone().requires_grad_(),
        entropy=batch["entropy"].clone().requires_grad_(),
        old_values=batch["old_values"],
        old_log_prob=batch["old_log_prob"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        clip_coef=0.2,
        **mask_kwargs,
    )
    return result, values.grad


@pytest.mark.parametrize("clip_vloss", [False, True])
def test_all_true_mask_value_loss_is_equal_across_critic_modes(clip_vloss):
    agent = PPO(_FixedHoldVecEnv(), **_ppo_kwargs(decision_level=True, clip_vloss=clip_vloss))
    batch = _synthetic_minibatch()
    all_true = torch.ones(16, dtype=torch.bool)

    agent.per_step_critic = True
    every_step, every_step_grad = _synthetic_update(agent, batch, decision_mask=all_true)
    agent.per_step_critic = False
    decisions_only, decisions_only_grad = _synthetic_update(agent, batch, decision_mask=all_true)

    assert every_step["value_loss"] == decisions_only["value_loss"]
    assert torch.equal(every_step_grad, decisions_only_grad)


@pytest.mark.parametrize("clip_vloss", [False, True])
def test_all_true_mask_matches_the_unmasked_update(clip_vloss):
    agent = PPO(_FixedHoldVecEnv(), **_ppo_kwargs(decision_level=True, clip_vloss=clip_vloss))
    batch = _synthetic_minibatch()

    unmasked, _ = _synthetic_update(agent, batch)
    masked, _ = _synthetic_update(agent, batch, decision_mask=torch.ones(16, dtype=torch.bool))

    assert "decisions" not in unmasked
    assert int(masked.pop("decisions")) == 16
    for key in _LOSS_KEYS:
        assert masked[key] == pytest.approx(unmasked[key], rel=1e-5, abs=1e-6), key


# ---------------------------------------------------------------------------
# CLI and checkpoint metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algorithm,module,builder",
    [
        ("ppo", "rl_garden.training.online.ppo", "build_ppo"),
        ("recurrent_ppo", "rl_garden.training.online.recurrent_ppo", "build_recurrent_ppo"),
        ("transformer_ppo", "rl_garden.training.online.transformer_ppo", "build_transformer_ppo"),
    ],
)
def test_decision_level_cli_reaches_constructor(monkeypatch, algorithm, module, builder):
    from rl_garden.training import inspection
    from rl_garden.training.online import registry

    args = registry.parse_args(
        [algorithm, "--log-type", "none", "--decision-level", "--no-per-step-critic"]
    )
    defaults = registry.parse_args([algorithm, "--log-type", "none"])
    assert defaults.decision_level is False
    assert defaults.per_step_critic is True

    captured: dict = {}

    def _capture_constructor_kwargs(target, /, **kwargs):
        captured.update(kwargs)
        return target

    monkeypatch.setattr(inspection, "construct_agent", _capture_constructor_kwargs)
    getattr(importlib.import_module(module), builder)(args, _FixedHoldVecEnv(), None, None, None)

    assert captured["decision_level"] is True
    assert captured["per_step_critic"] is False


def test_gail_cli_rejects_decision_level_at_construction():
    from rl_garden.training.online import registry
    from rl_garden.training.online.gail import build_gail

    args = registry.parse_args(["gail", "--log-type", "none", "--decision-level"])
    with pytest.raises(ValueError, match="GAIL does not support decision_level"):
        build_gail(args, _FixedHoldVecEnv(), None, None, None)


def test_checkpoint_metadata_records_decision_options():
    agent = PPO(_FixedHoldVecEnv(), **_ppo_kwargs(decision_level=True, per_step_critic=False))
    metadata = agent._checkpoint_metadata()
    assert metadata["decision_level"] is True
    assert metadata["per_step_critic"] is False

    default = PPO(_FixedHoldVecEnv(), **_ppo_kwargs())._checkpoint_metadata()
    assert default["decision_level"] is False
    assert default["per_step_critic"] is True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

DECISION_METRIC_KEYS = {
    "train/decisions",
    "train/decision_fraction",
    "train/env_steps_per_decision",
    "losses/decisions_per_minibatch_min",
    "losses/decisions_per_minibatch_mean",
    "policy/log_std/0",
    "policy/log_std/1",
}


def _train_one_window_with_recording_logger(env, **overrides) -> PPO:
    agent = PPO(env, **_ppo_kwargs(log_freq=1, num_minibatches=2, **overrides))
    agent.logger = _RecordingLogger()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent.learn(total_timesteps=NUM_ENVS * NUM_STEPS)
    return agent


def test_decision_metrics_are_logged():
    agent = _train_one_window_with_recording_logger(_FixedHoldVecEnv(), decision_level=True)
    scalars = agent.logger.scalars
    # Logged after the window's only update, so it is the final log-std.
    final_log_std = agent.policy.actor.log_std.detach().view(-1).tolist()

    assert DECISION_METRIC_KEYS <= set(scalars)
    env_steps = NUM_ENVS * NUM_STEPS
    decisions = int(_scripted_decisions().sum())
    assert scalars["train/decisions"] == [decisions]
    assert scalars["train/decision_fraction"][0] == pytest.approx(decisions / env_steps)
    assert scalars["train/env_steps_per_decision"][0] == pytest.approx(env_steps / decisions)
    assert scalars["losses/decisions_per_minibatch_mean"][0] == pytest.approx(decisions / 2)
    assert scalars["policy/log_std/0"] == [pytest.approx(final_log_std[0])]


@pytest.mark.parametrize("emit_decisions", [False, True])
def test_decision_metrics_are_absent_when_decision_level_is_off(emit_decisions):
    agent = _train_one_window_with_recording_logger(_FixedHoldVecEnv(emit_decisions=emit_decisions))
    logged = set(agent.logger.scalars)

    assert "losses/loss" in logged
    assert not logged & DECISION_METRIC_KEYS
    assert not any(key.startswith("policy/log_std") for key in logged)


# ---------------------------------------------------------------------------
# (i) no extra host synchronization in the rollout
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decision_rollout_adds_no_host_synchronization():
    def synchronizing_calls(decision_level: bool) -> int:
        agent = PPO(
            _FixedHoldVecEnv(device="cuda"),
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
        return sum("synchronizing" in str(w.message) for w in caught)

    synchronizing_calls(decision_level=False)  # warm-up: one-time CUDA lazy init syncs
    feature_off = synchronizing_calls(decision_level=False)
    assert feature_off > 0  # the debug mode is active (per-step reward logging syncs)
    assert synchronizing_calls(decision_level=True) == feature_off
