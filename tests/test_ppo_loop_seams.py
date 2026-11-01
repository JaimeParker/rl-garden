"""Overridable steps of the PPO training loop: ``_value_loss``,
``_next_done``, ``_after_env_step``, ``_rollout_extras`` /
``_rollout_extra_specs``, ``_extra_update_metrics`` and
``rollout_buffer_class`` / ``_build_rollout_buffer``. Their defaults are
pinned bit for bit by ``test_ppo_loop_snapshot.py``, ``test_ppo_seams.py``
and ``test_ppo_train_snapshot.py``."""

from __future__ import annotations

import re

import pytest
import torch
from _decision_fakes import NUM_ENVS, NUM_STEPS, _scripted_flags
from test_graph_losses import (
    _collect_window,
    _first_minibatch,
    _InfoTargetEnv,
    _learn,
    _make_agent,
)
from test_ppo_seams import _RecordingLogger, _TargetPolicy

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.buffers.recurrent_rollout_buffer import RecurrentRolloutBuffer
from rl_garden.buffers.rollout_buffer import RolloutBuffer

# --- _value_loss ----------------------------------------------------------------------------


class _L1ValuePPO(PPO):
    def _value_loss(self, *, values, old_values, returns, clip_coef, decision_mask):
        return (values - returns).abs().mean()


def _first_minibatch_update(agent):
    """Collect one window and take the update step of its first minibatch;
    returns ``(evaluation, result)``."""
    _collect_window(agent)
    evaluation = agent._evaluate_minibatch(_first_minibatch(agent))
    result = agent._ppo_minibatch_update(
        values=evaluation.values,
        log_prob=evaluation.log_prob,
        entropy=evaluation.entropy,
        old_values=evaluation.old_values,
        old_log_prob=evaluation.old_log_prob,
        advantages=evaluation.advantages,
        returns=evaluation.returns,
        clip_coef=agent.clip_coef,
    )
    return evaluation, result


def test_a_value_loss_override_changes_only_the_value_term():
    default_eval, default = _first_minibatch_update(_make_agent(PPO, ent_coef=0.01))
    custom_eval, custom = _first_minibatch_update(_make_agent(_L1ValuePPO, ent_coef=0.01))

    # The same minibatch under the same initial policy.
    assert torch.equal(default_eval.values, custom_eval.values)
    assert torch.equal(default_eval.returns, custom_eval.returns)
    for key in ("policy_loss", "entropy_loss", "approx_kl", "old_approx_kl", "clipfrac"):
        assert custom[key] == default[key], key

    l1 = (custom_eval.values - custom_eval.returns).abs().mean().item()
    assert custom["value_loss"] == pytest.approx(l1, rel=1e-6)
    assert custom["value_loss"] != pytest.approx(default["value_loss"])
    total = custom["policy_loss"] - 0.01 * custom["entropy_loss"] + 0.5 * custom["value_loss"]
    assert custom["loss"] == pytest.approx(total, rel=1e-6)


# --- _next_done -----------------------------------------------------------------------------


class _TruncationContinuesPPO(PPO):
    """A truncated episode is not cut: GAE runs through it."""

    def _next_done(self, terminations, truncations, infos):
        return terminations.to(self.device, torch.bool)


def test_a_next_done_override_is_the_episode_cut_the_rollout_buffer_stores():
    # Each agent seeds the global RNG when built: build, then collect.
    default = _make_agent(PPO)
    _collect_window(default)
    custom = _make_agent(_TruncationContinuesPPO)
    _collect_window(custom)
    terminated = _scripted_flags("terminated")[:NUM_STEPS]
    truncated = _scripted_flags("truncated")[:NUM_STEPS]
    assert truncated.any()

    assert torch.equal(default.rollout_buffer.dones, (terminated | truncated).float())
    assert torch.equal(custom.rollout_buffer.dones, terminated.float())
    # Only the cut changed: same rollout, same bootstrap values.
    for name in ("rewards", "values", "log_probs", "final_values"):
        assert torch.equal(
            getattr(custom.rollout_buffer, name), getattr(default.rollout_buffer, name)
        ), name
    # GAE propagates through the truncation of env 1 (after step 4) only
    # without the cut.
    step, env = truncated.nonzero()[0].tolist()
    assert not torch.equal(
        custom.rollout_buffer.advantages[: step + 1, env],
        default.rollout_buffer.advantages[: step + 1, env],
    )
    assert torch.equal(
        custom.rollout_buffer.advantages[step + 1 :, env],
        default.rollout_buffer.advantages[step + 1 :, env],
    )


# --- _rollout_extras, _rollout_extra_specs, _after_env_step ---------------------------------


ROLLOUT_EXTRA_SPECS = {
    "rollout_value": ((1,), torch.float32),
    "step_reward": ((1,), torch.float32),
}


class _RolloutExtrasPPO(PPO):
    """Stores the rollout-time value (stashed in ``_rollout_policy``) and
    the step's reward (stashed in ``_after_env_step``) next to the policy's
    info target, and regresses the current value on the stored one."""

    policy_class = _TargetPolicy

    def _rollout_extra_specs(self):
        return {**super()._rollout_extra_specs(), **ROLLOUT_EXTRA_SPECS}

    def _rollout_policy(self, obs):
        actions, values, log_prob, entropy = super()._rollout_policy(obs)
        self._step_value = values
        return actions, values, log_prob, entropy

    def _after_env_step(self, rewards, terminations, truncations, infos):
        self.after_env_step_calls += 1
        self._step_reward = rewards

    def _rollout_extras(self):
        return {"rollout_value": self._step_value, "step_reward": self._step_reward}

    def _aux_loss(self, data, evaluation, decision_mask):
        aux_loss, terms = super()._aux_loss(data, evaluation, decision_mask)
        stored_value = data.extras["rollout_value"].reshape(-1)
        self.read_extras.append((stored_value, evaluation.old_values))
        drift = (evaluation.values - stored_value).pow(2).mean()
        aux_loss = drift if aux_loss is None else aux_loss + drift
        return aux_loss, {**terms, "value_drift": drift.detach()}


def _rollout_extras_agent(env=None):
    agent = _make_agent(_RolloutExtrasPPO, env if env is not None else _InfoTargetEnv())
    agent.after_env_step_calls = 0
    agent.read_extras = []
    return agent


def test_rollout_extras_are_stored_next_to_info_extras_and_read_by_the_aux_loss():
    agent = _rollout_extras_agent()
    _collect_window(agent)
    buffer = agent.rollout_buffer
    assert agent.after_env_step_calls == NUM_STEPS
    assert set(buffer.extra_specs) == {"target", *ROLLOUT_EXTRA_SPECS}
    assert torch.equal(buffer.extras["target"], buffer.obs["state"])  # read from infos
    assert torch.equal(buffer.extras["rollout_value"][..., 0], buffer.values)
    assert torch.equal(buffer.extras["step_reward"][..., 0], buffer.rewards)

    del agent.train  # _collect_window replaced it on the instance
    metrics = agent.train()
    assert len(agent.read_extras) == agent.update_epochs * agent.num_minibatches
    for stored_value, old_values in agent.read_extras:
        assert torch.equal(stored_value, old_values)
    assert metrics["aux/value_drift"] >= 0


class _UndeclaredExtraPPO(PPO):
    def _rollout_extras(self):
        return {"surprise": torch.zeros(self.num_envs)}


class _UnreturnedExtraPPO(PPO):
    def _rollout_extra_specs(self):
        return {**super()._rollout_extra_specs(), "rollout_value": ((1,), torch.float32)}


def test_rollout_extras_must_be_declared_and_returned():
    with pytest.raises(ValueError, match=re.escape("_rollout_extras() returned ['surprise']")):
        _collect_window(_make_agent(_UndeclaredExtraPPO))
    with pytest.raises(ValueError, match="'rollout_value'.*not returned by _rollout_extras"):
        _collect_window(_make_agent(_UnreturnedExtraPPO))
    # A policy's own spec missing from infos keeps the env-side message.
    with pytest.raises(ValueError, match="rollout extra 'target' is missing from infos"):
        _collect_window(_rollout_extras_agent(_InfoTargetEnv(drop_target_at_step=3)))


# --- _extra_update_metrics ------------------------------------------------------------------


class _UpdateCountPPO(PPO):
    def _extra_update_metrics(self, stats):
        return {
            "gradient_steps": float(len(stats.losses)),
            "stopped_early": float(stats.stopped_early),
        }


@pytest.mark.parametrize(
    "target_kl, gradient_steps, stopped_early",
    [
        pytest.param(None, 4.0, 0.0, id="full-update"),
        # The seeded window of test_ppo_loop_snapshot.py's "ppo-target_kl"
        # stops at its second minibatch.
        pytest.param(1e-6, 1.0, 1.0, id="target_kl-stop"),
    ],
)
def test_extra_update_metrics_are_returned_and_logged(target_kl, gradient_steps, stopped_early):
    logger = _RecordingLogger()
    agent = _make_agent(_UpdateCountPPO, logger=logger, log_freq=1, target_kl=target_kl)
    (metrics,) = _learn(agent)
    assert metrics["gradient_steps"] == gradient_steps
    assert metrics["stopped_early"] == stopped_early
    assert {"loss", "policy_loss", "value_loss", "clipfrac", "learning_rate"} <= metrics.keys()
    assert logger.scalars["losses/gradient_steps"] == [gradient_steps]
    assert logger.scalars["losses/stopped_early"] == [stopped_early]


# --- rollout_buffer_class / _build_rollout_buffer -------------------------------------------


class _MarkedRolloutBuffer(RolloutBuffer):
    pass


class _MarkedRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    pass


class _MarkedBufferPPO(PPO):
    rollout_buffer_class = _MarkedRolloutBuffer


class _MarkedBufferRecurrentPPO(RecurrentPPO):
    rollout_buffer_class = _MarkedRecurrentRolloutBuffer


@pytest.mark.parametrize(
    "agent_cls, buffer_cls, default_cls, default_buffer_cls, extra_kwargs",
    [
        pytest.param(_MarkedBufferPPO, _MarkedRolloutBuffer, PPO, RolloutBuffer, {}, id="ppo"),
        pytest.param(
            _MarkedBufferRecurrentPPO,
            _MarkedRecurrentRolloutBuffer,
            RecurrentPPO,
            RecurrentRolloutBuffer,
            {"rnn_hidden_size": 8},
            id="gru",
        ),
    ],
)
def test_rollout_buffer_class_is_the_buffer_built_and_filled(
    agent_cls, buffer_cls, default_cls, default_buffer_cls, extra_kwargs
):
    assert type(_make_agent(default_cls, **extra_kwargs).rollout_buffer) is default_buffer_cls
    agent = _make_agent(agent_cls, **extra_kwargs)
    assert type(agent.rollout_buffer) is buffer_cls
    rebuilt = agent._build_rollout_buffer()
    assert type(rebuilt) is buffer_cls and rebuilt is not agent.rollout_buffer
    assert (rebuilt.num_steps, rebuilt.num_envs) == (NUM_STEPS, NUM_ENVS)
    (metrics,) = _learn(agent)
    assert agent.rollout_buffer.full
    assert metrics["value_loss"] > 0


def test_sequence_ppo_still_rejects_num_envs_not_divisible_by_num_minibatches():
    assert NUM_ENVS % 3 != 0
    message = (
        f"RecurrentPPO requires num_envs ({NUM_ENVS}) to be divisible by "
        "num_minibatches (3) for env-axis minibatching."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _make_agent(RecurrentPPO, rnn_hidden_size=8, num_minibatches=3)
