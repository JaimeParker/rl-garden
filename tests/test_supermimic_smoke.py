"""SuperMimic smoke and unit tests on ``PointMassTrackingEnv`` (CPU, small networks)."""
from __future__ import annotations

import math

import pytest
import torch
from _supermimic_fakes import PointMassTrackingEnv

from rl_garden.algorithms.supermimic import SuperMimic

NUM_ENVS = 4
SMALL = {
    "wm_hidden_dims": (32, 32),
    "head_hidden_dims": (16,),
    "goal_encoder_dims": (32,),
    "decoder_dims": (32,),
    "critic_dims": (16,),
    "horizon": 4,
    "wm_window": 2,
    "wm_batches": 1,
    "wm_batch_size": 8,
    "imagination_batch": 8,
    "warmup_updates": 1,
    "learning_starts": 4 * NUM_ENVS,
    "sim_steps_per_iteration": 2,
    "updates_per_iteration": 1,
    "buffer_size": 1024,
    "buffer_device": "cpu",
    "device": "cpu",
    "std_log": False,
    "seed": 0,
}


def make_agent(*, heads: str = "learned", extra_keys: bool = True, eval_env=None, **overrides) -> SuperMimic:
    env = PointMassTrackingEnv(NUM_ENVS, extra_keys=extra_keys)
    return SuperMimic(env, eval_env, heads=heads, **{**SMALL, **overrides})


def trained_agent(**kwargs) -> SuperMimic:
    agent = make_agent(**kwargs)
    agent.learn(64)
    return agent


@pytest.mark.parametrize("obs_history_len", [1, 3])
@pytest.mark.parametrize(
    "heads, extra_keys", [("learned", False), ("learned", True), ("analytic", True)]
)
def test_learn_runs_with_finite_metrics(monkeypatch, heads, extra_keys, obs_history_len):
    agent = make_agent(heads=heads, extra_keys=extra_keys, obs_history_len=obs_history_len)
    recorded = []
    gradient_step = agent._gradient_step

    def recording_gradient_step():
        metrics = gradient_step()
        recorded.append(metrics)
        return metrics

    monkeypatch.setattr(agent, "_gradient_step", recording_gradient_step)
    agent.learn(total_timesteps=200)
    assert recorded and any("actor/loss" in metrics for metrics in recorded)
    for metrics in recorded:
        assert "wm/loss" in metrics
        for key, value in metrics.items():
            assert math.isfinite(value), (key, value)


def test_actor_gradient_reaches_decoder_only():
    agent = trained_agent()
    starts = agent._sample_train_batch()["starts"]
    agent.policy.zero_grad(set_to_none=True)
    rollout = agent._imagine(starts)
    rollout["loss"].backward()

    decoder_grads = [p.grad for p in agent.policy.decoder.parameters()]
    assert all(g is not None for g in decoder_grads)
    assert sum(float(g.abs().sum()) for g in decoder_grads) > 0
    for module in (agent.world_model, agent.policy.critic):
        for name, parameter in module.named_parameters():
            assert parameter.grad is None, name
            assert parameter.requires_grad, name


def test_survival_weights_cut_rewards_and_bootstrap_after_termination(monkeypatch):
    agent = trained_agent()
    starts = agent._sample_train_batch()["starts"]

    def terminate_at_second_step(state):
        probs = torch.zeros(*state["s"].shape[:-1], 1)
        probs[1] = 1.0  # d_2 = 1: rows are d_1..d_H
        return probs

    monkeypatch.setattr(agent.world_model, "termination_prob", terminate_at_second_step)
    with torch.no_grad():
        rollout = agent._imagine(starts)
    rewards, gamma = rollout["rewards"], agent.gamma
    survival = torch.tensor([1.0, 1.0, 0.0, 0.0]).unsqueeze(-1)  # c_0..c_3
    expected = (gamma ** torch.arange(4.0).unsqueeze(-1) * survival * rewards).sum(0)
    assert float(rewards[2:].abs().sum()) > 0 and float(rollout["value_H"]) != 0.0
    assert float(rollout["survival_H"]) == 0.0
    torch.testing.assert_close(rollout["loss"], -expected.mean())


def test_critic_targets_match_hand_written_recursion():
    agent = make_agent(gamma=0.9, lam=0.8)
    torch.manual_seed(0)
    horizon, batch = 5, 3
    rewards = torch.randn(horizon, batch)
    dones = torch.rand(horizon, batch)
    values = torch.randn(horizon + 1, batch)
    expected = torch.empty(horizon, batch)
    following = values[horizon]
    for t in reversed(range(horizon)):
        following = rewards[t] + 0.9 * (1 - dones[t]) * ((1 - 0.8) * values[t + 1] + 0.8 * following)
        expected[t] = following
    torch.testing.assert_close(agent._critic_targets(rewards, dones, values), expected)


def _actor_rollout(loss: torch.Tensor) -> dict:
    return {"loss": loss, "rewards": torch.zeros(1), "survival_H": torch.tensor(1.0), "value_H": torch.tensor(0.0)}


def test_actor_step_clips_elements_and_skips_non_finite():
    agent = make_agent()
    bias = agent.policy.decoder[-1].bias
    metrics = agent._actor_step(_actor_rollout(-0.5 * bias[0]))  # raw gradient -0.5, norm 0.5 < 1
    assert float(bias.grad[0]) == pytest.approx(-agent.actor_grad_value)
    assert metrics["actor/grad_norm"] == pytest.approx(0.5)
    assert metrics["actor/skipped"] == 0.0

    before = [p.detach().clone() for p in agent.policy.actor_parameters()]
    metrics = agent._actor_step(_actor_rollout(bias.sum() * float("nan")))
    assert metrics["actor/skipped"] == 1.0
    for old, new in zip(before, agent.policy.actor_parameters()):
        assert torch.equal(old, new)


def test_cosine_learning_rate_schedule():
    decay = 100
    agent = make_agent(lr=1e-3, lr_final_ratio=0.1, lr_decay_updates=decay)
    for update in (1, decay // 2, decay, 2 * decay):
        agent._global_update = update
        progress = min(update - 1, decay) / decay
        expected = 1e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        assert agent._set_learning_rates() == pytest.approx(expected)
        for name in agent._optimizer_names():
            assert getattr(agent, name).param_groups[0]["lr"] == pytest.approx(expected)
    agent._global_update = 1
    assert agent._set_learning_rates() == pytest.approx(1e-3)


def test_warmup_updates_train_only_the_world_model():
    agent = make_agent(warmup_updates=3)
    agent.learn(16)  # fills the buffer; the phase starts at step 0
    actor_critic = agent.policy.actor_parameters() + agent.policy.critic_parameters()
    for update in range(5):
        agent._global_step = update * agent.training_freq
        before = [p.detach().clone() for p in actor_critic]
        metrics = agent._gradient_step()
        assert "wm/loss" in metrics
        unchanged = all(torch.equal(old, new) for old, new in zip(before, actor_critic))
        if update < 3:
            assert "actor/loss" not in metrics and unchanged
        else:
            assert "actor/loss" in metrics and "critic/loss" in metrics and not unchanged


def test_update_to_data_mapping():
    agent = make_agent(sim_steps_per_iteration=5, updates_per_iteration=3)
    assert agent.grad_steps_per_iteration == 3
    assert agent.steps_per_env == 5
    assert agent.training_freq == 5 * NUM_ENVS


def test_terminated_steps_keep_the_pre_reset_observation():
    """The off-policy base class stores the pre-reset observation on every
    episode end; SuperMimic's strict windows read it from the final-obs table."""
    agent = trained_agent()
    assert agent.bootstrap_at_done == "truncated"
    buffer, mdp = agent.replay_buffer, agent.env.unwrapped.analytic_mdp
    terminal = buffer.episode_ends & buffer.dones & (buffer._final_slot_ids >= 0)
    assert bool(terminal.any())
    final = buffer._final_obs[buffer._final_slot_ids[terminal]]
    t = final["state_time"].squeeze(-1).long()
    distance = (final["state_sim"][:, :2] - mdp.ref(t)).norm(dim=-1)
    assert bool((distance > 1.0).all())


def test_checkpoint_round_trip(tmp_path):
    agent = trained_agent()
    path = agent.save(tmp_path / "supermimic.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["hyperparameters"]["horizon"] == 4
    assert any(key.startswith("world_model.") for key in payload["state"]["policy"])
    assert set(payload["state"]["optimizers"]) == {"world_optimizer", "actor_optimizer", "critic_optimizer"}

    restored = make_agent(seed=1)
    restored.load(path, load_replay_buffer=False)
    restored_policy = restored.policy.state_dict()
    for key, value in agent.policy.state_dict().items():
        torch.testing.assert_close(restored_policy[key], value)
    for name in agent._optimizer_names():
        expected = getattr(agent, name).state_dict()["state"]
        actual = getattr(restored, name).state_dict()["state"]
        assert expected.keys() == actual.keys() and expected
        for index, entry in expected.items():
            for field, value in entry.items():
                torch.testing.assert_close(actual[index][field], value)
    assert restored._global_update == agent._global_update


def test_evaluation_resets_history_after_done(monkeypatch):
    agent = make_agent(eval_env=PointMassTrackingEnv(2, seed=1), num_eval_steps=40, obs_history_len=3)
    eval_action = agent._eval_action
    flagged = []

    def recording_eval_action(obs):
        pending = None if agent._eval_hist_rebuild else agent._eval_hist_reset.clone()
        action = eval_action(obs)
        if pending is not None and bool(pending.any()):
            flagged.append((pending, obs["state"].clone(), agent._eval_hist.clone()))
        return action

    monkeypatch.setattr(agent, "_eval_action", recording_eval_action)
    metrics = agent._evaluate()
    assert "return" in metrics
    assert flagged
    for rows, frame, history in flagged:
        assert torch.equal(history[rows], frame[rows].unsqueeze(1).expand(-1, 3, -1))


def test_analytic_heads_require_an_analytic_mdp():
    with pytest.raises(ValueError, match="AnalyticMDP"):
        make_agent(heads="analytic", extra_keys=False)


def test_analytic_heads_require_the_time_key():
    env = PointMassTrackingEnv(NUM_ENVS, time_key=False)
    assert "state_time" not in env.single_observation_space.spaces and hasattr(env, "analytic_mdp")
    with pytest.raises(ValueError, match="state_time"):
        SuperMimic(env, heads="analytic", **SMALL)


def test_imagination_uses_the_stored_start_history_without_extra_noise(monkeypatch):
    agent = trained_agent(heads="analytic", obs_history_len=3)
    assert agent.world_model.mdp.obs_noise is not None
    starts = agent._sample_train_batch()["starts"]
    act = agent.policy.act
    first_histories = []

    def recording_act(obs_hist, goal):
        if not first_histories:
            first_histories.append(obs_hist.detach().clone())
        return act(obs_hist, goal)

    monkeypatch.setattr(agent.policy, "act", recording_act)
    with torch.no_grad():
        agent._imagine(starts)
    assert torch.equal(first_histories[0], starts.obs["state"].float().transpose(0, 1))


def test_analytic_starts_reject_invalid_rows(monkeypatch):
    agent = trained_agent(heads="analytic")
    mdp = agent.world_model.mdp
    monkeypatch.setattr(mdp, "valid_start", lambda t, horizon: t % 2 == 0)
    starts = agent._sample_imagination_starts()
    t = starts.obs["state_time"][-1].squeeze(-1).long()
    assert t.shape == (agent.imagination_batch,) and bool((t % 2 == 0).all())
    assert starts.prev_action.shape[0] == agent.imagination_batch

    monkeypatch.setattr(mdp, "valid_start", lambda t, horizon: torch.zeros_like(t, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="valid imagination starts"):
        agent._sample_imagination_starts()
