"""SuperMimic smoke and unit tests on ``PointMassTrackingEnv`` (CPU, small networks)."""
from __future__ import annotations

import math

import pytest
import torch
from _supermimic_fakes import PointMassRelativeGoal, PointMassTrackingEnv

from rl_garden.algorithms.model_based_step import ModelBasedStepMixin
from rl_garden.algorithms.supermimic import SuperMimic, updates_per_env_step
from rl_garden.buffers.state_sequence_replay_buffer import HistorySample

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
    "replay_ratio": None,
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


def test_learned_heads_with_relative_goal_source_learn(monkeypatch):
    source = PointMassRelativeGoal()
    agent = make_agent(goal_source=source)
    assert agent.world_model.goal_source is source
    # Record the same state-relative goal that imagination will recompute.
    monkeypatch.setattr(agent.env.unwrapped._mdp, "goal", source.goal)
    recorded = []
    gradient_step = agent._gradient_step

    def recording_gradient_step():
        metrics = gradient_step()
        recorded.append(metrics)
        return metrics

    monkeypatch.setattr(agent, "_gradient_step", recording_gradient_step)
    agent.learn(total_timesteps=200)
    assert agent._global_update > 0
    assert recorded and any("actor/loss" in metrics for metrics in recorded)
    assert all(math.isfinite(value) for metrics in recorded for value in metrics.values())


def test_goal_source_is_adopted_from_env():
    env = PointMassTrackingEnv(NUM_ENVS)
    env.goal_source = PointMassRelativeGoal(ref_length=env.ref_length)
    agent = SuperMimic(env, heads="learned", **SMALL)
    assert agent.world_model.goal_source is env.goal_source


@pytest.mark.parametrize("heads", ["learned", "analytic"])
def test_explicit_goal_source_takes_precedence_over_env(heads):
    env = PointMassTrackingEnv(NUM_ENVS)
    env.goal_source = PointMassRelativeGoal()
    source = PointMassRelativeGoal()
    agent = SuperMimic(env, heads=heads, goal_source=source, **SMALL)
    assert agent.world_model.goal_source is source


def test_replay_ratio_learn_runs_with_fractional_updates_and_finite_metrics(monkeypatch):
    agent = make_agent(replay_ratio=3.0, sim_steps_per_iteration=None, updates_per_iteration=None)
    assert agent.updates_per_env_step == pytest.approx(0.5)
    recorded = []
    train = agent.train

    def recording_train(gradient_steps, compute_info=False):
        metrics = train(gradient_steps, compute_info=compute_info)
        recorded.append(metrics)
        return metrics

    monkeypatch.setattr(agent, "train", recording_train)
    agent.learn(total_timesteps=300)
    assert agent._global_update > 0
    assert any(not metrics for metrics in recorded)
    updates = [metrics for metrics in recorded if metrics]
    assert updates and all("wm/loss" in metrics for metrics in updates)
    assert any("actor/loss" in metrics for metrics in updates)
    assert all(math.isfinite(value) for metrics in updates for value in metrics.values())


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
    agent = make_agent(lr=1e-3, lr_final_ratio=0.03, lr_decay_updates=decay)
    for update in (1, decay // 2, decay, 2 * decay):
        agent._global_update = update
        progress = min(update - 1, decay) / decay
        expected = 1e-3 * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress)))
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


def test_updates_per_env_step_matches_official_sample_reuse():
    assert updates_per_env_step(4096, 16.0, 10, 2048, 7) == 0.4


def test_fractional_update_accumulator(monkeypatch):
    agent = make_agent(
        replay_ratio=3.0, sim_steps_per_iteration=None, updates_per_iteration=None,
        wm_batch_size=10, warmup_updates=0,
    )  # 4 * 3 / (1 * 10 * 3) = 0.4 updates per vector step.
    calls = []

    def recording_train(self, gradient_steps, compute_info=False):
        assert self is agent
        calls.append((gradient_steps, compute_info))
        return {"updates": float(gradient_steps)}

    monkeypatch.setattr(ModelBasedStepMixin, "train", recording_train)
    assert agent.updates_per_env_step == pytest.approx(0.4)
    assert agent.training_freq == NUM_ENVS and agent.steps_per_env == 1
    assert agent.grad_steps_per_iteration == 1
    assert agent.utd == pytest.approx(0.4 / NUM_ENVS)
    results = [agent.train(1, compute_info=True) for _ in range(5)]
    assert results[:2] == [{}, {}]
    assert calls == [(1, True)] * 2
    for _ in range(5):
        agent.train(1, compute_info=True)
    assert calls == [(1, True)] * 4
    assert agent._update_credit == pytest.approx(0.0, abs=1e-14)

    assert agent.train(5) == {"updates": 2.0}
    assert calls[-1] == (2, False)


def test_legacy_update_to_data_mapping(monkeypatch):
    agent = make_agent(sim_steps_per_iteration=5, updates_per_iteration=3)
    assert agent.grad_steps_per_iteration == 3
    assert agent.steps_per_env == 5
    assert agent.training_freq == 5 * NUM_ENVS
    assert agent.utd == pytest.approx(3 / (5 * NUM_ENVS))
    assert agent.updates_per_env_step == pytest.approx(3 / 5)
    calls = []
    monkeypatch.setattr(
        ModelBasedStepMixin, "train",
        lambda self, gradient_steps, compute_info=False: calls.append((gradient_steps, compute_info)) or {},
    )
    assert agent.train(3, compute_info=True) == {}
    assert calls == [(3, True)] and agent._update_credit == 0.0


def test_replay_ratio_above_one_keeps_one_scheduling_step(monkeypatch):
    agent = make_agent(replay_ratio=18.0, sim_steps_per_iteration=None, updates_per_iteration=None)
    assert agent.updates_per_env_step == 3.0
    assert agent.grad_steps_per_iteration == 1
    calls = []
    monkeypatch.setattr(
        ModelBasedStepMixin, "train",
        lambda self, gradient_steps, compute_info=False: calls.append(gradient_steps) or {},
    )
    agent.train(1)
    assert calls == [3]


@pytest.mark.parametrize("warmup_updates", [0, 1, 3])
def test_replay_ratio_buffer_warmup_and_metadata(warmup_updates):
    agent = make_agent(
        replay_ratio=3.0, sim_steps_per_iteration=None, updates_per_iteration=None,
        wm_batch_size=10, warmup_updates=warmup_updates, buffer_size=None, learning_starts=None,
    )
    assert agent.buffer_size == 1000 * NUM_ENVS
    assert agent.learning_starts == (agent.wm_window + 2) * NUM_ENVS
    assert agent.initial_training_phase.duration_steps == math.ceil(warmup_updates / 0.4) * NUM_ENVS
    metadata = agent._checkpoint_metadata()
    assert metadata["replay_ratio"] == 3.0
    assert metadata["updates_per_env_step"] == pytest.approx(0.4)
    assert metadata["max_step_reward"] is None
    assert "_update_credit" not in agent._training_state_dict()
    assert "_update_credit" not in agent._extra_checkpoint_state()


@pytest.mark.parametrize(
    "overrides",
    [
        {"replay_ratio": 16.0},
        {"replay_ratio": 16.0, "sim_steps_per_iteration": None},
        {"replay_ratio": 16.0, "updates_per_iteration": None},
        {"sim_steps_per_iteration": None},
        {"updates_per_iteration": None},
        {"sim_steps_per_iteration": 0},
        {"updates_per_iteration": 0},
    ],
)
def test_schedule_parameters_must_select_one_mode(overrides):
    with pytest.raises(ValueError, match="replay_ratio"):
        make_agent(**overrides)


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_replay_ratio_must_be_finite_and_positive(ratio):
    with pytest.raises(ValueError, match="replay_ratio"):
        updates_per_env_step(NUM_ENVS, ratio, 1, 8, 2)


def test_default_learning_rate_is_constant_and_legacy_warmup_unchanged():
    agent = make_agent(warmup_updates=3)
    assert agent.initial_training_phase.duration_steps == 3 * agent.training_freq
    for update in (1, agent.lr_decay_updates // 2, 2 * agent.lr_decay_updates):
        agent._global_update = update
        assert agent._set_learning_rates() == agent.lr


def test_policy_mlps_and_unnormalized_critic_input():
    agent = make_agent()
    for mlp in (agent.policy.goal_encoder, agent.policy.decoder, agent.policy.critic):
        for offset in range(0, len(mlp) - 1, 3):
            assert isinstance(mlp[offset], torch.nn.Linear)
            assert isinstance(mlp[offset + 1], torch.nn.LayerNorm)
            assert isinstance(mlp[offset + 2], torch.nn.GELU)
    assert not hasattr(agent.policy, "critic_norm")
    critic_in = torch.randn(3, agent.policy.critic_in_dim)
    assert torch.equal(agent.policy.value(critic_in), agent.policy.critic(critic_in))


@pytest.mark.parametrize("max_step_reward", [None, 0.01])
def test_max_step_reward_caps_imagined_bootstrap(monkeypatch, max_step_reward):
    agent = make_agent(gamma=0.9, max_step_reward=max_step_reward)
    obs, _ = agent.env.reset(seed=0)
    starts = HistorySample(
        obs={key: value.unsqueeze(0) for key, value in obs.items()},
        prev_action=torch.zeros(NUM_ENVS, 2),
        obs_valid=torch.ones(1, NUM_ENVS, dtype=torch.bool),
    )
    monkeypatch.setattr(agent.policy, "value", lambda inputs: inputs.new_full((*inputs.shape[:-1], 1), 4.0))
    monkeypatch.setattr(
        agent.world_model, "termination_prob", lambda state: state["s"].new_zeros((*state["s"].shape[:-1], 1))
    )
    monkeypatch.setattr(
        agent.world_model, "transition_reward",
        lambda before, action, after, prev: action.new_full((*action.shape[:-1], 1), 0.2),
    )
    rollout = agent._imagine(starts)
    bootstrap = 4.0 if max_step_reward is None else 0.1
    expected = sum(0.9**t * 0.2 for t in range(agent.horizon)) + 0.9**agent.horizon * bootstrap
    assert float(rollout["value_H"]) == pytest.approx(bootstrap)
    assert float(rollout["loss"]) == pytest.approx(-expected)


@pytest.mark.parametrize("max_step_reward", [None, 0.01])
def test_max_step_reward_caps_critic_targets_after_recursion(max_step_reward):
    agent = make_agent(gamma=0.9, lam=0.8, max_step_reward=max_step_reward)
    rewards = torch.tensor([[0.2, -2.0], [0.3, -3.0]])
    dones = torch.tensor([[0.0, 0.25], [0.5, 1.0]])
    values = torch.tensor([[7.0, 8.0], [4.0, 1.0], [5.0, 2.0]])
    # Last: [.3 + .9 * .5 * 5, -3] = [2.55, -3].
    # First: [.2 + .9 * (.2 * 4 + .8 * 2.55),
    #         -2 + .9 * .75 * (.2 * 1 + .8 * -3)] = [2.756, -3.485].
    expected = torch.tensor([[2.756, -3.485], [2.55, -3.0]])
    if max_step_reward is not None:
        expected = expected.clamp(max=0.1)
    torch.testing.assert_close(agent._critic_targets(rewards, dones, values), expected)


@pytest.mark.parametrize(
    "max_step_reward, gamma", [(float("nan"), 0.9), (float("inf"), 0.9), (0.01, 1.0)]
)
def test_value_cap_requires_a_finite_discounted_bound(max_step_reward, gamma):
    with pytest.raises(ValueError, match="max_step_reward"):
        make_agent(max_step_reward=max_step_reward, gamma=gamma)


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
