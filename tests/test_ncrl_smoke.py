"""NCRL stage-2 smoke tests on a fake 64x64 pixel env, a synthetic converted
prior dataset, a tiny ``RSSMPretrain`` checkpoint and its retrieval DB:
retrieval -> BC training -> BC warmup -> guidance windows -> updates; the
action-source rules; gradient isolation (dynamics actor loss, detached reward
head); optimizer ownership; DV2 lambda-return indexing; checkpoint round trip;
the two ablation switches."""
from __future__ import annotations

import math

import pytest
import torch

from _ncrl_online_synthetic import (
    ENV_ACTION_DIM,
    PADDED_ACTION_DIM,
    PixelVecEnv,
    make_pretrain_artifacts,
    tiny_ncrl_kwargs,
)

from rl_garden.algorithms.ncrl import (
    NCRL,
    _ACTION_FROM_ACTOR,
    _ACTION_FROM_BC,
    _ACTION_FROM_RANDOM,
    dv2_lambda_return,
    load_retrieval_db,
)
from rl_garden.observations.schema import ObservationContractError


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    return make_pretrain_artifacts(tmp_path_factory.mktemp("ncrl_smoke"))


def _agent(artifacts, num_envs: int = 1, eval_env=None, **overrides) -> NCRL:
    return NCRL(PixelVecEnv(num_envs=num_envs), eval_env=eval_env, **tiny_ncrl_kwargs(artifacts, **overrides))


def _prepared_agent(artifacts, **overrides) -> NCRL:
    """Retrieval + BC done, rollout state initialised, past warmup."""
    agent = _agent(artifacts, **overrides)
    agent._on_training_start(0)
    obs, _ = agent.env.reset()
    agent._on_env_reset(obs)
    agent._global_step = agent.warmup_steps + 1
    return agent


def _finite(metrics: dict[str, float]) -> None:
    assert metrics
    for key, value in metrics.items():
        assert math.isfinite(value), f"{key}={value}"


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


def test_full_pipeline_learns(artifacts):
    torch.manual_seed(0)
    agent = _agent(artifacts)
    agent.learn(total_timesteps=90)
    keys = agent._retrieved_episode_keys
    assert len(keys) == len(set(keys)) == 3
    assert all(embodiment == "walker" for embodiment, _ in keys)  # action dim 6 only
    assert agent.replay_buffer.prior_size == sum(
        {0: 14, 1: 16, 2: 12, 3: 15}[ep] for _, ep in agent._retrieved_episode_keys
    )
    assert agent._bc_trained
    assert agent._global_update > 0
    # Warmup (50 steps) is all BC; guided steps keep counting afterwards.
    assert int(agent._num_guided_steps) >= agent.warmup_steps
    _finite(agent.train(2))
    for action in agent.env.actions_seen:
        assert action.shape == (1, ENV_ACTION_DIM)


def test_replay_ratio_matches_train_ratio(artifacts):
    agent = _agent(artifacts, train_ratio=6.0)  # utd = 6 / (4 * 6) = 0.25
    assert (agent.steps_per_env, agent.training_freq, agent.grad_steps_per_iteration) == (4, 4, 1)
    agent = _agent(artifacts, num_envs=2, train_ratio=6.0)
    assert (agent.steps_per_env, agent.training_freq, agent.grad_steps_per_iteration) == (2, 4, 1)
    agent = _agent(artifacts, num_envs=2, train_ratio=48.0)  # utd = 2
    assert (agent.steps_per_env, agent.training_freq, agent.grad_steps_per_iteration) == (1, 2, 4)


def test_default_cadence_from_episode_length(artifacts):
    agent = _agent(artifacts, learning_starts=None, warmup_frames=15_000, episode_frames=1000, action_repeat=2)
    assert agent.episode_steps == 500
    assert agent.warmup_steps == 7_500
    assert agent.learning_starts == 5_000


# ----------------------------------------------------------------------
# Rollout action sources
# ----------------------------------------------------------------------


def test_warmup_actions_come_from_bc(artifacts):
    agent = _prepared_agent(artifacts, num_envs=2)
    agent._global_step = 1
    action, env_action, context = agent._rollout_action(agent.env.reset()[0], learning_has_started=False)
    assert (context["action_source"] == _ACTION_FROM_BC).all()
    torch.testing.assert_close(action, agent.bc_guide.act(agent._bc_frames).clamp(-1, 1))
    assert env_action.shape == (2, ENV_ACTION_DIM)


def test_guidance_window_returns_bc_action_and_advances_carry_with_it(artifacts):
    agent = _prepared_agent(artifacts)
    obs, _ = agent.env.reset()
    not_done = torch.zeros(1, dtype=torch.bool)
    agent._rollout_action(obs, True)  # episode step 0: windows resampled
    agent._post_rollout_step(None, not_done, not_done, {})
    agent._guidance_start.fill_(0.0)
    agent._guidance_width.fill_(5.0)  # guide 0 < t < 5

    obs, *_ = agent.env.step(torch.zeros(1, ENV_ACTION_DIM))
    action, env_action, context = agent._rollout_action(obs, True)  # episode step 1
    agent._post_rollout_step(None, not_done, not_done, {})
    assert (context["action_source"] == _ACTION_FROM_BC).all()
    expected = agent.bc_guide.act(agent._bc_frames).clamp(-1, 1)
    torch.testing.assert_close(action, expected)
    assert torch.equal(agent._rollout_state["prev_action"], action)  # next observe() uses the BC action
    assert torch.equal(context["new_state"]["prev_action"], action)
    assert torch.equal(env_action, action[:, :ENV_ACTION_DIM])
    assert torch.all(action[:, ENV_ACTION_DIM:] == 0)
    # The frame queue holds the reset frame twice, then the new frame.
    assert torch.equal(agent._bc_frames[0, 0], agent._bc_frames[0, 1])
    assert torch.equal(agent._bc_frames[0, 2], obs["rgb_default"][0])

    agent._guidance_width.fill_(0.0)
    action, _, context = agent._rollout_action(obs, True)
    assert (context["action_source"] == _ACTION_FROM_ACTOR).all()
    assert torch.all(action[:, ENV_ACTION_DIM:] == 0)  # masked actor sample


def test_window_bounds_are_strict(artifacts):
    agent = _prepared_agent(artifacts)
    agent._guidance_start.fill_(2.0)
    agent._guidance_width.fill_(3.0)
    sources = []
    for step in range(7):
        agent._episode_step.fill_(step)
        sources.append(int(agent._action_source()[0]))
    assert sources == [_ACTION_FROM_ACTOR] * 3 + [_ACTION_FROM_BC] * 2 + [_ACTION_FROM_ACTOR] * 2


def test_guidance_probability_schedule(artifacts):
    agent = _agent(artifacts, guidance_schedule=("linear", 1.0, 0.1, 100))
    agent._global_step = 0
    assert agent.guidance_probability() == pytest.approx(1.0)
    agent._global_step = 50
    assert agent.guidance_probability() == pytest.approx(0.55)
    agent._global_step = 10_000
    assert agent.guidance_probability() == pytest.approx(0.1)


def test_resampled_windows_are_multiples_of_a_tenth_episode(artifacts):
    agent = _prepared_agent(
        artifacts,
        num_envs=64,
        learning_starts=64 * 10,
        buffer_size=64 * 20,
        guidance_schedule=("linear", 1.0, 1.0, 1),
    )
    agent._resample_guidance_windows(torch.ones(64, dtype=torch.bool))
    width, start = agent._guidance_width, agent._guidance_start
    assert set(width.tolist()) <= {float(i) for i in range(1, 11)}  # L = 10 -> W in {1..10}
    assert (start >= 0).all() and (start <= 10 - width).all()


# ----------------------------------------------------------------------
# Gradient isolation / optimizers / loading
# ----------------------------------------------------------------------


def _batch_and_posterior(agent: NCRL):
    agent.learn(total_timesteps=70)
    batch = agent.replay_buffer.sample(agent.batch_size)
    losses, posterior = agent.world_model.model_loss(batch)
    return batch, losses, posterior


def test_dynamics_actor_gradient_reaches_only_the_actor(artifacts):
    agent = _agent(artifacts)
    batch, _, posterior = _batch_and_posterior(agent)
    agent.policy.zero_grad(set_to_none=True)
    imag = agent._imagination_losses(batch, posterior)
    imag["actor_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.actor.parameters())
    assert all(p.grad is None for p in agent.world_model.parameters())
    assert all(p.grad is None for p in agent.policy.slow_critic.parameters())
    assert all(p.grad is None for p in agent.policy.critic.parameters())
    assert all(p.requires_grad for p in agent.world_model.parameters())  # restored after imagination

    agent.policy.zero_grad(set_to_none=True)
    imag["critic_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.critic.parameters())
    assert all(p.grad is None for p in agent.policy.actor.parameters())
    assert all(p.grad is None for p in agent.world_model.parameters())


def test_reward_head_gradient_does_not_reach_the_encoder(artifacts):
    agent = _agent(artifacts)
    batch, losses, _ = _batch_and_posterior(agent)
    assert batch.reward_valid.sum() == 3  # round(0.75 * 4) online columns
    agent.policy.zero_grad(set_to_none=True)
    losses["rew"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.world_model.reward_head.parameters())
    reward_ids = {id(p) for p in agent.world_model.reward_head.parameters()}
    assert all(p.grad is None for p in agent.world_model.parameters() if id(p) not in reward_ids)


def test_optimizer_ownership(artifacts):
    agent = _agent(artifacts)
    groups = {
        name: {id(p) for group in getattr(agent, name).param_groups for p in group["params"]}
        for name in agent._optimizer_names()
    }
    world_ids = {id(p) for p in agent.world_model.parameters()}
    reward_ids = {id(p) for p in agent.world_model.reward_head.parameters()}
    assert groups["model_optimizer"] == world_ids - reward_ids
    assert groups["reward_optimizer"] == reward_ids
    assert groups["actor_optimizer"] == {id(p) for p in agent.policy.actor.parameters()}
    assert groups["critic_optimizer"] == {id(p) for p in agent.policy.critic.parameters()}
    slow_ids = {id(p) for p in agent.policy.slow_critic.parameters()}
    assert not slow_ids & set().union(*groups.values())


def test_pretrained_rssm_weights_are_loaded(artifacts):
    agent = _agent(artifacts)
    pretrained = torch.load(artifacts["checkpoint"], weights_only=False)["state"]["policy"]
    loaded = agent.policy.state_dict()
    rssm_keys = [k for k in pretrained if k.startswith("actor_extractor.")]
    assert rssm_keys
    for key in rssm_keys:
        assert torch.equal(loaded[key], pretrained[key]), key
    assert any(k.startswith("actor_extractor.reward_head.") for k in loaded)
    assert agent.padded_action_dim == PADDED_ACTION_DIM


def test_dv2_lambda_return_matches_official_recursion():
    """Port of ``3rd_party/ncrl/src/dreamer/dreamer_utils.py::lambda_return``
    called as ``lambda_return(reward[:-1], value[:-1], disc, bootstrap=value[-1])``."""
    torch.manual_seed(0)
    horizon, n, disc, lam = 5, 7, 0.99, 0.95
    reward = torch.randn(horizon + 1, n, 1)
    value = torch.randn(horizon + 1, n, 1)

    rew, val, boot = reward[:-1], value[:-1], value[-1]
    next_values = torch.cat([val[1:], boot[None]], 0)
    inputs = rew + disc * next_values * (1 - lam)
    last, outputs = boot, []
    for t in reversed(range(horizon)):
        last = inputs[t] + disc * lam * last
        outputs.append(last)
    official = torch.stack(outputs[::-1], 0)

    ours = dv2_lambda_return(reward, value, disc=disc, lam=lam)
    assert ours.shape == (horizon, n, 1)
    torch.testing.assert_close(ours, official)


# ----------------------------------------------------------------------
# Eval / checkpoint / validation
# ----------------------------------------------------------------------


def test_eval_sends_real_width_mean_actions(artifacts):
    eval_env = PixelVecEnv(num_envs=2)
    agent = _agent(artifacts, eval_env=eval_env, num_eval_steps=12)
    agent.learn(total_timesteps=1)
    metrics = agent._evaluate()
    assert math.isfinite(metrics["return"])
    assert all(a.shape == (2, ENV_ACTION_DIM) for a in eval_env.actions_seen)


def test_checkpoint_round_trip_and_resume_skips_retrieval_and_bc(artifacts, tmp_path, monkeypatch):
    agent = _agent(artifacts)
    agent.learn(total_timesteps=70)
    path = agent.save(tmp_path / "ncrl.pt")
    checkpoint = torch.load(path, weights_only=False)
    assert set(checkpoint["state"]["optimizers"]) == {
        "model_optimizer", "reward_optimizer", "actor_optimizer", "critic_optimizer"
    }
    assert checkpoint["metadata"]["hyperparameters"]["pretrained_world_model"] == str(artifacts["checkpoint"])
    assert checkpoint["metadata"]["hyperparameters"]["retrieval_db"] == str(artifacts["retrieval_db"])

    restored = _agent(artifacts, seed=7)
    restored.load(path)
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, restored.policy.state_dict()[key]), key
    for key, value in agent.bc_guide.state_dict().items():
        assert torch.equal(value, restored.bc_guide.state_dict()[key]), key
    assert restored._retrieved_episode_keys == agent._retrieved_episode_keys
    assert int(restored._num_guided_steps) == int(agent._num_guided_steps)
    assert restored._global_step == agent._global_step

    monkeypatch.setattr(restored, "_retrieve_prior_episodes", lambda: pytest.fail("retrieval re-ran on resume"))
    monkeypatch.setattr("rl_garden.algorithms.ncrl.train_bc_guide", lambda *a, **k: pytest.fail("BC retrained"))
    restored._on_training_start(0)
    assert restored.replay_buffer.prior_size == agent.replay_buffer.prior_size


def test_observation_space_must_match_the_pretrained_encoder(artifacts):
    with pytest.raises(ObservationContractError, match="must match the pretrained encoder"):
        NCRL(PixelVecEnv(extra_state=True), **tiny_ncrl_kwargs(artifacts))


def test_retrieval_db_from_another_encoder_is_rejected(artifacts, tmp_path):
    db = load_retrieval_db(artifacts["retrieval_db"])
    db["metadata"]["encoder_fingerprint"] += 1.0
    other = tmp_path / "other_db.pt"
    torch.save(db, other)
    with pytest.raises(ValueError, match="different encoder"):
        _agent(artifacts, retrieval_db=str(other))


def test_save_replay_buffer_is_rejected(artifacts):
    with pytest.raises(NotImplementedError, match="replay buffer"):
        _agent(artifacts, save_replay_buffer=True)


# ----------------------------------------------------------------------
# Ablations
# ----------------------------------------------------------------------


def test_ablation_no_rehearsal_disables_retrieval_and_guidance(artifacts):
    agent = _agent(artifacts, online_ratio=1.0, retrieval_db=None)
    assert not agent.guidance_enabled and agent.bc_guide is None
    agent.learn(total_timesteps=70)
    assert agent.replay_buffer.prior is None
    assert agent._global_update > 0
    batch = agent.replay_buffer.sample(agent.batch_size)
    assert batch.reward_valid.all()
    agent._global_step = 1
    _, _, context = agent._rollout_action(agent.env.reset()[0], False)
    assert (context["action_source"] == _ACTION_FROM_RANDOM).all()


def test_ablation_no_guidance_keeps_rehearsal(artifacts):
    agent = _agent(artifacts, use_guidance=False)
    assert agent.bc_guide is None
    agent.learn(total_timesteps=70)
    assert agent.replay_buffer.prior_size > 0
    assert int(agent._num_guided_steps) == 0
    assert agent.replay_buffer.sample(agent.batch_size).reward_valid.tolist() == [True] * 3 + [False]
