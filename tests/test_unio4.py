from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec, UniO4
from rl_garden.algorithms.bppo import BPPOCriticMixin
from rl_garden.algorithms.offline import OfflineRLAlgorithm
from rl_garden.algorithms.ppo import ppo_clip_policy_loss
from rl_garden.policies.unio4_mixture_policy import UniO4MixturePolicy


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)}),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


class _CompletedEpisodeEvalEnv:
    """Every step immediately completes an episode, mirroring the fixture
    already used in tests/test_offline_algorithm.py."""

    num_envs = 2
    single_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        return torch.zeros(2, 4), {}

    def step(self, actions):
        del actions
        base = 2 * self.step_idx
        self.step_idx += 1
        return (
            torch.zeros(2, 4),
            torch.zeros(2),
            torch.ones(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {
                "final_info": {
                    "episode": {
                        "r": torch.tensor([base + 1.0, base + 2.0]),
                        "l": torch.tensor([10.0, 20.0]),
                    }
                },
                "_final_info": torch.tensor([True, True]),
            },
        )


def _make_agent(**kwargs) -> UniO4:
    defaults = dict(
        env=_state_env(),
        buffer_size=256,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        critic_warmup_steps=3,
        bc_ensemble_steps=3,
        num_policies=3,
        value_hidden_dims=(16,),
        q_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        target_update_freq=1,
    )
    defaults.update(kwargs)
    return UniO4(**defaults)


def _fill(agent: UniO4, steps: int = 40, episode_len: int = 20) -> None:
    env = agent.env
    for t in range(steps):
        obs = torch.rand(env.num_envs, *env.single_observation_space.shape) * 2 - 1
        next_obs = torch.rand_like(obs) * 2 - 1
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.rand(env.num_envs)
        dones = torch.zeros(env.num_envs, dtype=torch.bool)
        episode_end = torch.zeros(env.num_envs, dtype=torch.bool)
        if (t + 1) % episode_len == 0:
            episode_end[:] = True
        agent.replay_buffer.add(
            obs, next_obs, actions, rewards, dones, episode_end=episode_end
        )


def test_rejects_dict_observation_space():
    with pytest.raises(TypeError):
        UniO4(env=_dict_env(), buffer_device="cpu", device="cpu")


def test_registry_discovers_unio4():
    from rl_garden.training.offline._registry import registry

    registry.discover()
    assert "unio4" in registry._entries


def test_unio4_mro_shares_bppo_critic_mixin():
    assert [c.__name__ for c in UniO4.__mro__][:3] == [
        "UniO4",
        "BPPOCriticMixin",
        "OfflineRLAlgorithm",
    ]
    assert issubclass(UniO4, BPPOCriticMixin)
    assert issubclass(UniO4, OfflineRLAlgorithm)


def test_policy_is_mixture_wrapper_over_actors():
    agent = _make_agent()
    assert isinstance(agent.policy, UniO4MixturePolicy)
    assert len(agent.policy.actors) == agent.num_policies
    for actor, policy_actor in zip(agent.actors, agent.policy.actors):
        assert actor is policy_actor  # same objects, not copies


def test_old_actors_start_synced_to_actors():
    agent = _make_agent()
    for actor, old_actor in zip(agent.actors, agent.old_actors):
        for p, q in zip(actor.parameters(), old_actor.parameters()):
            assert torch.equal(p, q)
        assert not any(p.requires_grad for p in old_actor.parameters())


# --- Phase CRITIC ---


def test_phase_critic_shared_not_per_member():
    """Uni-O4 shares ONE critic across the ensemble (verified against
    3rd_party/Uni-O4/main.py) -- no per-member value_net/q_net exist."""
    agent = _make_agent(critic_warmup_steps=5)
    assert hasattr(agent, "value_net") and hasattr(agent, "q_net")
    assert not hasattr(agent, "value_net_0")


def test_phase_critic_only_updates_critic_not_actors():
    agent = _make_agent(critic_warmup_steps=5)
    _fill(agent)
    actors_before = [[p.clone() for p in a.parameters()] for a in agent.actors]

    info = agent.train(2, compute_info=True)

    assert agent._phase_step == 2
    assert agent._critic_step == 2
    assert "value_loss" in info and np.isfinite(info["value_loss"])
    assert "q_loss" in info and np.isfinite(info["q_loss"])
    for before, actor in zip(actors_before, agent.actors):
        assert all(torch.equal(a, b) for a, b in zip(before, actor.parameters()))


def test_critic_step_matches_phase_step_only_through_phase_critic():
    """_critic_step (owned by BPPOCriticMixin) and _phase_step (UniO4's own
    3-phase gate) coincide during Phase CRITIC, then diverge once Phase
    BC_ENSEMBLE starts (the critic stops updating, _phase_step keeps
    advancing) -- this is exactly the scenario BPPOCriticMixin's dedicated
    counter was extracted to handle correctly."""
    agent = _make_agent(critic_warmup_steps=3, bc_ensemble_steps=3)
    _fill(agent)
    for step in range(3):
        agent.train(1)
        assert agent._critic_step == agent._phase_step == step + 1
    for _ in range(3):
        agent.train(1)  # Phase BC_ENSEMBLE
    assert agent._critic_step == 3  # frozen
    assert agent._phase_step == 6  # kept advancing


# --- Phase BC_ENSEMBLE ---


def test_bc_ensemble_loss_matches_joint_loss_formula():
    """Pin the exact loss formula from BC_ensemble.joint_train /
    BehaviorCloning.joint_loss (bc_kl='data', kl_type='heuristic'), not a
    gradient-magnitude story: with max_prob_a detached, the per-sample
    gradient coefficient is uniform; the diversity mechanism lives in the
    loss *value* via the +alpha_bc*max_prob_a additive term."""
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=5, num_policies=2)
    _fill(agent)
    agent.train(1)  # exhaust Phase CRITIC

    torch.manual_seed(0)
    data = agent._sample_train_batch()
    with torch.no_grad():
        log_prob_0 = agent.actors[0].actor.evaluate_action_log_prob(
            agent.actors[0].extract_features(data.obs), data.actions
        ).squeeze(-1)
        log_prob_1 = agent.actors[1].actor.evaluate_action_log_prob(
            agent.actors[1].extract_features(data.obs), data.actions
        ).squeeze(-1)
    bc_loss_0 = -log_prob_0
    expected_loss_0 = (
        (1 + agent.alpha_bc) * bc_loss_0.mean() + agent.alpha_bc * log_prob_1.mean()
    )

    torch.manual_seed(0)
    data2 = agent._sample_train_batch()
    info = agent._bc_ensemble_step(data2)

    assert info["bc_loss_0"] == pytest.approx(bc_loss_0.mean().item(), rel=1e-4)
    assert info["joint_loss_0"] == pytest.approx(expected_loss_0.item(), rel=1e-4)


def test_bc_ensemble_single_policy_falls_back_to_plain_bc():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=3, num_policies=1)
    _fill(agent)
    agent.train(1)  # Phase CRITIC
    before = [p.clone() for p in agent.actors[0].parameters()]
    info = agent.train(1, compute_info=True)
    assert "bc_loss_0" in info and np.isfinite(info["bc_loss_0"])
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actors[0].parameters()))


def test_phase_bc_ensemble_leaves_old_actors_untouched():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=5, num_policies=2)
    _fill(agent)
    agent.train(1)  # Phase CRITIC
    old_before = [[p.clone() for p in oa.parameters()] for oa in agent.old_actors]
    agent.train(3)  # Phase BC_ENSEMBLE
    for before, old_actor in zip(old_before, agent.old_actors):
        assert all(torch.equal(a, b) for a, b in zip(before, old_actor.parameters()))


# --- Phase IMPROVE ---


def test_phase_transition_resyncs_all_old_actors_then_updates_actors():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=2)
    _fill(agent)
    agent.train(2)  # exhaust CRITIC + BC_ENSEMBLE
    actors_before = [[p.clone() for p in a.parameters()] for a in agent.actors]

    info = agent.train(1, compute_info=True)

    assert agent._phase_b_step == 1
    assert "actor_loss_0" in info and "actor_loss_1" in info
    for i, before in enumerate(actors_before):
        assert any(
            not torch.equal(a, b) for a, b in zip(before, agent.actors[i].parameters())
        )


def test_clip_ratio_decays_once_per_iteration_not_per_member():
    """The exact bug flagged during planning: incrementing _phase_b_step /
    decaying _clip_ratio inside the per-member loop would burn through
    clip_decay_steps num_policies-x faster than intended."""
    agent = _make_agent(
        critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=4,
        clip_ratio=0.25, clip_decay=0.5, clip_decay_steps=10,
    )
    _fill(agent)
    agent.train(2)  # exhaust CRITIC + BC_ENSEMBLE
    start = agent._clip_ratio
    agent.train(1)
    assert agent._clip_ratio == pytest.approx(start * 0.5)
    assert agent._phase_b_step == 1


def test_each_member_ppo_clip_loss_matches_hand_computation():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=2)
    _fill(agent)
    agent.train(2)  # into Phase IMPROVE boundary
    torch.manual_seed(0)
    data = agent._sample_train_batch()

    with torch.no_grad():
        old_features = agent.old_actors[0].extract_features(data.obs)
        action, old_log_prob = agent.old_actors[0].actor.action_log_prob(old_features)
        advantage = (agent.q_net(data.obs, action) - agent.value_net(data.obs)).squeeze(-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = agent._weighted_advantage(advantage)
    new_features = agent.actors[0].extract_features(data.obs, stop_gradient=False)
    new_log_prob = agent.actors[0].actor.evaluate_action_log_prob(new_features, action)
    ratio = (new_log_prob.squeeze(-1) - old_log_prob.squeeze(-1)).exp()
    expected_clip_loss = ppo_clip_policy_loss(advantage, ratio, agent._clip_ratio)

    torch.manual_seed(0)
    data2 = agent._sample_train_batch()
    info = agent._improve_step(data2)
    assert info["actor_loss_0"] == pytest.approx(expected_clip_loss.item(), rel=1e-4)


# --- mixture policy ---


def test_mixture_policy_picks_higher_self_confidence_member():
    agent = _make_agent(num_policies=2)
    obs = torch.rand(3, 4) * 2 - 1
    action = agent.policy.predict(obs, deterministic=True)
    assert action.shape == (3, 2)

    with torch.no_grad():
        scores = []
        for actor in agent.actors:
            features = actor.extract_features(obs)
            det_action = actor.actor.deterministic_action(features)
            scores.append(
                actor.actor.evaluate_action_log_prob(features, det_action).squeeze(-1)
            )
        winner = torch.stack(scores, dim=0).argmax(dim=0)
        expected = torch.stack(
            [actor.actor.deterministic_action(actor.extract_features(obs)) for actor in agent.actors],
            dim=0,
        )[winner, torch.arange(obs.shape[0])]
    assert torch.allclose(action, expected)


# --- eval / per-member old_actors gating ---


def test_evaluate_runs_n_plus_one_rollouts_with_expected_keys():
    agent = _make_agent(
        num_policies=2, num_eval_episodes=2, num_eval_steps=5,
        eval_env=_CompletedEpisodeEvalEnv(),
    )
    metrics = agent._evaluate()
    assert "return" in metrics
    assert "return_0" in metrics and "return_1" in metrics
    assert isinstance(agent.policy, UniO4MixturePolicy)  # restored after per-member loop


def test_old_actors_sync_independently_per_member():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=2)
    _fill(agent)
    agent.train(3)  # into Phase IMPROVE

    agent._log_eval_metrics({"return": 0.0, "return_0": 10.0, "return_1": -10.0}, step=1)
    assert agent._best_eval_score == [10.0, -10.0]
    assert all(
        torch.equal(p, q)
        for p, q in zip(agent.old_actors[0].parameters(), agent.actors[0].parameters())
    )

    agent.train(1)  # diverge actors again
    agent._log_eval_metrics({"return": 0.0, "return_0": 5.0, "return_1": 0.0}, step=2)
    assert agent._best_eval_score == [10.0, 0.0]
    assert not all(
        torch.equal(p, q)
        for p, q in zip(agent.old_actors[0].parameters(), agent.actors[0].parameters())
    )
    assert all(
        torch.equal(p, q)
        for p, q in zip(agent.old_actors[1].parameters(), agent.actors[1].parameters())
    )


def test_old_actors_sync_skips_nan_member_without_blocking_others():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=2)
    _fill(agent)
    agent.train(3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent._log_eval_metrics({"return": 0.0, "return_1": 3.0}, step=1)  # return_0 missing
        assert any("member 0" in str(w.message) for w in caught)
    assert agent._best_eval_score[0] == float("-inf")
    assert agent._best_eval_score[1] == 3.0


def test_old_actors_sync_noop_during_earlier_phases():
    agent = _make_agent(critic_warmup_steps=100, bc_ensemble_steps=100, num_policies=2)
    _fill(agent)
    agent.train(1)  # still Phase CRITIC
    agent._log_eval_metrics({"return": 0.0, "return_0": 999.0, "return_1": 999.0}, step=1)
    assert agent._best_eval_score == [float("-inf"), float("-inf")]


# --- checkpointing ---


def test_checkpoint_roundtrip_preserves_all_ensemble_state():
    agent = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=2)
    _fill(agent)
    agent.train(3)
    agent._log_eval_metrics({"return": 0.0, "return_0": 5.0, "return_1": -1.0}, step=1)

    with tempfile.TemporaryDirectory() as tmp:
        path = agent.save(os.path.join(tmp, "unio4.pt"))
        fresh = _make_agent(critic_warmup_steps=1, bc_ensemble_steps=1, num_policies=2)
        fresh.load(path, load_replay_buffer=False)

    assert fresh._phase_step == agent._phase_step
    assert fresh._phase_b_step == agent._phase_b_step
    assert fresh._critic_step == agent._critic_step
    assert fresh._best_eval_score == agent._best_eval_score
    assert len(fresh._best_eval_score) == agent.num_policies
    for k, v in agent.policy.state_dict().items():
        assert torch.equal(fresh.policy.state_dict()[k], v), k
    for old_actor, fresh_old_actor in zip(agent.old_actors, fresh.old_actors):
        for k, v in old_actor.state_dict().items():
            assert torch.equal(fresh_old_actor.state_dict()[k], v)
    for k, v in agent.value_net.state_dict().items():
        assert torch.equal(fresh.value_net.state_dict()[k], v)
