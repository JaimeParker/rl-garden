from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import BC, BPPO, OfflineEnvSpec
from rl_garden.observations import ObservationContractError, ObsGroups


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


def _dict_image_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "rgb_cam": spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> BPPO:
    defaults = dict(
        env=_state_env(),
        buffer_size=256,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        critic_warmup_steps=3,
        value_hidden_dims=(16,),
        q_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        target_update_freq=1,
    )
    defaults.update(kwargs)
    return BPPO(**defaults)


def _fill(agent: BPPO, steps: int = 40, episode_len: int = 20) -> None:
    env = agent.env
    # env.single_observation_space is Dict({"state": Box}) -- a bare Box env
    # (as _state_env() constructs) is boundary-normalized by
    # BaseAlgorithm.__init__ (rl_garden.envs.wrappers.VectorizedDictStateWrapper).
    obs_shape = env.single_observation_space["state"].shape
    for t in range(steps):
        state = torch.rand(env.num_envs, *obs_shape) * 2 - 1
        next_state = torch.rand_like(state) * 2 - 1
        obs = {"state": state}
        next_obs = {"state": next_state}
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.rand(env.num_envs)
        dones = torch.zeros(env.num_envs, dtype=torch.bool)
        episode_end = torch.zeros(env.num_envs, dtype=torch.bool)
        if (t + 1) % episode_len == 0:
            episode_end[:] = True
        agent.replay_buffer.add(
            obs, next_obs, actions, rewards, dones, episode_end=episode_end
        )


def test_accepts_state_only_dict_observation_space():
    # BPPO's replay buffer/critic are Dict-based now -- a pure-state Dict env
    # (no images) is exactly what a bare Box env normalizes to at the
    # boundary, so it must construct identically to _state_env().
    agent = BPPO(env=_dict_env(), buffer_device="cpu", device="cpu")
    assert agent.observation_encoders.schema.keys == ("state",)


def test_rejects_image_observation_space():
    # BPPO is state-only by construction (BPPOCriticMixin._setup_observation_
    # encoders's has_images guard): value_net/q_net/BCPolicy all assume a
    # flat feature vector, with no Dict/image handling anywhere.
    with pytest.raises(ObservationContractError, match="images"):
        BPPO(env=_dict_image_env(), buffer_device="cpu", device="cpu")


def _multi_key_state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "state_object_pose": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def test_multi_key_state_observation_trains_one_step():
    # Regression test: value_net/q_net used to be sized from the actor
    # extractor's width (which concatenates every state_<name> key) but read
    # a raw data.obs["state"] tensor at forward time -- a shape mismatch
    # crash whenever more than one state key was present. Both nets now read
    # through the critic-role extractor (BPPOCriticMixin._build_critic/
    # _phase_a_step), which performs the same concatenation.
    agent = BPPO(
        env=_multi_key_state_env(),
        buffer_size=256,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        critic_warmup_steps=3,
        value_hidden_dims=(16,),
        q_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        target_update_freq=1,
    )
    assert agent.observation_encoders.critic_or_actor.features_dim == 7

    env = agent.env
    state_shape = env.single_observation_space["state"].shape
    pose_shape = env.single_observation_space["state_object_pose"].shape
    for t in range(40):
        obs = {
            "state": torch.rand(env.num_envs, *state_shape) * 2 - 1,
            "state_object_pose": torch.rand(env.num_envs, *pose_shape) * 2 - 1,
        }
        next_obs = {
            "state": torch.rand(env.num_envs, *state_shape) * 2 - 1,
            "state_object_pose": torch.rand(env.num_envs, *pose_shape) * 2 - 1,
        }
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.rand(env.num_envs)
        dones = torch.zeros(env.num_envs, dtype=torch.bool)
        episode_end = torch.zeros(env.num_envs, dtype=torch.bool)
        if (t + 1) % 20 == 0:
            episode_end[:] = True
        agent.replay_buffer.add(
            obs, next_obs, actions, rewards, dones, episode_end=episode_end
        )

    shared_extractor = agent.observation_encoders.critic_or_actor
    assert shared_extractor is agent.policy.actor_extractor

    agent.train(2)  # Phase A (critic warmup)
    agent.train(1)  # Phase B (actor, post-warmup)

    # BPPO/UniO4 are state-only (Box obs only, per test_rejects_image_
    # observation_space), so the shared extractor is always a FlattenExtractor
    # with zero learnable parameters -- there is nothing for a before/after
    # parameter-movement check to observe. Assert the underlying mechanism
    # directly instead: under "shared_critic_grad" (the default here),
    # policy.extract_actor_features detaches (BasePolicy.
    # actor_features_detached), so gradient never reaches the shared
    # extractor through the actor path, while extract_critic_features (used
    # by BPPOCriticMixin._build_critic/_phase_a_step) never detaches.
    probe_obs = {
        "state": (torch.rand(env.num_envs, *state_shape) * 2 - 1).requires_grad_(),
        "state_object_pose": (
            torch.rand(env.num_envs, *pose_shape) * 2 - 1
        ).requires_grad_(),
    }
    assert not agent.policy.extract_actor_features(probe_obs).requires_grad
    assert shared_extractor.extract(probe_obs).requires_grad


def test_multi_key_state_privileged_critic_separate_encoders_trains_one_step():
    # The state-only privileged-critic idiom: actor sees only "state", critic
    # sees "state" + "state_object_pose", with independent encoders.
    agent = BPPO(
        env=_multi_key_state_env(),
        buffer_size=256,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        critic_warmup_steps=3,
        value_hidden_dims=(16,),
        q_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        target_update_freq=1,
        obs_groups=ObsGroups(actor=("state",), critic=("state", "state_object_pose")),
        encoder_sharing="separate",
    )
    assert agent.observation_encoders.actor.features_dim == 4
    assert agent.observation_encoders.critic.features_dim == 7
    assert agent.observation_encoders.actor is not agent.observation_encoders.critic

    env = agent.env
    state_shape = env.single_observation_space["state"].shape
    pose_shape = env.single_observation_space["state_object_pose"].shape
    for t in range(40):
        obs = {
            "state": torch.rand(env.num_envs, *state_shape) * 2 - 1,
            "state_object_pose": torch.rand(env.num_envs, *pose_shape) * 2 - 1,
        }
        next_obs = {
            "state": torch.rand(env.num_envs, *state_shape) * 2 - 1,
            "state_object_pose": torch.rand(env.num_envs, *pose_shape) * 2 - 1,
        }
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.rand(env.num_envs)
        dones = torch.zeros(env.num_envs, dtype=torch.bool)
        episode_end = torch.zeros(env.num_envs, dtype=torch.bool)
        if (t + 1) % 20 == 0:
            episode_end[:] = True
        agent.replay_buffer.add(
            obs, next_obs, actions, rewards, dones, episode_end=episode_end
        )

    actor_extractor = agent.observation_encoders.actor
    critic_extractor = agent.observation_encoders.critic

    agent.train(2)  # Phase A (critic warmup)
    agent.train(1)  # Phase B (actor, post-warmup)

    # Same reasoning as test_multi_key_state_observation_trains_one_step:
    # both extractors are parameter-free FlattenExtractors here (state-only
    # obs), so assert the mechanism instead of a param diff. Under
    # "separate", actor_extractor is not agent.observation_encoders.critic,
    # so BasePolicy.actor_features_detached is False and neither role
    # detaches -- each extractor is a genuinely independent object trained
    # only by its own role's loss.
    probe_obs = {
        "state": (torch.rand(env.num_envs, *state_shape) * 2 - 1).requires_grad_(),
        "state_object_pose": (
            torch.rand(env.num_envs, *pose_shape) * 2 - 1
        ).requires_grad_(),
    }
    assert agent.policy.extract_actor_features(probe_obs).requires_grad
    assert critic_extractor.extract(probe_obs).requires_grad
    assert actor_extractor is not critic_extractor


def test_old_policy_starts_synced_to_policy():
    agent = _make_agent()
    for p, q in zip(agent.policy.parameters(), agent.old_policy.parameters()):
        assert torch.equal(p, q)
    assert not any(p.requires_grad for p in agent.old_policy.parameters())


def test_phase_a_only_updates_critic_not_actor():
    agent = _make_agent(critic_warmup_steps=5)
    _fill(agent)
    actor_before = [p.clone() for p in agent.policy.parameters()]

    info = agent.train(2, compute_info=True)

    assert agent._phase_step == 2
    assert "value_loss" in info and np.isfinite(info["value_loss"])
    assert "q_loss" in info and np.isfinite(info["q_loss"])
    assert all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.parameters())
    )


def test_critic_step_matches_phase_step_throughout_phase_a():
    """BPPOCriticMixin owns a dedicated _critic_step counter (decoupled from
    the subclass's own phase gate for UniO4's sake, see bppo.py), but for
    BPPO specifically -- a 2-phase gate where Phase A is always first and the
    critic never updates again once Phase B starts -- the two must coincide
    throughout Phase A. This is what makes "tests/test_bppo.py passes
    unchanged after the BPPOCriticMixin extraction" an actual proof of
    behavior preservation."""
    agent = _make_agent(critic_warmup_steps=5, target_update_freq=2)
    _fill(agent)
    for step in range(5):
        agent.train(1)
        assert agent._critic_step == agent._phase_step == step + 1


def test_phase_a_to_b_transition_resyncs_old_policy_then_updates_actor():
    agent = _make_agent(critic_warmup_steps=2)
    _fill(agent)
    agent.train(2)  # exhaust Phase A
    assert agent._phase_step == 2

    value_before = [p.clone() for p in agent.value_net.parameters()]
    actor_before = [p.clone() for p in agent.policy.parameters()]

    info = agent.train(1, compute_info=True)

    assert agent._phase_step == 3
    assert agent._phase_b_step == 1
    assert "actor_loss" in info
    # V/Q frozen during Phase B.
    assert all(
        torch.equal(a, b) for a, b in zip(value_before, agent.value_net.parameters())
    )
    # Actor did change (one PPO-clip gradient step taken).
    assert any(
        not torch.equal(a, b)
        for a, b in zip(actor_before, agent.policy.parameters())
    )


def test_phase_b_ratio_is_one_on_first_step_after_transition():
    """old_policy == policy exactly at the A->B boundary, so the very first
    ratio = exp(new_log_prob - old_log_prob) must be 1.0."""
    agent = _make_agent(critic_warmup_steps=1)
    _fill(agent)
    agent.train(1)  # Phase A
    info = agent.train(1, compute_info=True)  # first Phase B step
    assert info["ratio"] == pytest.approx(1.0, abs=1e-5)


def test_entropy_weight_lowers_loss_not_raises_it():
    """loss = clip_loss - entropy_weight * entropy. Entropy of a Gaussian is
    always finite but its sign depends on std (can be negative for std<1/sqrt(2*pi*e)),
    so this only pins the *formula*, not a >=0 assumption: recompute clip_loss
    and entropy independently and check the combination matches exactly --
    catches an inverted sign (loss = clip_loss + entropy_weight*entropy),
    which would silently reward low-entropy (collapsing) policies instead of
    penalizing them."""
    from rl_garden.algorithms.ppo import ppo_clip_policy_loss

    agent = _make_agent(critic_warmup_steps=1, entropy_weight=0.7)
    _fill(agent)
    agent.train(1)  # Phase A

    torch.manual_seed(0)
    data = agent._sample_train_batch()
    with torch.no_grad():
        old_features = agent.old_policy.extract_features(data.obs)
        action, old_log_prob = agent.old_policy.actor.action_log_prob(old_features)
        advantage = (
            agent.q_net(data.obs["state"], action) - agent.value_net(data.obs["state"])
        ).squeeze(-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = agent._weighted_advantage(advantage)

    new_features = agent.policy.extract_features(data.obs, stop_gradient=False)
    new_log_prob = agent.policy.actor.evaluate_action_log_prob(new_features, action)
    ratio = (new_log_prob.squeeze(-1) - old_log_prob.squeeze(-1)).exp()
    expected_clip_loss = ppo_clip_policy_loss(advantage, ratio, agent._clip_ratio)
    expected_entropy = agent.policy.actor.entropy(new_features).squeeze(-1).mean()
    expected_loss = expected_clip_loss - agent.entropy_weight * expected_entropy

    torch.manual_seed(0)
    data2 = agent._sample_train_batch()
    agent._phase_b_step = 0  # avoid decaying clip_ratio a second time
    info = agent._phase_b_step_update(data2)

    assert info["actor_loss"] == pytest.approx(expected_clip_loss.item(), rel=1e-4)
    assert info["loss"] == pytest.approx(expected_loss.item(), rel=1e-4)


def test_clip_ratio_decays_then_freezes():
    agent = _make_agent(critic_warmup_steps=1, clip_ratio=0.25, clip_decay=0.5, clip_decay_steps=2)
    _fill(agent)
    agent.train(1)  # Phase A
    start = agent._clip_ratio
    agent.train(1)
    after_one = agent._clip_ratio
    agent.train(1)
    after_two = agent._clip_ratio
    agent.train(1)  # beyond clip_decay_steps=2 -- must not decay further
    after_three = agent._clip_ratio

    assert after_one == pytest.approx(start * 0.5)
    assert after_two == pytest.approx(start * 0.25)
    assert after_three == pytest.approx(after_two)


def test_q_target_masks_true_terminations_and_timeout_boundary():
    agent = _make_agent(critic_warmup_steps=100)
    env = agent.env
    state = torch.rand(1, *env.single_observation_space["state"].shape) * 2 - 1
    next_state = torch.rand(1, *env.single_observation_space["state"].shape) * 2 - 1
    obs = {"state": state}
    next_obs = {"state": next_state}
    action = torch.rand(1, *env.single_action_space.shape) * 2 - 1
    reward = torch.rand(1)

    # Step 0: true termination (dones=True).
    agent.replay_buffer.add(
        obs, next_obs, action, reward,
        torch.ones(1, dtype=torch.bool), episode_end=torch.ones(1, dtype=torch.bool),
    )
    # Step 1: artificial timeout boundary (dones=False, episode_end=True).
    agent.replay_buffer.add(
        obs, next_obs, action, reward,
        torch.zeros(1, dtype=torch.bool), episode_end=torch.ones(1, dtype=torch.bool),
    )
    # Steps 2-3: normal mid-episode transitions.
    for _ in range(2):
        agent.replay_buffer.add(
            obs, next_obs, action, reward,
            torch.zeros(1, dtype=torch.bool), episode_end=torch.zeros(1, dtype=torch.bool),
        )

    data = agent.replay_buffer._index_batch(
        torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0, 0, 0])
    )
    valid = (~data.dones.bool()) & data.next_action_valid
    assert valid.tolist() == [False, False, True, True]


def test_old_policy_syncs_only_on_eval_improvement():
    agent = _make_agent(critic_warmup_steps=1)
    _fill(agent)
    agent.train(2)  # into Phase B

    agent.policy.load_state_dict(
        {k: v + 1.0 if v.is_floating_point() else v for k, v in agent.policy.state_dict().items()}
    )
    assert not torch.equal(
        next(iter(agent.policy.parameters())), next(iter(agent.old_policy.parameters()))
    )

    agent._log_eval_metrics({"return": -10.0}, step=1)  # below -inf baseline still "improves"
    assert agent._best_eval_score == -10.0
    assert torch.equal(
        next(iter(agent.policy.parameters())), next(iter(agent.old_policy.parameters()))
    )

    # Diverge policy again; a *worse* score must not re-sync.
    agent.policy.load_state_dict(
        {k: v + 1.0 if v.is_floating_point() else v for k, v in agent.policy.state_dict().items()}
    )
    agent._log_eval_metrics({"return": -20.0}, step=2)
    assert agent._best_eval_score == -10.0
    assert not torch.equal(
        next(iter(agent.policy.parameters())), next(iter(agent.old_policy.parameters()))
    )


def test_old_policy_sync_skips_nan_eval_score_with_warning():
    agent = _make_agent(critic_warmup_steps=1)
    _fill(agent)
    agent.train(2)
    before = agent._best_eval_score

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent._log_eval_metrics({}, step=1)  # no "return" key -> NaN
        assert any("NaN" in str(w.message) for w in caught)
    assert agent._best_eval_score == before


def test_old_policy_sync_noop_during_phase_a():
    agent = _make_agent(critic_warmup_steps=100)
    _fill(agent)
    agent.train(1)  # still Phase A
    agent._log_eval_metrics({"return": 999.0}, step=1)
    assert agent._best_eval_score == float("-inf")


def test_load_actor_from_transfers_only_policy():
    bc = BC(
        env=_state_env(), device="cpu", buffer_device="cpu", buffer_size=64,
        batch_size=16, net_arch=[16], tanh_squash=False,
    )
    for _ in range(16):
        env = bc.env
        # env.single_observation_space is Dict({"state": Box}) -- bc.env's
        # bare Box is boundary-normalized by BaseAlgorithm.__init__ (see
        # rl_garden.envs.wrappers.VectorizedDictStateWrapper).
        state = torch.rand(1, *env.single_observation_space["state"].shape) * 2 - 1
        obs = {"state": state}
        next_obs = {"state": torch.rand_like(state)}
        actions = torch.rand(1, *env.single_action_space.shape) * 2 - 1
        rewards = torch.rand(1)
        dones = torch.zeros(1)
        bc.replay_buffer.add(obs, next_obs, actions, rewards, dones)
    bc.train(3)

    with tempfile.TemporaryDirectory() as tmp:
        path = bc.save(os.path.join(tmp, "bc.pt"))

        agent = _make_agent()
        gs, gu, phase = agent._global_step, agent._global_update, agent._phase_step
        value_before = [p.clone() for p in agent.value_net.parameters()]

        agent.load_actor_from(path)

        for k, v in bc.policy.state_dict().items():
            assert torch.equal(agent.policy.state_dict()[k], v)
            assert torch.equal(agent.old_policy.state_dict()[k], v)
        assert agent._global_step == gs
        assert agent._global_update == gu
        assert agent._phase_step == phase
        assert all(
            torch.equal(a, b) for a, b in zip(value_before, agent.value_net.parameters())
        )


def test_checkpoint_roundtrip_preserves_phase_state():
    agent = _make_agent(critic_warmup_steps=2)
    _fill(agent)
    agent.train(3)
    agent._log_eval_metrics({"return": 7.0}, step=1)

    with tempfile.TemporaryDirectory() as tmp:
        path = agent.save(os.path.join(tmp, "bppo.pt"))
        fresh = _make_agent(critic_warmup_steps=2)
        fresh.load(path, load_replay_buffer=False)

    assert fresh._phase_step == agent._phase_step
    assert fresh._phase_b_step == agent._phase_b_step
    assert fresh._critic_step == agent._critic_step == 2
    assert fresh._clip_ratio == pytest.approx(agent._clip_ratio)
    assert fresh._best_eval_score == agent._best_eval_score == 7.0
    for k, v in agent.value_net.state_dict().items():
        assert torch.equal(fresh.value_net.state_dict()[k], v)
    for k, v in agent.q_net.state_dict().items():
        assert torch.equal(fresh.q_net.state_dict()[k], v)


def test_registry_discovers_bppo():
    from rl_garden.training.offline._registry import registry

    registry.discover()
    assert "bppo" in registry._entries


def test_bppo_mro_is_offline_only():
    from rl_garden.algorithms.bppo import BPPOCriticMixin
    from rl_garden.algorithms.offline import OfflineRLAlgorithm

    assert [c.__name__ for c in BPPO.__mro__][:3] == [
        "BPPO",
        "BPPOCriticMixin",
        "OfflineRLAlgorithm",
    ]
    assert issubclass(BPPO, BPPOCriticMixin)
    assert issubclass(BPPO, OfflineRLAlgorithm)
