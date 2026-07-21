from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.nstep_buffer import NStepDictReplayBuffer
from rl_garden.common.checkpoint import checkpoint_dict, save_checkpoint_file
from rl_garden.common.types import NStepReplayBufferSample
from rl_garden.common.training_phase import InitialTrainingPhase


class DummyVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


class DummyDictVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
                "state": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


class FixedRewardEvalEnv(DummyVecEnv):
    def __init__(self, rewards: list[float], dones: list[bool]) -> None:
        super().__init__()
        self.num_envs = 1
        self.rewards = rewards
        self.dones = dones
        self.step_idx = 0

    def reset(self, seed: int | None = None):
        del seed
        self.step_idx = 0
        return torch.zeros(1, *self.single_observation_space.shape), {}

    def step(self, actions):
        del actions
        reward = torch.tensor([self.rewards[self.step_idx]], dtype=torch.float32)
        done = torch.tensor([self.dones[self.step_idx]], dtype=torch.bool)
        obs = torch.full(
            (1, *self.single_observation_space.shape),
            float(self.step_idx + 1),
        )
        infos = {}
        if done.any():
            episode_return = torch.tensor([sum(self.rewards[: self.step_idx + 1])])
            infos = {"final_info": {"episode": {"return": episode_return}}}
        self.step_idx += 1
        return obs, reward, done, torch.zeros_like(done), infos


def _agent(**kwargs) -> SAC:
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
    }
    params.update(kwargs)
    return SAC(
        env=DummyVecEnv(),
        **params,
    )


def _dict_agent(**kwargs) -> SAC:
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 4,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
        "image_keys": ("rgb",),
        "proprio_latent_dim": 4,
    }
    params.update(kwargs)
    return SAC(
        env=DummyDictVecEnv(),
        **params,
    )


def _fill(agent: SAC, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        kwargs = (
            {"episode_end": torch.zeros(env.num_envs, dtype=torch.bool)}
            if agent.nstep > 1
            else {}
        )
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones, **kwargs)


def _fill_dict(agent: SAC, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = {
            "rgb": torch.randint(
                0,
                256,
                (env.num_envs, 64, 64, 3),
                dtype=torch.uint8,
            ),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb": torch.randint(
                0,
                256,
                (env.num_envs, 64, 64, 3),
                dtype=torch.uint8,
            ),
            "state": torch.randn(env.num_envs, 4),
        }
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        kwargs = (
            {"episode_end": torch.zeros(env.num_envs, dtype=torch.bool)}
            if agent.nstep > 1
            else {}
        )
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones, **kwargs)


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _params_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def test_initial_training_phase_rejects_independent_encoder_update():
    with pytest.raises(ValueError, match="requires update_critic=True"):
        InitialTrainingPhase(
            duration_steps=10,
            update_actor=False,
            update_critic=False,
            update_encoder=True,
        )


def test_sac_redq_target_uses_subsampled_critics():
    agent = _agent(n_critics=10, critic_subsample_size=2)
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)
    next_action, _, next_features = agent.policy.actor_action_log_prob(data.next_obs)

    q_sub = agent.policy.q_values_subsampled(
        next_features,
        next_action,
        subsample_size=agent.critic_subsample_size,
        target=True,
    )
    target_q = agent._target_q(data)

    assert q_sub.shape == (2, agent.batch_size, 1)
    assert target_q.shape == (agent.batch_size, 1)
    assert torch.isfinite(target_q).all()


def test_backup_entropy_flag_controls_entropy_term_in_target_q():
    agent = _agent()
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    next_action = torch.zeros(agent.batch_size, *agent.env.single_action_space.shape)
    next_log_prob = torch.full((agent.batch_size, 1), 2.0)
    next_features = agent.policy.extract_features(data.next_obs, stop_gradient=True)
    agent._target_action_log_prob = lambda data: (next_action, next_log_prob, next_features)

    alpha = agent._current_alpha().detach()
    discounts = agent._target_discounts(data)

    agent.backup_entropy = True
    target_with_entropy = agent._target_q(data)
    agent.backup_entropy = False
    target_without_entropy = agent._target_q(data)

    expected_diff = -discounts * alpha * next_log_prob
    torch.testing.assert_close(target_with_entropy - target_without_entropy, expected_diff)


def test_backup_entropy_defaults_to_true():
    agent = _agent()
    assert agent.backup_entropy is True
    assert agent._backup_entropy_enabled() is True


def test_sac_core_high_utd_update_runs():
    agent = _agent(n_critics=4, critic_subsample_size=2, utd=2.0, batch_size=8)
    _fill(agent)

    info = agent.train(gradient_steps=2, compute_info=True)

    assert agent._global_update == 2
    assert info["utd_ratio"] == 2.0
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_sac_nstep_defaults_to_existing_replay_buffer():
    agent = _dict_agent()

    assert agent.nstep == 1
    assert agent._extra_batch_slice_keys == ()
    assert isinstance(agent.replay_buffer, DictReplayBuffer)
    assert not isinstance(agent.replay_buffer, NStepDictReplayBuffer)


def test_sac_nstep_uses_buffer_discounts_and_supports_high_utd():
    agent = _dict_agent(nstep=3, gamma=0.8, utd=2.0, batch_size=8)
    _fill_dict(agent)

    assert isinstance(agent.replay_buffer, NStepDictReplayBuffer)
    assert agent._extra_batch_slice_keys == ("discounts",)
    data = agent.replay_buffer.sample(agent.batch_size)
    assert isinstance(data, NStepReplayBufferSample)
    torch.testing.assert_close(
        agent._target_discounts(data),
        data.discounts.reshape(-1, 1),
    )

    info = agent.train(gradient_steps=2, compute_info=True)

    assert info["utd_ratio"] == 2.0
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_sac_actor_loss_uses_all_critics():
    agent = _agent(n_critics=5, critic_subsample_size=2)
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    _, _, features = agent.policy.actor_action_log_prob(data.obs)
    q_all = agent.policy.q_values_subsampled(
        features, data.actions, subsample_size=None, target=False
    )
    loss, log_prob = agent._actor_loss(data.obs)

    assert q_all.shape[0] == 5
    assert loss.shape == ()
    assert log_prob.shape == (agent.batch_size, 1)


def test_eval_q_mc_disabled_by_default():
    eval_env = FixedRewardEvalEnv(
        rewards=[1.0, 2.0, 3.0],
        dones=[False, False, True],
    )
    agent = _agent(eval_env=eval_env, num_eval_steps=3, gamma=0.5)
    # q_mc_diagnostics defaults to False — no q_mc/* keys expected
    metrics = agent._evaluate()

    assert not any(k.startswith("q_mc/") for k in metrics)


def test_eval_q_mc_metrics_use_complete_episode_returns():
    eval_env = FixedRewardEvalEnv(
        rewards=[1.0, 2.0, 3.0],
        dones=[False, False, True],
    )
    agent = _agent(eval_env=eval_env, num_eval_steps=3, gamma=0.5, q_mc_diagnostics=True)
    q_values = iter(
        [
            torch.tensor([2.0]),
            torch.tensor([2.0]),
            torch.tensor([2.0]),
        ]
    )
    agent._eval_q_values = lambda obs, actions: next(q_values)  # type: ignore[method-assign]

    metrics = agent._evaluate()

    expected_returns = torch.tensor([2.75, 3.5, 3.0])
    expected_errors = torch.tensor([2.0, 2.0, 2.0]) - expected_returns
    assert metrics["q_mc/num_steps"] == 3.0
    assert metrics["q_mc/q_mean"] == 2.0
    assert metrics["q_mc/mc_return_mean"] == pytest.approx(
        float(expected_returns.mean().item())
    )
    assert metrics["q_mc/mean_error"] == pytest.approx(
        float(expected_errors.mean().item())
    )
    assert metrics["q_mc/abs_error"] == pytest.approx(
        float(expected_errors.abs().mean().item())
    )
    assert metrics["q_mc/rmse"] == pytest.approx(
        math.sqrt(float(expected_errors.square().mean().item()))
    )


def test_eval_q_mc_metrics_skip_incomplete_episodes():
    eval_env = FixedRewardEvalEnv(
        rewards=[1.0, 2.0, 3.0],
        dones=[False, False, False],
    )
    agent = _agent(eval_env=eval_env, num_eval_steps=3, q_mc_diagnostics=True)
    agent._eval_q_values = lambda obs, actions: torch.tensor([1.0])  # type: ignore[method-assign]

    metrics = agent._evaluate()

    assert "q_mc/abs_error" not in metrics
    assert "q_mc/num_steps" not in metrics


class TruncatingEvalEnv(FixedRewardEvalEnv):
    """Like FixedRewardEvalEnv but fires truncations instead of terminations."""

    def step(self, actions):
        obs, reward, terminations, _, infos = super().step(actions)
        truncations = terminations.clone()
        terminations = torch.zeros_like(terminations)
        if truncations.any():
            infos["final_observation"] = torch.full(
                (1, *self.single_observation_space.shape), 99.0
            )
        return obs, reward, terminations, truncations, infos


def test_eval_q_mc_bootstrap_at_truncation():
    # Episode: rewards=[1,2,3], truncated at step 3, bootstrap V(s_final)=4.0
    # gamma=0.5: G_2=3+0.5*4=5, G_1=2+0.5*5=4.5, G_0=1+0.5*4.5=3.25
    eval_env = TruncatingEvalEnv(
        rewards=[1.0, 2.0, 3.0],
        dones=[False, False, True],
    )
    agent = _agent(eval_env=eval_env, num_eval_steps=3, gamma=0.5, q_mc_diagnostics=True)
    # Steps 0-2: Q=2.0; step 2 also triggers bootstrap call: Q(s_final)=4.0
    q_vals = iter([
        torch.tensor([2.0]),
        torch.tensor([2.0]),
        torch.tensor([2.0]),
        torch.tensor([4.0]),  # bootstrap V(s_final)
    ])
    agent._eval_q_values = lambda obs, actions: next(q_vals)  # type: ignore[method-assign]

    metrics = agent._evaluate()

    expected_returns = torch.tensor([3.25, 4.5, 5.0])
    assert metrics["q_mc/mc_return_mean"] == pytest.approx(
        float(expected_returns.mean().item())
    )


def test_actor_diagnostics_preserves_cpu_rng_state():
    agent = _agent()
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    torch.manual_seed(123)
    expected = torch.rand(4)

    torch.manual_seed(123)
    diagnostics = agent._actor_diagnostics(data)
    actual = torch.rand(4)

    assert "action_saturation" in diagnostics
    torch.testing.assert_close(actual, expected)


def test_critic_only_freezes_actor_and_encoder_but_updates_critic():
    agent = _dict_agent(
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000,
            update_actor=False,
            update_critic=True,
            update_encoder=False,
        )
    )
    agent._start_initial_training_phase()
    _fill_dict(agent)

    actor_before = _clone_params(agent.policy.actor)
    encoder_before = _clone_params(agent.policy.features_extractor)
    critic_before = _clone_params(agent.policy.critic)
    alpha_before = agent._current_alpha().detach().clone()

    info = agent.train(gradient_steps=1, compute_info=True)

    assert info["actor_loss"] == 0.0
    assert not _params_changed(actor_before, agent.policy.actor)
    assert not _params_changed(encoder_before, agent.policy.features_extractor)
    assert _params_changed(critic_before, agent.policy.critic)
    torch.testing.assert_close(agent._current_alpha(), alpha_before)


def test_critic_only_can_update_encoder():
    agent = _dict_agent(
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000,
            update_actor=False,
            update_critic=True,
            update_encoder=True,
        )
    )
    agent._start_initial_training_phase()
    _fill_dict(agent)

    actor_before = _clone_params(agent.policy.actor)
    encoder_before = _clone_params(agent.policy.features_extractor)

    agent.train(gradient_steps=1)

    assert not _params_changed(actor_before, agent.policy.actor)
    assert _params_changed(encoder_before, agent.policy.features_extractor)


def test_critic_only_high_utd_does_not_update_actor():
    agent = _agent(
        batch_size=8,
        utd=2.0,
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000,
            update_actor=False,
            update_critic=True,
            update_encoder=True,
        ),
    )
    agent._start_initial_training_phase()
    _fill(agent)
    actor_before = _clone_params(agent.policy.actor)

    info = agent.train(gradient_steps=2, compute_info=True)

    assert info["utd_ratio"] == 2.0
    assert info["actor_loss"] == 0.0
    assert not _params_changed(actor_before, agent.policy.actor)


def test_collect_only_phase_skips_sampling_and_updates(monkeypatch):
    agent = _agent(
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000,
            update_actor=False,
            update_critic=False,
            update_encoder=False,
        )
    )
    agent._start_initial_training_phase()
    monkeypatch.setattr(
        agent.replay_buffer,
        "sample",
        lambda *_: pytest.fail("collect-only phase must not sample replay"),
    )

    assert agent.train(gradient_steps=4, compute_info=True) == {}
    assert agent._global_update == 0


def test_initial_phase_ends_relative_to_start_step():
    agent = _agent(
        initial_training_phase=InitialTrainingPhase(
            duration_steps=100,
            update_actor=False,
            update_critic=True,
            update_encoder=True,
        )
    )
    agent._global_step = 500
    agent._start_initial_training_phase()

    agent._global_step = 599
    assert not agent._training_update_mask().update_actor

    agent._global_step = 600
    assert agent._training_update_mask().update_actor


def test_initial_phase_random_action_probability_one_uses_random_actions(
    monkeypatch,
):
    agent = _agent(
        initial_training_phase=InitialTrainingPhase(
            duration_steps=100,
            update_actor=False,
            update_critic=True,
            update_encoder=True,
            random_action_prob=1.0,
        )
    )
    agent._start_initial_training_phase()
    obs = torch.zeros(agent.env.num_envs, 4)
    policy_actions = torch.full((agent.env.num_envs, 2), 0.25)
    random_actions = torch.full((agent.env.num_envs, 2), -0.75)
    monkeypatch.setattr(agent, "_policy_action", lambda _: policy_actions)
    monkeypatch.setattr(agent, "_explore_action", lambda _: random_actions)

    actions, env_actions, context = agent._rollout_action(
        obs, learning_has_started=False
    )

    torch.testing.assert_close(actions, random_actions)
    torch.testing.assert_close(env_actions, random_actions)
    assert context is None


def test_load_actor_checkpoint_transfers_bc_actor_and_encoder(tmp_path):
    agent = _dict_agent()
    source_policy = agent.policy.state_dict()
    for key in list(source_policy):
        if key.startswith(("features_extractor.", "actor.")):
            source_policy[key] = torch.ones_like(source_policy[key])

    ckpt = checkpoint_dict(
        algorithm_class="BC",
        global_step=0,
        global_update=0,
        observation_space=agent.env.single_observation_space,
        action_space=agent.env.single_action_space,
        hyperparameters={},
        state={"policy": source_policy, "optimizers": {}, "extra": {}},
    )
    path = save_checkpoint_file(tmp_path / "bc.pt", ckpt)

    agent.load_actor_checkpoint(str(path))

    loaded = agent.policy.state_dict()
    for key, value in loaded.items():
        if key.startswith(("features_extractor.", "actor.")):
            torch.testing.assert_close(value, torch.ones_like(value))


def test_load_actor_checkpoint_rejects_incomplete_actor_architecture(tmp_path):
    agent = _dict_agent(net_arch={"pi": [16, 16, 16], "qf": [16]})
    source_agent = _dict_agent(net_arch={"pi": [16, 16], "qf": [16]})
    ckpt = checkpoint_dict(
        algorithm_class="BC",
        global_step=0,
        global_update=0,
        observation_space=agent.env.single_observation_space,
        action_space=agent.env.single_action_space,
        hyperparameters={},
        state={"policy": source_agent.policy.state_dict(), "optimizers": {}, "extra": {}},
    )
    path = save_checkpoint_file(tmp_path / "bc_two_layer.pt", ckpt)

    with pytest.raises(ValueError, match="missing in source checkpoint"):
        agent.load_actor_checkpoint(str(path))


def test_actor_diagnostics_preserves_cuda_rng_state():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    agent = _agent(device="cuda", buffer_device="cuda")
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    torch.cuda.manual_seed_all(123)
    expected = torch.rand(4, device="cuda")

    torch.cuda.manual_seed_all(123)
    diagnostics = agent._actor_diagnostics(data)
    actual = torch.rand(4, device="cuda")

    assert "action_saturation" in diagnostics
    torch.testing.assert_close(actual, expected)


def test_q_landscape_diagnostics_default_is_not_called(monkeypatch: pytest.MonkeyPatch):
    agent = _agent()
    _fill(agent)

    def _forbidden(*args, **kwargs):
        raise AssertionError("q landscape diagnostics should be disabled by default")

    monkeypatch.setattr(agent.policy, "q_landscape_diagnostics", _forbidden)
    info = agent.train(gradient_steps=1, compute_info=True)

    assert "q_uniform_var" not in info


def test_q_landscape_diagnostics_preserves_cpu_rng_state():
    agent = _agent(q_landscape_diagnostics=True)
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    torch.manual_seed(123)
    expected = torch.rand(4)

    torch.manual_seed(123)
    diagnostics = agent._q_landscape_diagnostics(data)
    actual = torch.rand(4)

    assert "q_uniform_var" in diagnostics
    assert "q_action_grad_norm" in diagnostics
    assert "feature_dormant_ratio" in diagnostics
    torch.testing.assert_close(actual, expected)


def test_q_landscape_diagnostics_preserves_cuda_rng_state():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    agent = _agent(device="cuda", buffer_device="cuda", q_landscape_diagnostics=True)
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    torch.cuda.manual_seed_all(123)
    expected = torch.rand(4, device="cuda")

    torch.cuda.manual_seed_all(123)
    diagnostics = agent._q_landscape_diagnostics(data)
    actual = torch.rand(4, device="cuda")

    assert "q_uniform_var" in diagnostics
    torch.testing.assert_close(actual, expected)


def test_critic_forward_uses_critic_encoder_when_configured():
    from rl_garden.encoders import FlattenExtractor

    class _RecordingFlatten(FlattenExtractor):
        def __init__(self, observation_space, features_dim):
            super().__init__(observation_space)
            self.calls = 0

        def extract(self, obs, stop_gradient=False):
            self.calls += 1
            return super().extract(obs, stop_gradient=stop_gradient)

    env = DummyVecEnv()
    critic_ext = _RecordingFlatten(env.single_observation_space, features_dim=4)
    agent = _agent()
    agent.policy.critic_features_extractor = critic_ext
    _fill(agent, steps=8)
    agent.train(gradient_steps=1, compute_info=True)
    assert critic_ext.calls > 0


def test_target_q_reuses_actor_features_when_no_separate_critic_encoder():
    agent = _agent()
    assert agent.policy.has_separate_critic_encoder is False
    calls = {"n": 0}
    orig = agent.policy.critic_extract_features

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    agent.policy.critic_extract_features = _spy
    _fill(agent, steps=8)
    agent.train(gradient_steps=1, compute_info=True)
    # _critic_forward calls critic_extract_features once per gradient step;
    # _target_q must NOT call it again when there is no separate critic
    # encoder (it must reuse the actor-derived next_features instead).
    assert calls["n"] == 1


def test_target_q_uses_critic_encoder_when_configured():
    from rl_garden.encoders import FlattenExtractor

    env = DummyVecEnv()
    critic_ext = FlattenExtractor(env.single_observation_space)
    agent = _agent()
    agent.policy.critic_features_extractor = critic_ext
    calls = {"n": 0}
    orig = agent.policy.critic_extract_features

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    agent.policy.critic_extract_features = _spy
    _fill(agent, steps=8)
    agent.train(gradient_steps=1, compute_info=True)
    # One call each from _critic_forward (current obs), _target_q (next_obs),
    # and _actor_loss (current obs, re-scored through the critic's own
    # encoder), now that a separate critic encoder exists.
    assert calls["n"] == 3


def test_prepare_batch_called_on_both_encoders_when_separate():
    from rl_garden.encoders import FlattenExtractor

    env = DummyVecEnv()
    critic_ext = FlattenExtractor(env.single_observation_space)
    agent = _agent()
    agent.policy.critic_features_extractor = critic_ext

    actor_calls = {"n": 0}
    critic_calls = {"n": 0}
    orig_actor_prepare = agent.policy.features_extractor.prepare_batch
    orig_critic_prepare = critic_ext.prepare_batch

    def _actor_spy(*a, **k):
        actor_calls["n"] += 1
        return orig_actor_prepare(*a, **k)

    def _critic_spy(*a, **k):
        critic_calls["n"] += 1
        return orig_critic_prepare(*a, **k)

    agent.policy.features_extractor.prepare_batch = _actor_spy
    critic_ext.prepare_batch = _critic_spy
    _fill(agent, steps=8)
    agent.train(gradient_steps=1, compute_info=True)
    assert actor_calls["n"] >= 1
    assert critic_calls["n"] >= 1
