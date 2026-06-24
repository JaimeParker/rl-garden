from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.common.checkpoint import checkpoint_dict, save_checkpoint_file
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
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


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
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


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


def test_sac_core_high_utd_update_runs():
    agent = _agent(n_critics=4, critic_subsample_size=2, utd=2.0, batch_size=8)
    _fill(agent)

    info = agent.train(gradient_steps=2, compute_info=True)

    assert agent._global_update == 2
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
