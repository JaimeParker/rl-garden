from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec, UniO4, UniO4OPE
from rl_garden.algorithms import unio4_ope as unio4_ope_module


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> UniO4OPE:
    defaults = dict(
        env=_state_env(),
        task="halfcheetah-medium-v2",
        buffer_size=256,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        critic_warmup_steps=2,
        bc_ensemble_steps=2,
        num_policies=2,
        value_hidden_dims=(16,),
        q_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        target_update_freq=1,
        dynamics_hidden_dims=(8, 8),
        dynamics_weight_decay=(1e-5, 1e-5, 1e-5),
        dynamics_n_ensemble=2,
        dynamics_n_elites=1,
        dynamics_max_epochs=2,
        dynamics_max_epochs_since_update=1,
        dynamics_batch_size=32,
        ope_rollout_length=5,
        ope_rollout_batch_size=8,
        ope_gating_freq=1,
    )
    defaults.update(kwargs)
    return UniO4OPE(**defaults)


def _fill(agent, steps: int = 60, episode_len: int = 20) -> None:
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


def test_mro_subclasses_unio4():
    assert issubclass(UniO4OPE, UniO4)


def test_requires_task_for_termination_fn():
    with pytest.raises(TypeError):
        UniO4OPE(env=_state_env(), device="cpu", buffer_device="cpu")


def test_rejects_unknown_task():
    with pytest.raises(ValueError, match="No termination_fn"):
        _make_agent(task="not-a-real-task")


def test_registry_discovers_unio4_ope():
    from rl_garden.training.offline._registry import registry

    registry.discover()
    assert "unio4_ope" in registry._entries


def test_real_eval_sync_is_noop_but_still_logs():
    """UniO4OPE's real-env eval keeps computing/logging (via super()'s own
    _log_eval_metrics call), but must never sync old_actors from it -- that
    now comes exclusively from OPE gating."""
    agent = _make_agent()
    agent._phase_step = agent._improve_phase_start  # pretend Phase IMPROVE started
    old_actors_before = [
        {k: v.clone() for k, v in oa.state_dict().items()} for oa in agent.old_actors
    ]

    agent._log_eval_metrics({"return_0": 999.0, "return_1": 999.0}, step=1)

    assert agent._best_eval_score == [float("-inf")] * agent.num_policies
    for before, oa in zip(old_actors_before, agent.old_actors):
        for k, v in oa.state_dict().items():
            assert torch.equal(before[k], v)


def test_dynamics_trained_lazily_exactly_once_on_first_improve_train_call():
    torch.manual_seed(0)
    agent = _make_agent()
    _fill(agent, steps=60, episode_len=20)

    # UniO4.train() increments _phase_step to _improve_phase_start ON the
    # (critic_warmup_steps + bc_ensemble_steps)-th call (it processes the
    # *last* BC step, then increments past the boundary) -- so dynamics
    # training fires on that same call, not a subsequent one.
    boundary_calls = agent.critic_warmup_steps + agent.bc_ensemble_steps
    assert not agent._dynamics_trained
    for _ in range(boundary_calls - 1):
        agent.train(1)
    assert not agent._dynamics_trained  # still in earlier phases

    agent.train(1)  # the boundary call: _phase_step reaches _improve_phase_start
    assert agent._dynamics_trained
    assert agent.dynamics_input_mean is not None
    assert agent.dynamics_input_std is not None
    trained_mean = agent.dynamics_input_mean.clone()

    agent.train(1)  # first real Phase IMPROVE step: must not retrain
    assert torch.equal(agent.dynamics_input_mean, trained_mean)


def test_ope_gating_syncs_old_actors_when_score_improves(monkeypatch):
    torch.manual_seed(0)
    agent = _make_agent(ope_gating_freq=1)
    _fill(agent, steps=60, episode_len=20)
    boundary_calls = agent.critic_warmup_steps + agent.bc_ensemble_steps
    for _ in range(boundary_calls - 1):
        agent.train(1)
    agent.train(1)  # boundary call: trains dynamics model, no gating check yet
    assert agent._dynamics_trained

    scores = iter([10.0, 5.0, 20.0, 1.0])  # member 0: 10 -> 20 (sync twice); member 1: 5 -> 1 (no second sync)

    def fake_rollout_q_mean(*args, **kwargs):
        return next(scores)

    monkeypatch.setattr(unio4_ope_module, "rollout_q_mean", fake_rollout_q_mean)

    agent.train(1)  # first real Phase IMPROVE step (updates actors) + gating check #1
    actors0_after_ppo = {k: v.clone() for k, v in agent.actors[0].state_dict().items()}
    assert agent._best_ope_score == [10.0, 5.0]
    for k, v in agent.old_actors[0].state_dict().items():
        assert torch.equal(actors0_after_ppo[k], v)  # synced to the just-updated actor

    agent.train(1)  # gating check #2: scores 20.0, 1.0 -- member 0 improves, member 1 doesn't
    assert agent._best_ope_score == [20.0, 5.0]


def test_checkpoint_roundtrip_includes_dynamics_model(tmp_path):
    torch.manual_seed(0)
    agent = _make_agent()
    _fill(agent, steps=60, episode_len=20)
    for _ in range(agent.critic_warmup_steps + agent.bc_ensemble_steps + 1):
        agent.train(1)
    assert agent._dynamics_trained

    path = agent.save(str(tmp_path / "ckpt.pt"))
    agent2 = _make_agent()
    agent2.load(path, load_replay_buffer=False)

    assert agent2._dynamics_trained
    for (n1, p1), (n2, p2) in zip(
        agent.dynamics_model.named_parameters(), agent2.dynamics_model.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)
    assert torch.equal(agent.dynamics_input_mean, agent2.dynamics_input_mean)
    assert torch.equal(agent.dynamics_input_std, agent2.dynamics_input_std)
    assert torch.equal(agent.dynamics_model.elites, agent2.dynamics_model.elites)
    assert agent2._best_ope_score == agent._best_ope_score
