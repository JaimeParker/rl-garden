"""Tests for TDMPC2Multitask, driven via the generic run_offline_pretraining
loop against a synthetic 3-task mmap dataset (no simulator/hardware)."""
from __future__ import annotations

import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, run_offline_pretraining
from rl_garden.algorithms.tdmpc2.multitask import TDMPC2Multitask

_TASKS = ["task_a", "task_b", "task_c"]
_OBS_DIMS = [4, 6, 5]
_ACTION_DIMS = [2, 3, 2]
_EPISODE_LENGTHS = [10, 8, 12]


def _make_env_spec() -> OfflineEnvSpec:
    return OfflineEnvSpec(
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(max(_OBS_DIMS),), dtype="float32"),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(max(_ACTION_DIMS),), dtype="float32"),
        num_envs=1,
    )


def _make_agent(tmp_path, mmap_name="buf", **kwargs) -> TDMPC2Multitask:
    params = dict(
        buffer_size=500,
        batch_size=8,
        horizon=2,
        task_dim=8,
        latent_dim=16,
        enc_dim=8,
        mlp_dim=16,
        num_q=2,
        num_bins=11,
        device="cpu",
    )
    params.update(kwargs)
    return TDMPC2Multitask(
        env=_make_env_spec(),
        tasks=_TASKS,
        obs_dims=_OBS_DIMS,
        action_dims=_ACTION_DIMS,
        episode_lengths=_EPISODE_LENGTHS,
        mmap_dir=str(tmp_path / mmap_name),
        mmap_mode="create",
        **params,
    )


def _fill(agent: TDMPC2Multitask, episodes_per_task: int = 5) -> None:
    for task_idx, (odim, adim, elen) in enumerate(zip(_OBS_DIMS, _ACTION_DIMS, _EPISODE_LENGTHS)):
        for _ in range(episodes_per_task):
            obs = torch.zeros(elen, max(_OBS_DIMS))
            obs[:, :odim] = torch.randn(elen, odim)
            action = torch.zeros(elen, max(_ACTION_DIMS))
            action[:, :adim] = torch.randn(elen, adim).clamp(-1, 1)
            agent.replay_buffer.load_episode(obs, action, torch.ones(elen), task_idx)


def test_rejects_mismatched_task_metadata_lengths(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        TDMPC2Multitask(
            env=_make_env_spec(),
            tasks=_TASKS,
            obs_dims=_OBS_DIMS[:2],
            action_dims=_ACTION_DIMS,
            episode_lengths=_EPISODE_LENGTHS,
            mmap_dir=str(tmp_path / "buf"),
            device="cpu",
        )


def test_gradient_step_produces_finite_losses_and_updates_both_optimizers(tmp_path):
    agent = _make_agent(tmp_path)
    _fill(agent)
    world_model = agent.policy.world_model

    dyn_before = [p.detach().clone() for p in world_model._dynamics.parameters()]
    pi_before = [p.detach().clone() for p in world_model._pi.parameters()]

    info = agent.train(gradient_steps=1, compute_info=True)

    assert info["total_loss"] == info["total_loss"]  # not NaN
    assert any(not torch.equal(a, b) for a, b in zip(dyn_before, world_model._dynamics.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(pi_before, world_model._pi.parameters()))


def test_predict_action_masking_holds_through_a_training_step(tmp_path):
    agent = _make_agent(tmp_path)
    _fill(agent)
    agent.train(gradient_steps=1)
    world_model = agent.policy.world_model

    z = torch.randn(5, world_model.latent_dim)
    task = torch.tensor([0, 2, 0, 2, 0])  # action_dim=2 < max=3
    action, _ = world_model.pi(z, task)
    assert torch.all(action[:, 2] == 0.0)


def test_run_offline_pretraining_drives_agent_and_writes_checkpoints(tmp_path):
    agent = _make_agent(
        tmp_path,
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_freq=3,
        log_freq=2,
        std_log=False,
    )
    _fill(agent)

    run_offline_pretraining(
        agent,
        num_steps=10,
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_freq=3,
        log_freq=2,
        std_log=False,
        eval_freq=0,
    )

    assert agent._global_step == 10
    assert agent._global_update == 10
    ckpt_dir = tmp_path / "ckpts"
    assert (ckpt_dir / "offline_pretrained.pt").exists()
    assert any(p.name.startswith("checkpoint_") for p in ckpt_dir.iterdir())


def test_checkpoint_round_trip_preserves_policy_state(tmp_path):
    agent = _make_agent(tmp_path, mmap_name="buf1")
    _fill(agent)
    agent.train(gradient_steps=2)

    ckpt_path = agent.save(tmp_path / "ckpt.pt", include_replay_buffer=False)

    agent2 = _make_agent(tmp_path, mmap_name="buf2")
    agent2.load(str(ckpt_path), load_replay_buffer=False)

    sd1 = agent.policy.state_dict()
    sd2 = agent2.policy.state_dict()
    assert set(sd1) == set(sd2)
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), f"mismatch at {key}"
    assert agent2._global_step == agent._global_step
    assert agent2._global_update == agent._global_update
