from __future__ import annotations

import os
import tempfile

import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import DiffusionBC, DiffusionCMDistillOnline, OfflineEnvSpec
from rl_garden.algorithms.diffusion_cm_distill import _scalings_for_boundary_conditions
from rl_garden.envs.wrappers import ActionChunkWrapper

OBS_DIM = 5
ACTION_DIM = 2
EPISODE_LEN = 6


class _FakeEnv(gym.Env):
    """SAME_STEP-autoreset fake vector env, fixed episode length (matches
    ``test_dppo_smoke.py``'s fixture -- DiffusionCMDistillOnline needs the
    same shape of environment DPPO's own smoke test uses)."""

    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self._step_count = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._step_count.zero_()
        return torch.randn(self.num_envs, OBS_DIM), {}

    def step(self, action):
        del action
        self._step_count += 1
        done = self._step_count >= EPISODE_LEN
        reward = torch.ones(self.num_envs)
        final_obs = torch.randn(self.num_envs, OBS_DIM)
        info = {}
        if done.any():
            info = {
                "final_observation": final_obs,
                "_final_observation": done.clone(),
                "final_info": {"episode": {"return": (self._step_count.float() * reward)}},
                "_final_info": done.clone(),
            }
            self._step_count[done] = 0
        obs = torch.randn(self.num_envs, OBS_DIM)
        terminated = done.clone()
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, reward, terminated, truncated, info


def _make_env(num_envs: int, act_steps: int) -> ActionChunkWrapper:
    return ActionChunkWrapper(_FakeEnv(num_envs), act_steps=act_steps)


def _make_agent(**overrides):
    env = _make_env(num_envs=4, act_steps=2)
    kwargs = dict(
        env=env,
        num_steps=3,
        horizon_steps=2,
        act_steps=2,
        denoising_steps=5,
        ft_denoising_steps=3,
        actor_mlp_dims=[16, 16, 16],
        critic_mlp_dims=[16, 16, 16],
        cm_mlp_dims=[16, 16, 16],
        update_epochs=2,
        update_batch_size=8,
        eval_freq=0,
        device="cpu",
        cm_lr=1e-3,
    )
    kwargs.update(overrides)
    return DiffusionCMDistillOnline(**kwargs)


def test_mro_has_no_ppo():
    from rl_garden.algorithms.base_algorithm import BaseAlgorithm
    from rl_garden.algorithms.dppo import DPPO, DPPOCore
    from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
    from rl_garden.algorithms.ppo import PPO

    mro = DiffusionCMDistillOnline.__mro__
    assert mro[:4] == (DiffusionCMDistillOnline, DPPO, DPPOCore, OnPolicyAlgorithm)
    assert BaseAlgorithm in mro
    assert PPO not in mro
    assert issubclass(DiffusionCMDistillOnline, DPPO)


def test_registry_discovers_diffusion_cm_distill_online():
    from rl_garden.training.online._registry import registry

    registry.discover()
    assert "diffusion_cm_distill_online" in registry._entries


def test_rejects_unchunked_env():
    env = _FakeEnv(num_envs=2)
    with pytest.raises(ValueError, match="ActionChunkWrapper"):
        DiffusionCMDistillOnline(
            env=env,
            num_steps=2,
            horizon_steps=3,
            act_steps=3,
            denoising_steps=4,
            ft_denoising_steps=2,
            device="cpu",
        )


def test_scalings_for_boundary_conditions_matches_rl100_formula():
    """RL-100's cm_util.py formula, generalized with a tunable
    ``timestep_scaling`` (default 0.1 reproduces upstream's literal,
    hardcoded ``timestep / 0.1`` body exactly). Pin the exact formula, not
    just its shape."""
    t = torch.tensor([0.0, 3.0, 10.0])
    c_skip, c_out = _scalings_for_boundary_conditions(t, sigma_data=0.5, timestep_scaling=0.1)

    scaled = t / 0.1
    expected_c_skip = 0.5**2 / (scaled**2 + 0.5**2)
    expected_c_out = scaled / (scaled**2 + 0.5**2) ** 0.5
    assert torch.allclose(c_skip, expected_c_skip)
    assert torch.allclose(c_out, expected_c_out)
    # t=0 is the identity boundary condition: c_skip=1, c_out=0.
    assert torch.allclose(c_skip[0], torch.tensor(1.0))
    assert torch.allclose(c_out[0], torch.tensor(0.0))


def test_scalings_for_boundary_conditions_timestep_scaling_grades_the_curve():
    """cm_timestep_scaling exists precisely because the upstream default
    (0.1) is calibrated for a ~1000-step schedule and collapses to a near-
    binary t=0-vs-everything-else switch at this repo's much shorter
    denoising_steps (5-20) -- a wider scaling should visibly grade c_skip
    across a short schedule instead of collapsing it immediately past t=0."""
    t = torch.tensor([0.0, 1.0, 4.0])
    _, c_out_narrow = _scalings_for_boundary_conditions(t, sigma_data=0.5, timestep_scaling=0.1)
    _, c_out_wide = _scalings_for_boundary_conditions(t, sigma_data=0.5, timestep_scaling=10.0)

    # t=1 already saturates c_out~1 under the narrow (default) scaling...
    assert c_out_narrow[1] > 0.99
    # ...but stays meaningfully below 1 under a scaling sized to the schedule.
    assert c_out_wide[1] < 0.9


def test_learn_and_train_runs_and_produces_finite_losses():
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)
    losses = agent.train()
    for key, value in losses.items():
        assert np.isfinite(value), (key, value)
    assert "cm_distill_loss" in losses


def test_distill_step_does_not_leak_gradient_into_teacher():
    """Pinned design requirement: the distillation step's teacher forward
    must run under no_grad, since the teacher (actor/actor_ft) is
    simultaneously being PPO-fine-tuned every iteration -- an accidental
    grad-enabled teacher forward would silently nudge PPO's own params."""
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)

    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    actor_ft_before = [p.clone() for p in agent.policy.actor_ft.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]

    agent._distill_step()

    assert all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(actor_ft_before, agent.policy.actor_ft.parameters())
    )
    assert all(
        torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
    )


def test_train_updates_teacher_student_and_ema_target():
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)

    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    actor_ft_before = [p.clone() for p in agent.policy.actor_ft.parameters()]
    target_before = [p.clone() for p in agent.cm_target.parameters()]
    student_before = [p.clone() for p in agent.cm_student.parameters()]

    agent.train()

    assert all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(actor_ft_before, agent.policy.actor_ft.parameters())
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(student_before, agent.cm_student.parameters())
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(target_before, agent.cm_target.parameters())
    )


def test_ema_target_update_matches_polyak_formula():
    torch.manual_seed(0)
    agent = _make_agent(cm_ema_decay=0.9)
    agent.learn(total_timesteps=3 * 4 * 2)

    target_before = [p.clone() for p in agent.cm_target.parameters()]
    agent._distill_step()
    student_after = [p.clone() for p in agent.cm_student.parameters()]
    target_after = list(agent.cm_target.parameters())

    tau = 1.0 - agent.cm_ema_decay
    for tb, sa, ta in zip(target_before, student_after, target_after):
        expected = tb * (1.0 - tau) + sa * tau
        assert torch.allclose(ta, expected, atol=1e-6)


def test_checkpoint_roundtrip_includes_cm_student_and_target(tmp_path):
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)
    agent.train()

    path = agent.save(str(tmp_path / "ckpt.pt"))
    agent2 = _make_agent()
    agent2.load(path, load_replay_buffer=False)

    for (n1, p1), (n2, p2) in zip(
        agent.cm_student.named_parameters(), agent2.cm_student.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)
    for (n1, p1), (n2, p2) in zip(
        agent.cm_target.named_parameters(), agent2.cm_target.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_bc_checkpoint_warm_starts_cm_student_and_target(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "bc.h5"
    with h5py.File(path, "w") as f:
        for i in range(4):
            g = f.create_group(f"traj_{i}")
            g.create_dataset(
                "obs", data=rng.standard_normal((21, OBS_DIM)).astype(np.float32)
            )
            g.create_dataset(
                "actions", data=(rng.random((20, ACTION_DIM)).astype(np.float32) * 2 - 1)
            )
            g.create_dataset("rewards", data=np.zeros(20, dtype=np.float32))
            dones = np.zeros(20, dtype=np.float32)
            dones[-1] = 1.0
            g.create_dataset("dones", data=dones)

    bc_env = OfflineEnvSpec(
        spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
    )
    bc_agent = DiffusionBC(
        env=bc_env,
        dataset_path=str(path),
        horizon_steps=2,
        cond_steps=1,
        denoising_steps=5,
        mlp_dims=[16, 16, 16],
        batch_size=8,
        device="cpu",
    )
    bc_agent.train(5)
    bc_ckpt = bc_agent.save(tmp_path / "bc.pt")

    env = _make_env(num_envs=2, act_steps=2)
    agent = DiffusionCMDistillOnline(
        env=env,
        bc_checkpoint=str(bc_ckpt),
        num_steps=2,
        horizon_steps=2,
        act_steps=2,
        denoising_steps=5,
        ft_denoising_steps=2,
        actor_mlp_dims=[16, 16, 16],
        critic_mlp_dims=[16, 16, 16],
        cm_mlp_dims=[16, 16, 16],
        device="cpu",
    )
    for (na, pa), (nf, pf) in zip(
        agent.cm_student.named_parameters(), bc_agent.ema_policy.net.named_parameters()
    ):
        assert na == nf
        assert torch.allclose(pa, pf)
    for (na, pa), (nf, pf) in zip(
        agent.cm_target.named_parameters(), bc_agent.ema_policy.net.named_parameters()
    ):
        assert torch.allclose(pa, pf)
    assert not any(p.requires_grad for p in agent.cm_target.parameters())
