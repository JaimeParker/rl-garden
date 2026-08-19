from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import DPPO
from rl_garden.envs.wrappers import ActionChunkWrapper

OBS_DIM = 5
ACTION_DIM = 2
EPISODE_LEN = 6


class _FakeEnv(gym.Env):
    """SAME_STEP-autoreset fake vector env, fixed episode length."""

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


def test_dppo_learn_runs_and_produces_finite_losses():
    torch.manual_seed(0)
    env = _make_env(num_envs=4, act_steps=2)
    agent = DPPO(
        env=env,
        num_steps=3,
        horizon_steps=2,
        act_steps=2,
        denoising_steps=5,
        ft_denoising_steps=3,
        actor_mlp_dims=[16, 16, 16],
        critic_mlp_dims=[16, 16, 16],
        update_epochs=2,
        update_batch_size=8,
        eval_freq=0,
        device="cpu",
    )
    agent.learn(total_timesteps=3 * 4 * 2)
    losses = agent.train()
    for key, value in losses.items():
        assert np.isfinite(value), (key, value)
    assert agent.rollout_buffer.pos == agent.num_steps or agent.rollout_buffer.full


def test_dppo_rejects_unchunked_env():
    # ACTION_DIM=2 != act_steps=3: the raw (unwrapped) single_action_space
    # shape (ACTION_DIM,) cannot be mistaken for a chunked (act_steps,
    # ACTION_DIM) shape here, unlike a same-valued choice would.
    env = _FakeEnv(num_envs=2)
    try:
        DPPO(env=env, num_steps=2, horizon_steps=3, act_steps=3, denoising_steps=4, ft_denoising_steps=2, device="cpu")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "ActionChunkWrapper" in str(e)


def test_dppo_bc_checkpoint_loads_into_actor_and_actor_ft(tmp_path):
    import h5py
    from gymnasium import spaces as gym_spaces

    from rl_garden.algorithms import DiffusionBC, OfflineEnvSpec

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
        gym_spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        gym_spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
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
    agent = DPPO(
        env=env,
        bc_checkpoint=str(bc_ckpt),
        num_steps=2,
        horizon_steps=2,
        act_steps=2,
        denoising_steps=5,
        ft_denoising_steps=2,
        actor_mlp_dims=[16, 16, 16],
        critic_mlp_dims=[16, 16, 16],
        device="cpu",
    )
    for (na, pa), (nf, pf) in zip(
        agent.policy.actor.named_parameters(), bc_agent.ema_policy.net.named_parameters()
    ):
        assert na == nf
        assert torch.allclose(pa, pf)
    for (na, pa), (nf, pf) in zip(
        agent.policy.actor_ft.named_parameters(), bc_agent.ema_policy.net.named_parameters()
    ):
        assert torch.allclose(pa, pf)
    assert not any(p.requires_grad for p in agent.policy.actor.parameters())
    assert all(p.requires_grad for p in agent.policy.actor_ft.parameters())
