"""The TD-MPC2 actor update must receive dQ/da through the critic while
leaving the critic's own parameters without gradient (upstream's
``_detach_Qs``). With ``entropy_coef=0`` the Q term is the only source of
actor gradient, so a zero actor gradient means the Q path is cut."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.tdmpc2 import TDMPC2
from rl_garden.algorithms.tdmpc2_multitask import TDMPC2Multitask

_ACTION_DIM = 2


class _SpacesOnlyEnv:
    num_envs = 1
    single_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, _ACTION_DIM), dtype=np.float32)


def _randomize_critic_heads(policy) -> None:
    # The critic's last layers are zero-initialized (TD-MPC2 init), which
    # makes Q constant in the action at init; give them weights so dQ/da != 0.
    with torch.no_grad():
        for q in policy.critic.qs:
            q[-1].weight.normal_(std=0.5)


def _assert_actor_gets_q_gradient_and_critic_does_not(agent, zs, task_args=()) -> None:
    _randomize_critic_heads(agent.policy)
    agent.world_optimizer.zero_grad(set_to_none=True)
    agent._update_pi(zs, task_args)

    action_dim = agent.policy.actor[-1].out_features // 2
    mean_head_grad = agent.policy.actor[-1].weight.grad[:action_dim]
    assert mean_head_grad.norm() > 0
    for name, param in agent.policy.critic.named_parameters():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0, name


def test_single_task_actor_receives_q_gradient():
    torch.manual_seed(0)
    agent = TDMPC2(
        env=_SpacesOnlyEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=50,
        batch_size=4,
        episode_length=5,
        seed_steps=6,
        horizon=2,
        num_samples=8,
        num_elites=4,
        num_pi_trajs=2,
        iterations=1,
        latent_dim=8,
        mlp_dim=8,
        num_q=2,
        num_bins=11,
        dropout=0.0,
        entropy_coef=0.0,
        eval_freq=0,
    )
    zs = torch.randn(3, 4, agent.latent_dim)
    _assert_actor_gets_q_gradient_and_critic_does_not(agent, zs)


def test_multitask_actor_receives_q_gradient(tmp_path):
    torch.manual_seed(0)
    obs_dims, action_dims = [4, 3], [2, 1]
    agent = TDMPC2Multitask(
        env=OfflineEnvSpec(
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        ),
        tasks=["a", "b"],
        obs_dims=obs_dims,
        action_dims=action_dims,
        episode_lengths=[10, 10],
        mmap_dir=str(tmp_path / "buf"),
        buffer_size=50,
        batch_size=4,
        horizon=2,
        task_dim=4,
        latent_dim=8,
        enc_dim=8,
        mlp_dim=8,
        num_q=2,
        num_bins=11,
        dropout=0.0,
        entropy_coef=0.0,
        device="cpu",
    )
    zs = torch.randn(3, 4, agent.latent_dim)
    task = torch.tensor([0, 1, 0, 1])
    _assert_actor_gets_q_gradient_and_critic_does_not(agent, zs, (task,))
