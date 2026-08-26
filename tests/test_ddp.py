"""Tests for rl_garden/common/ddp.py's single-node DDP primitives.

Uses torch.multiprocessing.spawn with the gloo backend to run real
multi-process torch.distributed groups on CPU -- the standard pattern for
testing torch.distributed code without GPU hardware. Target functions must be
importable at module level (spawn pickles them), so every worker body is a
top-level function, not a nested one.
"""
from __future__ import annotations

import socket
from contextlib import closing

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.ppo import PPO
from rl_garden.common.ddp import (
    allreduce_grads,
    allreduce_mean,
    broadcast_module_state,
    ddp_rank,
    ddp_world_size,
    is_ddp_active,
    pin_backend_config_device,
    shutdown_ddp,
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )


def _spawn(worker, world_size: int, extra_args: tuple = ()) -> None:
    port = _free_port()
    mp.spawn(worker, args=(world_size, port, *extra_args), nprocs=world_size, join=True)


class _TinyModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((3,), value))
        self.register_buffer("stat", torch.full((2,), value))


def _broadcast_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    module = _TinyModule(value=float(rank + 1))  # rank 0 -> 1.0, rank 1 -> 2.0
    broadcast_module_state(module, src=0)
    assert torch.equal(module.weight, torch.full((3,), 1.0))
    assert torch.equal(module.stat, torch.full((2,), 1.0))
    dist.destroy_process_group()


def test_broadcast_module_state_two_processes():
    _spawn(_broadcast_worker, world_size=2)


def _allreduce_grads_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    module = _TinyModule(value=0.0)
    module.weight.grad = torch.full((3,), float(rank + 1))  # 1.0, 2.0 -> mean 1.5
    allreduce_grads(module)
    assert torch.allclose(module.weight.grad, torch.full((3,), 1.5))
    dist.destroy_process_group()


def test_allreduce_grads_two_processes():
    _spawn(_allreduce_grads_worker, world_size=2)


def _allreduce_grads_skips_none_grad_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    module = _TinyModule(value=0.0)
    # No .grad set anywhere -- must not raise.
    allreduce_grads(module)
    assert module.weight.grad is None
    dist.destroy_process_group()


def test_allreduce_grads_noop_when_no_grad():
    _spawn(_allreduce_grads_skips_none_grad_worker, world_size=2)


def _allreduce_mean_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    value = torch.tensor(float(rank + 1))  # 1.0, 2.0 -> mean 1.5
    result = allreduce_mean(value)
    assert torch.allclose(result, torch.tensor(1.5))
    dist.destroy_process_group()


def test_allreduce_mean_two_processes():
    _spawn(_allreduce_mean_worker, world_size=2)


def _pin_backend_device_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    from rl_garden.common.env_args import IsaacLabConfig, ManiSkillConfig

    assert is_ddp_active()
    isaaclab_cfg = IsaacLabConfig()
    assert isaaclab_cfg.sim_device == "cuda:0"
    pin_backend_config_device(isaaclab_cfg, rank)
    assert isaaclab_cfg.sim_device == f"cuda:{rank}"

    maniskill_cfg = ManiSkillConfig()
    assert maniskill_cfg.sim_backend == "gpu"
    pin_backend_config_device(maniskill_cfg, rank)
    assert maniskill_cfg.sim_backend == f"gpu:{rank}"
    assert maniskill_cfg.render_backend == f"gpu:{rank}"
    dist.destroy_process_group()


def test_pin_backend_config_device_rewrites_when_ddp_active():
    _spawn(_pin_backend_device_worker, world_size=2)


def _pin_backend_device_via_cli_entrypoint_worker(rank: int, world_size: int, port: int) -> None:
    """pin_backend_config_device must rewrite the *real* backend_config object
    reached via _ppo_env_request(args, run_name) -- not just a config
    instance built directly in a test, which wouldn't catch a frozen
    dataclass or another mutability obstacle on the real call path."""
    _init_gloo(rank, world_size, port)
    from rl_garden.training.online.ppo import PPOArgs, _ppo_env_request

    args = PPOArgs(
        env_backend="isaaclab", eval_freq=0, log_freq=0, obs_mode="state",
        num_envs=4, num_eval_envs=4,
    )
    req = _ppo_env_request(args, "testrun")
    assert req.backend_config.sim_device == "cuda:0"

    pin_backend_config_device(req.backend_config, rank)

    assert req.backend_config.sim_device == f"cuda:{rank}"
    dist.destroy_process_group()


def test_pin_backend_config_device_rewrites_real_object_from_cli_entrypoint():
    _spawn(_pin_backend_device_via_cli_entrypoint_worker, world_size=2)


def test_pin_backend_config_device_is_noop_when_ddp_inactive():
    from rl_garden.common.env_args import IsaacLabConfig

    assert not is_ddp_active()
    cfg = IsaacLabConfig()
    pin_backend_config_device(cfg, local_rank=1)
    assert cfg.sim_device == "cuda:0"  # untouched


def test_ddp_helpers_are_noop_outside_a_process_group():
    assert not is_ddp_active()
    assert ddp_rank() == 0
    assert ddp_world_size() == 1
    module = _TinyModule(value=5.0)
    broadcast_module_state(module, src=0)
    assert torch.equal(module.weight, torch.full((3,), 5.0))  # untouched
    value = torch.tensor(3.0)
    assert torch.equal(allreduce_mean(value), value)
    shutdown_ddp()  # must not raise outside a process group


def _shutdown_ddp_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    assert is_ddp_active()
    shutdown_ddp()
    assert not is_ddp_active()


def test_shutdown_ddp_destroys_the_process_group():
    _spawn(_shutdown_ddp_worker, world_size=2)


PRIVILEGED_DIM = 5
ACTION_DIM = 2


class _FakeBoxEnv:
    def __init__(self, num_envs: int = 4, episode_len: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self._t = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _obs(self):
        return torch.randn(self.num_envs, PRIVILEGED_DIM)

    def reset(self, seed=None):
        del seed
        self._t.zero_()
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._t[terminated] = 0
        return self._obs(), reward, terminated, truncated, {}


def _ppo_target_kl_raises_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    assert is_ddp_active()
    try:
        PPO(_FakeBoxEnv(), device="cpu", target_kl=0.1, eval_freq=0, log_freq=0)
        raised = False
    except ValueError as exc:
        raised = "target_kl" in str(exc)
    assert raised
    dist.destroy_process_group()


def test_ppo_raises_when_target_kl_set_under_ddp():
    _spawn(_ppo_target_kl_raises_worker, world_size=2)


def _ppo_ddp_e2e_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    base_seed = 7
    agent = PPO(
        _FakeBoxEnv(num_envs=4),
        device="cpu",
        seed=base_seed + rank,  # rank-diversified, mirrors _runner.py's effective_seed
        num_steps=6,
        num_minibatches=2,
        update_epochs=2,
        target_kl=None,
        eval_freq=0,
        log_freq=0,
        net_arch=[16],
    )
    broadcast_module_state(agent.policy, src=0)
    for p in agent.policy.parameters():
        assert p.requires_grad

    agent.learn(total_timesteps=agent.num_steps * agent.num_envs * 3)

    # Gather rank 0's final flattened weights and compare against this rank's.
    local_flat = torch.cat([p.detach().reshape(-1) for p in agent.policy.parameters()])
    gathered = [torch.zeros_like(local_flat) for _ in range(world_size)] if rank == 0 else None
    dist.gather(local_flat, gather_list=gathered, dst=0)
    if rank == 0:
        for other in gathered[1:]:
            assert torch.allclose(gathered[0], other, atol=1e-5), (
                "PPO policy weights diverged across DDP ranks after training -- "
                "grad-allreduce insertion point is not wired correctly."
            )
    dist.destroy_process_group()


def test_ppo_ddp_end_to_end_weights_converge_across_ranks():
    _spawn(_ppo_ddp_e2e_worker, world_size=2)
