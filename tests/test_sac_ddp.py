"""Tests for rl_garden/algorithms/sac_ddp.py's single-node multi-GPU DDP
support for SAC.

Mirrors tests/test_ddp.py's pattern: torch.multiprocessing.spawn with the
gloo backend for real multi-process torch.distributed groups on CPU. Target
functions must be importable at module level (spawn pickles them), so every
worker body is a top-level function, not a nested one.
"""
from __future__ import annotations

import socket
from contextlib import closing

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.algorithms.sac_ddp import SACDDP
from rl_garden.common.ddp import allreduce_param_grads, broadcast_module_state


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


class _DummyVecEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, 2)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, 2)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        obs = torch.randn(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _make_agent(cls, seed: int = 0, **overrides):
    kwargs = dict(
        env=_DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        seed=seed,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# allreduce_param_grads
# ---------------------------------------------------------------------------


def _allreduce_param_grads_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    p1 = torch.nn.Parameter(torch.zeros(3))
    p2 = torch.nn.Parameter(torch.zeros(2))
    p1.grad = torch.full((3,), float(rank + 1))  # 1.0, 2.0 -> mean 1.5
    p2.grad = torch.full((2,), float(rank + 1) * 10)  # 10, 20 -> mean 15
    allreduce_param_grads([p1, p2])
    assert torch.allclose(p1.grad, torch.full((3,), 1.5))
    assert torch.allclose(p2.grad, torch.full((2,), 15.0))
    dist.destroy_process_group()


def test_allreduce_param_grads_two_processes():
    _spawn(_allreduce_param_grads_worker, world_size=2)


def test_allreduce_param_grads_noop_outside_a_process_group():
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.full((3,), 5.0)
    allreduce_param_grads([p])
    assert torch.equal(p.grad, torch.full((3,), 5.0))  # untouched


# ---------------------------------------------------------------------------
# _ddp_extra_broadcast_modules
# ---------------------------------------------------------------------------


def test_sac_ddp_extra_broadcast_modules_returns_alpha_tuner():
    agent = _make_agent(SACDDP)
    assert agent.alpha_tuner is not None
    assert agent._ddp_extra_broadcast_modules() == [agent.alpha_tuner]


def test_plain_sac_extra_broadcast_modules_is_empty():
    agent = _make_agent(SAC)
    assert agent._ddp_extra_broadcast_modules() == []


def _alpha_broadcast_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    agent = _make_agent(SACDDP, seed=7 + rank)
    # log_alpha's init value is a deterministic constant (log(init_value)),
    # not seed-dependent -- mutate it per-rank first to simulate the
    # post-training divergence the broadcast is meant to fix.
    with torch.no_grad():
        agent.alpha_tuner.log_alpha.fill_(float(rank + 1))  # rank0->1.0, rank1->2.0
    for module in agent._ddp_extra_broadcast_modules():
        broadcast_module_state(module, src=0)
    assert torch.equal(agent.alpha_tuner.log_alpha, torch.full((1,), 1.0))
    dist.destroy_process_group()


def test_alpha_tuner_broadcast_syncs_log_alpha_across_ranks():
    _spawn(_alpha_broadcast_worker, world_size=2)


# ---------------------------------------------------------------------------
# _sync_ddp_grads call-count/argument check (single-process, no DDP needed --
# this pins "6 sites, right params each" independent of whether the
# allreduce itself is exercised, which the end-to-end test below covers)
# ---------------------------------------------------------------------------


class _RecordingSACDDP(SACDDP):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sync_calls: list[int] = []

    def _sync_ddp_grads(self, params: list) -> None:
        self.sync_calls.append(len(params))


def test_sync_ddp_grads_called_three_times_per_train_call():
    agent = _make_agent(_RecordingSACDDP, utd=0.5)
    agent.learn(total_timesteps=8)  # populate the buffer via a real rollout
    agent.sync_calls.clear()

    critic_n = len(list(agent.policy.critic_and_encoder_parameters()))
    actor_n = len(list(agent.policy.actor_parameters()))
    alpha_n = len(list(agent._alpha_parameters()))

    agent.train(gradient_steps=1, compute_info=False)
    assert agent.sync_calls == [critic_n, actor_n, alpha_n]


def test_sync_ddp_grads_called_three_times_per_train_high_utd_call():
    agent = _make_agent(_RecordingSACDDP, utd=0.5)
    agent.learn(total_timesteps=8)
    agent.sync_calls.clear()

    critic_n = len(list(agent.policy.critic_and_encoder_parameters()))
    actor_n = len(list(agent.policy.actor_parameters()))
    alpha_n = len(list(agent._alpha_parameters()))

    agent.train_high_utd(utd_ratio=2, compute_info=False)
    assert agent.sync_calls == [critic_n, critic_n, actor_n, alpha_n]


# ---------------------------------------------------------------------------
# End-to-end: SACDDP converges, plain SAC (contrast) does not
# ---------------------------------------------------------------------------


def _gather_flat_params(module: torch.nn.Module, rank: int, world_size: int):
    local = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    gathered = [torch.zeros_like(local) for _ in range(world_size)] if rank == 0 else None
    dist.gather(local, gather_list=gathered, dst=0)
    return gathered


def _sac_ddp_e2e_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    agent = _make_agent(SACDDP, seed=7 + rank, utd=0.5)
    broadcast_module_state(agent.policy, src=0)
    for module in agent._ddp_extra_broadcast_modules():
        broadcast_module_state(module, src=0)

    agent.learn(total_timesteps=32)

    policy_gathered = _gather_flat_params(agent.policy, rank, world_size)
    alpha_gathered = _gather_flat_params(agent.alpha_tuner, rank, world_size)
    if rank == 0:
        for other in policy_gathered[1:]:
            assert torch.allclose(policy_gathered[0], other, atol=1e-5), (
                "SACDDP policy weights diverged across DDP ranks after "
                "training -- grad-allreduce insertion is not wired correctly."
            )
        for other in alpha_gathered[1:]:
            assert torch.allclose(alpha_gathered[0], other, atol=1e-5)
    dist.destroy_process_group()


def test_sac_ddp_end_to_end_weights_converge_across_ranks():
    _spawn(_sac_ddp_e2e_worker, world_size=2)


def _plain_sac_contrast_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    agent = _make_agent(SAC, seed=7 + rank, utd=0.5)
    # Broadcast policy weights the same way the SACDDP arm does, so the only
    # variable between this test and the SACDDP one is whether gradients get
    # synced during training -- not whether initial weights matched.
    broadcast_module_state(agent.policy, src=0)

    agent.learn(total_timesteps=32)

    policy_gathered = _gather_flat_params(agent.policy, rank, world_size)
    if rank == 0:
        diverged = any(
            not torch.allclose(policy_gathered[0], other, atol=1e-5)
            for other in policy_gathered[1:]
        )
        assert diverged, (
            "plain SAC's weights matched across ranks after training under "
            "an active DDP group -- _sync_ddp_grads's no-op default is not "
            "actually inert."
        )
    dist.destroy_process_group()


def test_plain_sac_does_not_converge_across_ranks_under_ddp():
    _spawn(_plain_sac_contrast_worker, world_size=2)


# ---------------------------------------------------------------------------
# build_sac: transparent class selection + mmap_dir rank-salting
# ---------------------------------------------------------------------------


def _build_sac_worker(rank: int, world_size: int, port: int, tmp_dir: str) -> None:
    _init_gloo(rank, world_size, port)
    import os
    from unittest.mock import patch

    from rl_garden.training.online.sac import SACArgs, build_sac

    # construct_agent(target, **kwargs) just calls target(**kwargs); patching
    # it to capture (target, kwargs) instead of actually constructing a real
    # SAC/SACDDP tests build_sac's own new class-selection/mmap_dir logic in
    # isolation, without needing a real Dict-obs vision pipeline (mmap_dir is
    # only valid for Dict/image observation spaces -- see sac.py:590-594 --
    # which this test's plain Box _DummyVecEnv deliberately doesn't provide,
    # since that restriction is pre-existing SAC behavior, not part of this
    # feature).
    captured: dict = {}

    def fake_construct_agent(target, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs
        return object()

    mmap_dir = os.path.join(tmp_dir, "buf")
    args = SACArgs(
        obs_mode="state", mmap_dir=mmap_dir, mmap_mode="create",
        eval_freq=0, log_freq=0, buffer_device="cpu", buffer_size=32,
        batch_size=8, learning_starts=1,
    )
    with patch("rl_garden.training.inspection.construct_agent", fake_construct_agent):
        build_sac(args, _DummyVecEnv(), None, None, None)
    assert captured["target"].__name__ == "SACDDP"
    assert captured["kwargs"]["mmap_dir"] == os.path.join(mmap_dir, f"rank{rank}")

    open_args = SACArgs(
        obs_mode="state", mmap_dir=mmap_dir, mmap_mode="open",
        eval_freq=0, log_freq=0, buffer_device="cpu", buffer_size=32,
        batch_size=8, learning_starts=1,
    )
    try:
        with patch("rl_garden.training.inspection.construct_agent", fake_construct_agent):
            build_sac(open_args, _DummyVecEnv(), None, None, None)
        raised = False
    except SystemExit:
        raised = True
    assert raised
    dist.destroy_process_group()


def test_build_sac_selects_ddp_class_and_rank_salts_mmap_dir(tmp_path):
    _spawn(_build_sac_worker, world_size=2, extra_args=(str(tmp_path),))


def test_build_sac_selects_plain_sac_outside_ddp():
    from rl_garden.training.online.sac import SACArgs, build_sac

    args = SACArgs(
        obs_mode="state", eval_freq=0, log_freq=0, buffer_device="cpu",
        buffer_size=32, batch_size=8, learning_starts=1,
    )
    agent = build_sac(args, _DummyVecEnv(), None, None, None)
    assert type(agent).__name__ == "SAC"
