"""Single-node multi-GPU DDP primitives: hand-rolled, not
``torch.nn.parallel.DistributedDataParallel``.

Ports rsl_rl's approach (``3rd_party/rsl_rl/rsl_rl/algorithms/ppo.py``): a
one-time weight broadcast at startup, then a manual grad-flatten +
``all_reduce`` + scatter-back after every ``backward()`` instead of wrapping
the model. Every function here is a no-op when
``torch.distributed.is_initialized()`` is False (i.e. the process was not
launched under ``torchrun``), so callers can invoke them unconditionally with
zero behavior change for the single-process case.

Algorithm-agnostic on purpose (not PPO-specific): any future off-policy DDP
work can reuse these same primitives.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Iterable

import torch
import torch.distributed as dist


def is_ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    return dist.get_rank() if is_ddp_active() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if is_ddp_active() else 1


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def init_ddp() -> None:
    """Initialize the NCCL process group when launched under ``torchrun``.

    No-op when ``WORLD_SIZE`` is unset or ``<= 1`` (plain ``python`` launch)
    or a process group is already initialized. Pins this process to its
    local GPU (``torch.cuda.set_device``) before doing anything else, since
    later device-resolution (``get_device("auto")``, ManiSkill's index-less
    ``"gpu"`` backend string) both follow "current CUDA device."
    """
    if is_ddp_active():
        return
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return
    torch.cuda.set_device(ddp_local_rank())
    dist.init_process_group(backend="nccl")


def shutdown_ddp() -> None:
    """Destroy the process group initialized by ``init_ddp``, if any.

    No-op unless DDP is active. Without this, NCCL warns ``destroy_process_
    group() was not called before program exit, which can leak resources``
    on every DDP run -- confirmed via a real 2-GPU ``torchrun`` smoke test,
    not caught by any single-process or ``gloo`` test (those call
    ``dist.destroy_process_group()`` explicitly in the test body itself).
    """
    if is_ddp_active():
        dist.destroy_process_group()


def pin_backend_config_device(backend_config: Any, local_rank: int) -> None:
    """Rewrite an env backend config's device fields to this rank's GPU.

    No-op unless DDP is active, so a single-process user's explicit device
    choice (e.g. ``--isaaclab.sim-device cuda:1``) is never overridden.

    Generic across backend config dataclasses in ``common/env_args.py``:
    rewrites any field literally named ``device``/``sim_device`` holding a
    ``"cuda"``/``"cuda:N"`` string, and any field named
    ``sim_backend``/``render_backend`` holding a ``"gpu"``/``"gpu:N"``
    string (ManiSkill's convention). Fields with any other name or value are
    left untouched -- a future backend's differently-named device field
    needs its name added here, not a redesign.
    """
    if not is_ddp_active():
        return
    for field in dataclasses.fields(backend_config):
        value = getattr(backend_config, field.name, None)
        if not isinstance(value, str):
            continue
        if field.name in ("device", "sim_device") and value.split(":")[0] == "cuda":
            setattr(backend_config, field.name, f"cuda:{local_rank}")
        elif field.name in ("sim_backend", "render_backend") and value.split(":")[0] == "gpu":
            setattr(backend_config, field.name, f"gpu:{local_rank}")


def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast ``module``'s parameters/buffers from rank ``src`` to all
    other ranks, in place. No-op unless DDP is active.

    Deliberately per-tensor ``dist.broadcast``, not ``broadcast_object_list``:
    the latter pickles its payload into a tensor and needs the current CUDA
    device set correctly under NCCL -- a class of bug that only surfaces
    under a real multi-GPU launch. Per-tensor broadcast sidesteps the
    object-pickling path entirely. ``state_dict()`` tensors share storage
    with the module's own parameters/buffers, so broadcasting them in place
    updates the module directly -- no ``load_state_dict()`` call needed.
    """
    if not is_ddp_active():
        return
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=src)


def allreduce_param_grads(params: Iterable[torch.nn.Parameter]) -> None:
    """Average a set of parameters' gradients across all ranks, in place.

    No-op unless DDP is active. Direct port of rsl_rl's
    ``reduce_parameters()``: flattens every given param's ``.grad``
    (skipping params with no grad) into one tensor, ``all_reduce(SUM)``,
    divides by ``ddp_world_size()``, scatters back into each param's
    ``.grad``.

    Takes a plain parameter iterable rather than a single ``nn.Module``
    because some optimizers' trainable parameters span multiple submodules
    (e.g. SAC's critic head plus a separately-owned shared encoder,
    ``SACPolicy.critic_and_encoder_parameters()``) -- see ``allreduce_grads``
    below for the common single-module case.
    """
    if not is_ddp_active():
        return
    params = [p for p in params if p.grad is not None]
    if not params:
        return
    grads = [p.grad.view(-1) for p in params]
    flat = torch.cat(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= ddp_world_size()
    offset = 0
    for p in params:
        numel = p.grad.numel()
        p.grad.copy_(flat[offset : offset + numel].view_as(p.grad))
        offset += numel


def allreduce_grads(module: torch.nn.Module) -> None:
    """Average ``module``'s parameter gradients across all ranks, in place.

    No-op unless DDP is active. Thin wrapper around
    ``allreduce_param_grads(module.parameters())`` for the common case where
    an optimizer's trainable parameters all live on one ``nn.Module``.
    """
    allreduce_param_grads(module.parameters())


def allreduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks. No-op unless DDP is active."""
    if not is_ddp_active():
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / ddp_world_size()
