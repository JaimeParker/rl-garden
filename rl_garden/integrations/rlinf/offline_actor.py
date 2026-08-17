"""RLinf offline actor adapter.

Drives rl-garden's offline algorithms inside RLinf's ``OfflineRunner``
(``3rd_party/RLinf/rlinf/runners/offline_runner.py``). See
``docs/design/rlinf-integration.md``, "Offline-uniform contract
(BC, IQL, CQL, AWAC, TD3+BC)" and "Design principle: adapters target the
hook contract".

Subclasses RLinf's plain ``Worker`` rather than ``EmbodiedFSDPActor``:
``OfflineRunner`` only duck-types 7 methods on its ``actor`` (typed ``Any``
there -- no base class beyond being launchable via ``Worker.create_group``
is required), while ``EmbodiedFSDPActor.__init__``/``init_worker()`` pull in
FSDP model-building, weight-sync, and component-placement machinery this
adapter doesn't use: it builds and trains an ordinary rl-garden
``nn.Module``/optimizer pair directly, never RLinf's own model factory.
Inheriting that machinery only to override all of it away would be the
opposite of the "smallest possible foothold" this adapter targets.

RLinf is optional: this module is importable without it (the class exists
with an ``object`` fallback base), but instantiating ``RLGardenOfflineActor``
requires it.
"""
from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rl_garden.algorithms import (
    AWAC,
    BC,
    CQL,
    IQL,
    TD3BC,
    OfflineEnvSpec,
    OfflineRLAlgorithm,
)
from rl_garden.common.types import ReplayBufferSample
from rl_garden.integrations.rlinf import require_rlinf

try:
    from rlinf.scheduler import Worker

    _RLINF_AVAILABLE = True
except ImportError:
    Worker = object  # type: ignore[assignment,misc]
    _RLINF_AVAILABLE = False


# The offline-uniform contract (docs/design/rlinf-integration.md,
# "Offline-uniform contract (BC, IQL, CQL, AWAC, TD3+BC)"). Cal-QL is
# deliberately excluded -- see that document's "Known, unscheduled" table:
# its mc_returns field needs its own data pipeline and silently degrades to
# plain CQL if omitted, so selecting it here raises rather than training
# wrong without warning.
_ALGORITHMS: dict[str, type[OfflineRLAlgorithm]] = {
    "bc": BC,
    "iql": IQL,
    "cql": CQL,
    "awac": AWAC,
    "td3_bc": TD3BC,
}


def resolve_algorithm(name: str) -> type[OfflineRLAlgorithm]:
    """Look up an offline-uniform-contract algorithm by config name.

    Raises ``ValueError`` for unknown names, including Cal-QL specifically:
    it is deliberately unsupported by this adapter (its ``mc_returns``
    field needs its own data pipeline and silently degrades to plain CQL if
    omitted -- see docs/design/rlinf-integration.md, "Known, unscheduled").
    Kept RLinf-independent and separate from :meth:`RLGardenOfflineActor.init_worker`
    so this dispatch/fail-loudly logic is testable without RLinf installed.
    """
    algo_cls = _ALGORITHMS.get(name)
    if algo_cls is None:
        valid = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(
            f"Unsupported cfg.actor.model.rlgarden_algorithm={name!r}. "
            f"Supported (offline-uniform contract): {valid}. "
            "Cal-QL is deliberately not supported by this adapter -- see "
            "docs/design/rlinf-integration.md, 'Known, unscheduled'."
        )
    return algo_cls


def _dataset_batch_to_sample(batch: dict[str, torch.Tensor]) -> ReplayBufferSample:
    """Convert one RLinf D4RL ``DataLoader`` batch into rl-garden's batch shape.

    RLinf's ``D4RLDataset`` (``3rd_party/RLinf/rlinf/data/datasets/d4rl.py``)
    yields ``masks`` = 1 - terminals (the not-done convention); rl-garden's
    ``ReplayBufferSample.dones`` is the terminal flag itself.
    """
    return ReplayBufferSample(
        obs=batch["observations"],
        next_obs=batch["next_observations"],
        actions=batch["actions"],
        rewards=batch["rewards"],
        dones=1.0 - batch["masks"],
    )


class RLGardenOfflineActor(Worker):
    """RLinf offline actor that delegates training to a wrapped rl-garden algorithm.

    Implements the 7-method contract RLinf's ``OfflineRunner`` drives:
    ``init_worker``, ``load_checkpoint``, ``sync_model_to_rollout``,
    ``set_global_step``, ``run_training``, ``save_checkpoint``, and
    ``.worker_group_name``.

    Per "Design principle: adapters target the hook contract" in
    ``docs/design/rlinf-integration.md``, this class never references a
    concrete algorithm by name outside ``_ALGORITHMS`` -- the concrete
    algorithm is selected entirely by
    ``cfg.actor.model.rlgarden_algorithm``.
    """

    def __init__(self, cfg: Any) -> None:
        require_rlinf()
        Worker.__init__(self)

        self.cfg = cfg
        self.worker_group_name = cfg.actor.group_name
        self._algo: OfflineRLAlgorithm | None = None
        self._data_loader: DataLoader | None = None
        self._data_iter = None
        self._gradient_steps_per_call = int(
            cfg.actor.model.get("gradient_steps_per_call", 1)
        )

    def init_worker(self) -> None:
        from rlinf.data.datasets.d4rl import build_d4rl_dataset_from_cfg

        dataset = build_d4rl_dataset_from_cfg(self.cfg)
        obs_dim, action_dim = dataset.get_obs_action_dims()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"Failed to infer obs_dim/action_dim from offline dataset "
                f"(got obs_dim={obs_dim}, action_dim={action_dim})."
            )

        env_spec = OfflineEnvSpec(
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            action_space=spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            ),
        )

        model_cfg = self.cfg.actor.model
        algo_cls = resolve_algorithm(model_cfg.rlgarden_algorithm)
        algo_kwargs = dict(model_cfg.get("rlgarden_algorithm_kwargs", {}))

        # buffer_size=1: the replay buffer this constructs is never
        # populated or sampled from -- _sample_train_batch is overridden
        # below to pull from RLinf's own per-rank DataLoader instead.
        self._algo = algo_cls(env=env_spec, buffer_size=1, **algo_kwargs)

        per_rank_batch_size = int(model_cfg.get("batch_size", self._algo.batch_size))
        if per_rank_batch_size != self._algo.batch_size:
            raise ValueError(
                f"cfg.actor.model.batch_size ({per_rank_batch_size}) must match "
                f"the algorithm's own batch_size ({self._algo.batch_size}) -- "
                "the DataLoader batch size and the batch size "
                "_sample_train_batch is called with must agree."
            )

        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=bool(self.cfg.data.get("shuffle", True)),
            seed=int(self.cfg.data.get("seed", 42)),
            drop_last=True,
        )
        self._data_loader = DataLoader(
            dataset,
            batch_size=per_rank_batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=int(self.cfg.data.get("num_workers", 0)),
            pin_memory=True,
        )
        self._data_iter = iter(self._data_loader)

        if not hasattr(self._algo, "_sample_train_batch"):
            raise AttributeError(
                f"{algo_cls.__name__} has no _sample_train_batch to override -- "
                "rl-garden's offline-uniform contract may have changed; "
                "see docs/design/rlinf-integration.md."
            )
        self._algo._sample_train_batch = types.MethodType(
            lambda _self, _batch_size: self._next_batch(), self._algo
        )

    def _next_batch(self) -> ReplayBufferSample:
        assert self._data_iter is not None, "init_worker() must run before training."
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            batch = next(self._data_iter)
        batch = {k: v.to(self._algo.device) for k, v in batch.items()}
        return _dataset_batch_to_sample(batch)

    def run_training(self) -> dict[str, float]:
        assert self._algo is not None, "init_worker() must run before run_training()."
        return self._algo.train(
            gradient_steps=self._gradient_steps_per_call, compute_info=True
        )

    def save_checkpoint(self, path: str, step: int) -> None:
        del step  # rl-garden's save() uses its own tracked _global_step.
        assert self._algo is not None, "init_worker() must run before save_checkpoint()."
        self._algo.save(path)

    def load_checkpoint(self, path: str) -> None:
        assert self._algo is not None, "init_worker() must run before load_checkpoint()."
        self._algo.load(path)

    def set_global_step(self, step: int) -> None:
        # No public setter exists on BaseAlgorithm for this; _global_step is
        # otherwise only advanced internally by rl-garden's own loops.
        assert self._algo is not None, "init_worker() must run before set_global_step()."
        self._algo._global_step = int(step)

    async def sync_model_to_rollout(self) -> None:
        """Stub: this adapter targets the eval-disabled offline path only.

        ``OfflineRunner`` only calls this when eval is enabled
        (``runner.val_check_interval > 0`` or ``runner.only_eval``) --
        Phase 1's scope (docs/design/rlinf-integration.md) explicitly
        targets eval disabled, matching the "smallest possible foothold".
        """
        return
