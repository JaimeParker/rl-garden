"""RLinf async SAC actor adapter.

Drives rl-garden's SAC/RLPD algorithms inside RLinf's ``AsyncEmbodiedRunner``
(``RLinf/rlinf/runners/async_embodied_runner.py``). See
``docs/design/rlinf-integration.md``, "SACCore contract (SAC, RLPD)".

Subclasses RLinf's plain ``Worker`` rather than ``EmbodiedSACFSDPPolicy``:
``EmbodiedSACFSDPPolicy.setup_model_and_optimizer`` unconditionally FSDP-wraps
the model (``FSDPStrategyBase.create`` only accepts ``"fsdp"``/``"fsdp2"``,
no plain-``nn.Module`` passthrough), and ``save_checkpoint``/
``load_checkpoint``/``sync_model_to_rollout`` all assume FSDP-shaped state.
This adapter builds and trains an ordinary rl-garden ``nn.Module``/optimizer
pair directly -- inheriting the FSDP machinery only to override all of it
away would be the opposite of the smallest foothold this adapter targets
(same reasoning as the offline adapter's choice not to subclass
``EmbodiedFSDPActor``).

RLinf is optional: this module is importable without it (the class exists
with an ``object`` fallback base), but instantiating ``RLGardenSACActor``
requires it.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

import torch

from rl_garden.algorithms import RLPD, SAC, OfflineEnvSpec, OffPolicyAlgorithm
from rl_garden.common.types import ReplayBufferSample
from rl_garden.integrations.rlinf import require_rlinf

try:
    from rlinf.scheduler import Worker

    _RLINF_AVAILABLE = True
except ImportError:
    Worker = object  # type: ignore[assignment,misc]
    _RLINF_AVAILABLE = False


# The SACCore contract (docs/design/rlinf-integration.md, "SACCore contract
# (SAC, RLPD)"). TD3/DrQv2 (the DDPG contract) and RLPDHybrid (a second
# optimizer path outside _critic_loss/_actor_loss) are deliberately excluded
# -- see that document's "Known, unscheduled" table.
_ALGORITHMS: dict[str, type[OffPolicyAlgorithm]] = {
    "sac": SAC,
    "rlpd": RLPD,
}


def resolve_algorithm(name: str) -> type[OffPolicyAlgorithm]:
    """Look up a SACCore-contract algorithm by config name.

    Raises ``ValueError`` for unknown names. Kept RLinf-independent and
    separate from :meth:`RLGardenSACActor.init_worker` so this dispatch
    logic is testable without RLinf installed (mirrors
    ``rl_garden.integrations.rlinf.offline_actor.resolve_algorithm``).
    """
    algo_cls = _ALGORITHMS.get(name)
    if algo_cls is None:
        valid = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(
            f"Unsupported cfg.actor.model.rlgarden_algorithm={name!r}. "
            f"Supported (SACCore contract): {valid}. TD3/DrQv2 (DDPG "
            "contract) and RLPDHybrid are deliberately not supported by "
            "this adapter -- see docs/design/rlinf-integration.md, "
            "'Known, unscheduled'."
        )
    return algo_cls


def build_env_spec(obs_dim: int, action_dim: int) -> OfflineEnvSpec:
    """Build a state-based ``OfflineEnvSpec`` from flat obs/action dims.

    Shared by the actor and rollout sides so both construct an identical
    policy -- required because ``PatchWeightSyncer.init_sender`` enforces
    exact state-dict key-set equality between sender and receiver.
    """
    import numpy as np
    from gymnasium import spaces

    return OfflineEnvSpec(
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        ),
        action_space=spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        ),
    )


def build_algo_from_cfg(cfg: Any) -> OffPolicyAlgorithm:
    """Construct a SAC/RLPD instance from ``cfg.actor.model``.

    Used by the actor (which trains the full algorithm) and, via
    ``.policy``, by the rollout worker (which only needs the network) --
    the same construction path on both sides guarantees an identical
    ``SACPolicy`` for weight-sync key-set equality, per
    ``docs/design/rlinf-integration.md``.
    """
    model_cfg = cfg.actor.model
    algo_cls = resolve_algorithm(model_cfg.rlgarden_algorithm)
    algo_kwargs = dict(model_cfg.get("rlgarden_algorithm_kwargs", {}))
    obs_dim = int(model_cfg.obs_dim)
    action_dim = int(model_cfg.action_dim)
    env_spec = build_env_spec(obs_dim, action_dim)
    return algo_cls(env=env_spec, **algo_kwargs)


def trajectory_batch_to_sample(
    batch: dict[str, Any], device: torch.device
) -> ReplayBufferSample:
    """Convert one ``TrajectoryReplayBuffer.sample(...)`` batch into rl-garden's shape.

    ``batch["terminations"]`` (not ``batch["dones"]``) is the source for
    rl-garden's ``dones``: RLinf's ``EnvOutput.dones = terminations |
    truncations`` (``rlinf/workers/env/env_worker.py:486,530-532``,
    unaffected by ``env.train.ignore_terminations``, which only gates
    episode-metric logging, not this tensor) conflates true termination
    with time-limit truncation. rl-garden's ``dones`` is specifically the
    bootstrap-suppression terminal flag (see ``bootstrap_at_done`` in
    ``rl_garden/algorithms/off_policy.py``) -- using the combined field
    would silently suppress bootstrapping at every truncation too.

    With ``num_action_chunks: 1`` (this adapter's only supported value, see
    the entry config), ``actions``/``rewards`` carry a size-1 chunk
    dimension that is squeezed here.
    """
    obs = batch["curr_obs"]["states"]
    next_obs = batch["next_obs"]["states"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["terminations"]
    if actions.dim() == 3:
        actions = actions.squeeze(1)
    if rewards.dim() == 2:
        rewards = rewards.squeeze(1)
    if dones.dim() == 2:
        dones = dones.squeeze(1)
    return ReplayBufferSample(
        obs=obs.to(device),
        next_obs=next_obs.to(device),
        actions=actions.to(device),
        rewards=rewards.to(device).float(),
        dones=dones.to(device).float(),
    )


class _TrajectoryReplayBufferShim:
    """Adapts RLinf's ``TrajectoryReplayBuffer`` to rl-garden's replay-buffer interface.

    Exactly the two members rl-garden's algorithm code calls:
    ``sample(batch_size)`` (``SACCore._sample_train_batch``,
    ``PriorDataReplayMixin._sample_train_batch``) and ``__len__()``
    (``PriorDataReplayMixin._sample_train_batch`` checks
    ``len(self.replay_buffer) == 0``).

    Deliberately not a monkey-patch of ``_sample_train_batch`` itself:
    ``RLPD`` overrides that method to mix online (this buffer) and offline
    data (``rl_garden/buffers/prior_data_replay.py:133-152``) -- patching
    the method away would silently make RLPD train as plain SAC with zero
    offline data. Swapping the *buffer object* instead means both
    ``SACCore``'s and ``PriorDataReplayMixin``'s own
    ``_sample_train_batch`` implementations route through this shim
    correctly, with zero adapter-side branching between SAC and RLPD.
    """

    def __init__(self, raw_buffer: Any, device: torch.device) -> None:
        self._raw = raw_buffer
        self._device = device

    def sample(self, batch_size: int) -> ReplayBufferSample:
        batch = self._raw.sample(batch_size)
        if not batch:
            raise RuntimeError(
                f"TrajectoryReplayBuffer.sample({batch_size}) returned an "
                "empty batch. Callers must check is_ready()/__len__() "
                "before sampling; a short/empty batch here would silently "
                "corrupt gradient-step batching."
            )
        actual = batch["rewards"].shape[0]
        if actual != batch_size:
            raise RuntimeError(
                f"TrajectoryReplayBuffer.sample({batch_size}) returned "
                f"{actual} transitions -- RLinf's sample_chunks() caps the "
                "return at the current window size instead of raising. "
                "Wait for more data before sampling this batch size."
            )
        return trajectory_batch_to_sample(batch, self._device)

    def __len__(self) -> int:
        return len(self._raw)


class RLGardenSACActor(Worker):
    """RLinf async actor that delegates training to a wrapped rl-garden SAC/RLPD instance.

    Implements the contract RLinf's ``AsyncEmbodiedRunner`` drives:
    ``init_worker``, ``load_checkpoint``, ``sync_model_to_rollout``,
    ``recv_rollout_trajectories``, ``run_training``, ``save_checkpoint``,
    ``stop``, and ``.worker_group_name``. ``set_global_step`` and
    ``compute_advantages_and_returns`` are never called by the async path
    (only by the sync ``EmbodiedRunner.run()``, which
    ``AsyncEmbodiedRunner.run()`` fully overrides) and are not implemented.

    Per "Design principle: adapters target the hook contract" in
    ``docs/design/rlinf-integration.md``, this class never references a
    concrete algorithm by name outside ``_ALGORITHMS`` -- the concrete
    algorithm is selected entirely by ``cfg.actor.model.rlgarden_algorithm``.
    """

    def __init__(self, cfg: Any) -> None:
        require_rlinf()
        Worker.__init__(self)

        self.cfg = cfg
        self.worker_group_name = cfg.actor.group_name
        self._algo: OffPolicyAlgorithm | None = None
        self._raw_replay_buffer: Any = None
        self._recv_queue: "queue.Queue[Any]" = queue.Queue()
        self._recv_rollout_thread: threading.Thread | None = None
        self.should_stop = False
        self._weight_syncer = None
        self._param_names_need_sync: list[str] = []
        self._version = 0
        self._gradient_steps_per_call = int(
            cfg.actor.model.get("gradient_steps_per_call", 1)
        )
        self._min_buffer_size = int(
            cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        )
        self._recv_drain_max_trajectories = int(
            cfg.actor.get("recv_drain_max_trajectories", 1024)
        )

    def init_worker(self) -> None:
        from rlinf.data.storage.replay import TrajectoryReplayBuffer
        from rlinf.hybrid_engines.weight_syncer import WeightSyncer
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.utils.utils import collect_param_names_need_sync

        self._component_placement = HybridComponentPlacement(self.cfg, Cluster())
        self._algo = build_algo_from_cfg(self.cfg)
        if not torch.device(self._algo.device).type == "cuda":
            raise RuntimeError(
                "RLGardenSACActor requires a CUDA device: PatchWeightSyncer's "
                "sender-side snapshot raises if the sender's state dict "
                "isn't already on the accelerator when snapshot_device=cpu."
            )

        import os

        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self._raw_replay_buffer = TrajectoryReplayBuffer(
            seed=self.cfg.actor.get("seed", 1234),
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )
        # Replace the buffer object (not _sample_train_batch) -- see
        # _TrajectoryReplayBufferShim's docstring for why this matters for RLPD.
        if not hasattr(self._algo, "replay_buffer"):
            raise AttributeError(
                f"{type(self._algo).__name__} has no replay_buffer attribute -- "
                "rl-garden's SACCore contract may have changed; see "
                "docs/design/rlinf-integration.md."
            )
        self._algo.replay_buffer = _TrajectoryReplayBufferShim(
            self._raw_replay_buffer, self._algo.device
        )

        patch_cfg = self.cfg.weight_syncer.get("patch", None)
        if patch_cfg is not None:
            init_sync_enabled = bool(patch_cfg.get("init_sync", {}).get("enabled", False))
            delta_encoding = bool(patch_cfg.get("delta_encoding", True))
            if delta_encoding and not init_sync_enabled:
                raise RuntimeError(
                    "cfg.weight_syncer.patch.delta_encoding=True requires "
                    "init_sync.enabled=True -- otherwise the rollout worker "
                    "starts from an independently random-initialized policy "
                    "and the actor's deltas never correct that base, "
                    "silently freezing the rollout at a wrong network."
                )
        self._weight_syncer = WeightSyncer.create(self.cfg.weight_syncer)
        self._param_names_need_sync = collect_param_names_need_sync(
            self._algo.policy
        )

    def load_checkpoint(self, path: str) -> None:
        assert self._algo is not None, "init_worker() must run before load_checkpoint()."
        self._algo.load(path)

    def save_checkpoint(self, path: str, step: int) -> None:
        del step  # rl-garden's save() uses its own tracked _global_step.
        assert self._algo is not None, "init_worker() must run before save_checkpoint()."
        self._algo.save(path)

    async def recv_rollout_trajectories(self, input_channel: Any) -> None:
        if self._recv_rollout_thread is None or not self._recv_rollout_thread.is_alive():
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel: Any) -> None:
        while not self.should_stop:
            trajectory = input_channel.get()
            self._recv_queue.put(trajectory)

    def _drain_received_trajectories(self) -> None:
        recv_list = []
        for _ in range(self._recv_drain_max_trajectories):
            try:
                recv_list.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break
        if recv_list:
            self._raw_replay_buffer.add_trajectories(recv_list)

    async def run_training(self) -> dict[str, float]:
        assert self._algo is not None, "init_worker() must run before run_training()."
        self._drain_received_trajectories()
        if not self._raw_replay_buffer.is_ready(self._min_buffer_size):
            return {}
        return self._algo.train(
            gradient_steps=self._gradient_steps_per_call, compute_info=True
        )

    async def sync_model_to_rollout(self) -> None:
        assert self._algo is not None, "init_worker() must run before sync_model_to_rollout()."
        rollout_group_name = self.cfg.rollout.group_name
        state_dict = self._algo.policy.state_dict()

        async def send_func(data: Any) -> None:
            if self._rank != 0:
                return
            rollout_world_size = self._component_placement.get_world_size("rollout")
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (rollout_group_name, list(range(rollout_world_size))),
                ],
                src=(self._group_name, 0),
                async_op=True,
            ).async_wait()

        async def recv_func() -> Any:
            return await self.recv(
                src_group_name=rollout_group_name,
                src_rank=0,
                async_op=True,
            ).async_wait()

        if not self._weight_syncer.sender_initialized():
            await self._weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self._param_names_need_sync,
                is_sender=(self._rank == 0),
            )
        await self._weight_syncer.sync(state_dict, send_func, version=self._version)
        self._version += 1

    async def stop(self) -> None:
        self.should_stop = True
        recv_thread = self._recv_rollout_thread
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
