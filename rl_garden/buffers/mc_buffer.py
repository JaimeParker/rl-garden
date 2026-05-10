"""Monte Carlo return computation for replay buffers.

Extends existing replay buffers (TensorReplayBuffer, DictReplayBuffer) with
on-the-fly MC return computation for Cal-QL. Tracks episode boundaries and
computes discounted returns when sampling batches.

Key features:
- Mixin pattern: works with both Tensor and Dict buffers
- Episode boundary tracking via done flags
- Vectorized GPU-native MC return table (built lazily, invalidated on add())
- ~100× faster than per-sample loop on large buffers
- Optional sparse-reward MC handling: failed episodes use infinite-horizon
  approximation ``r_neg / (1 - γ)`` (mirrors the WSRL/Cal-QL reference's
  ``calc_return_to_go`` for antmaze/adroit/kitchen-style environments).
"""
from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.types import MCReplayBufferSample, Obs, TensorDict


class MCReplayBufferMixin:
    """Mixin to add MC return computation to replay buffers.

    This mixin extends the base replay buffer with:
    - Episode boundary tracking via done flags
    - Lazy vectorized MC return table (cached, invalidated on add())
    - Efficient GPU-native implementation
    - Optional sparse-reward MC handling

    Usage:
        class MCTensorReplayBuffer(MCReplayBufferMixin, TensorReplayBuffer):
            pass
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        gamma: float = 0.99,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=storage_device,
            sample_device=sample_device,
        )
        self.gamma = gamma
        self.sparse_reward_mc = sparse_reward_mc
        self.sparse_negative_reward = sparse_negative_reward
        self.success_threshold = success_threshold
        # Cached MC return table; invalidated whenever add() is called.
        # Shape: (per_env_buffer_size, num_envs).
        self._mc_table: torch.Tensor | None = None
        # Per-step success flag (only allocated when sparse_reward_mc is True).
        # Stored as the same dtype as rewards for cheap arithmetic.
        if sparse_reward_mc:
            self._step_success = torch.zeros(
                (self.per_env_buffer_size, num_envs),
                device=self.storage_device,
                dtype=torch.float32,
            )
        else:
            self._step_success = None

    # ------------------------------------------------------------------
    # Cache invalidation: any add() invalidates the MC table.
    # ------------------------------------------------------------------

    def add(
        self,
        *args,
        success: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Add a transition. Optional ``success`` tensor records per-step success.

        ``success`` should be a (num_envs,) tensor (bool or float) marking which
        envs achieved success at this step. If not provided and
        ``sparse_reward_mc`` is enabled, success is inferred from
        ``reward >= success_threshold``.
        """
        self._mc_table = None
        if self.sparse_reward_mc and self._step_success is not None:
            if success is None:
                # Best-effort fallback: infer from current reward signal
                reward = (
                    args[3] if len(args) >= 4 else kwargs.get("reward")
                )
                if reward is None:
                    raise ValueError(
                        "sparse_reward_mc=True requires either explicit success= "
                        "or a positional reward arg in add()."
                    )
                success_tensor = (
                    reward.to(self.storage_device) >= self.success_threshold
                ).to(self._step_success.dtype)
            else:
                success_tensor = success.to(self.storage_device).to(
                    self._step_success.dtype
                )
            self._step_success[self.pos] = success_tensor
        return super().add(*args, **kwargs)

    # ------------------------------------------------------------------
    # Vectorized MC return table.
    # ------------------------------------------------------------------

    def _chronological_order(self) -> torch.Tensor:
        T = self.per_env_buffer_size
        if self.full:
            return torch.cat(
                [
                    torch.arange(self.pos, T, device=self.storage_device),
                    torch.arange(0, self.pos, device=self.storage_device),
                ]
            )
        return torch.arange(0, self.pos, device=self.storage_device)

    def _build_mc_table(self) -> torch.Tensor:
        """Build full (T, N) MC return table via a single backward sweep.

        Standard recurrence:
            G_T = r_T
            G_t = r_t + γ * G_{t+1} * (1 - done_t)

        With ``sparse_reward_mc`` enabled, transitions belonging to a *failed*
        episode (no step in [t, episode_end] is marked success) are assigned
        the infinite-horizon value ``r_neg / (1 - γ)`` instead.
        """
        T = self.per_env_buffer_size
        rewards = self.rewards  # (T, N)
        dones = self.dones      # (T, N)
        order = self._chronological_order()

        mc = torch.zeros_like(rewards)
        running = torch.zeros(self.num_envs, device=self.storage_device, dtype=rewards.dtype)

        if self.sparse_reward_mc and self._step_success is not None:
            inf_horizon_value = self.sparse_negative_reward / (1.0 - self.gamma)
            # acc_succ[t] tracks suffix episode success: OR of step_success
            # over [t, episode_end]. For sparse-reward envs where success only
            # occurs at the terminal step, this equals "episode succeeded".
            acc_succ = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.storage_device
            )
            for t in torch.flip(order, dims=(0,)):
                step_succ_t = self._step_success[t] > 0.5
                done_mask = dones[t] > 0.5
                # Reset accumulator when crossing into a new (older) episode.
                acc_succ = step_succ_t | (acc_succ & ~done_mask)
                running = rewards[t] + self.gamma * running * (1.0 - dones[t])
                mc[t] = torch.where(
                    acc_succ,
                    running,
                    torch.full_like(running, inf_horizon_value),
                )
        else:
            for t in torch.flip(order, dims=(0,)):
                running = rewards[t] + self.gamma * running * (1.0 - dones[t])
                mc[t] = running

        return mc

    def _compute_mc_returns(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        """Compute Monte Carlo returns for sampled transitions via cached table."""
        if self._mc_table is None:
            self._mc_table = self._build_mc_table()
        return self._mc_table[batch_inds, env_inds]

    # ------------------------------------------------------------------
    # Sample with MC returns attached.
    # ------------------------------------------------------------------

    def _index_batch(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> MCReplayBufferSample:
        """Override host buffer's ``_index_batch`` to attach MC returns."""
        base_sample = super()._index_batch(batch_inds, env_inds)
        # Index the MC table on storage_device for cache locality, then move.
        storage_batch_inds = batch_inds.to(self.storage_device)
        storage_env_inds = env_inds.to(self.storage_device)
        mc_returns = self._compute_mc_returns(storage_batch_inds, storage_env_inds)
        return MCReplayBufferSample(
            obs=base_sample.obs,
            next_obs=base_sample.next_obs,
            actions=base_sample.actions,
            rewards=base_sample.rewards,
            dones=base_sample.dones,
            mc_returns=mc_returns.to(self.sample_device),
        )

    def sample(self, batch_size: int) -> MCReplayBufferSample:
        """Sample batch with MC returns computed via cached table."""
        upper = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper, size=(batch_size,), device=self.storage_device)
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,), device=self.storage_device)
        return self._index_batch(batch_inds, env_inds)


class MCTensorReplayBuffer(MCReplayBufferMixin, TensorReplayBuffer):
    """TensorReplayBuffer with Monte Carlo return computation.

    Usage:
        buffer = MCTensorReplayBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_envs=16,
            buffer_size=1_000_000,
            gamma=0.99,
        )
        sample = buffer.sample(256)
        # sample.mc_returns contains MC returns for Cal-QL
    """

    pass


class MCDictReplayBuffer(MCReplayBufferMixin, DictReplayBuffer):
    """DictReplayBuffer with Monte Carlo return computation.

    Usage:
        buffer = MCDictReplayBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_envs=16,
            buffer_size=1_000_000,
            gamma=0.99,
        )
        sample = buffer.sample(256)
        # sample.mc_returns contains MC returns for Cal-QL
    """

    pass
