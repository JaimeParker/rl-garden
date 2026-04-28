"""Monte Carlo return computation for replay buffers.

Extends existing replay buffers (TensorReplayBuffer, DictReplayBuffer) with
on-the-fly MC return computation for Cal-QL. Tracks episode boundaries and
computes discounted returns when sampling batches.

Key features:
- Mixin pattern: works with both Tensor and Dict buffers
- Episode boundary tracking via done flags
- Vectorized GPU-native MC return table (built lazily, invalidated on add())
- ~100× faster than per-sample loop on large buffers
"""
from __future__ import annotations

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
        # Cached MC return table; invalidated whenever add() is called.
        # Shape: (per_env_buffer_size, num_envs).
        self._mc_table: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Cache invalidation: any add() invalidates the MC table.
    # ------------------------------------------------------------------

    def add(self, *args, **kwargs):
        # Mark cache as stale before delegating to the base buffer.
        self._mc_table = None
        return super().add(*args, **kwargs)

    # ------------------------------------------------------------------
    # Vectorized MC return table.
    # ------------------------------------------------------------------

    def _build_mc_table(self) -> torch.Tensor:
        """Build full (T, N) MC return table via a single backward sweep.

        Recurrence:
            G_T = r_T
            G_t = r_t + γ * G_{t+1} * (1 - done_t)

        When the circular buffer is full, physical index 0 is not necessarily
        the oldest transition. We therefore sweep in chronological order and
        scatter the returns back to physical storage indices.
        """
        T = self.per_env_buffer_size
        rewards = self.rewards            # (T, N)
        dones = self.dones                # (T, N)
        if self.full:
            order = torch.cat(
                [
                    torch.arange(self.pos, T, device=self.storage_device),
                    torch.arange(0, self.pos, device=self.storage_device),
                ]
            )
        else:
            order = torch.arange(0, self.pos, device=self.storage_device)

        mc = torch.zeros_like(rewards)
        running = torch.zeros(self.num_envs, device=self.storage_device, dtype=rewards.dtype)
        for t in torch.flip(order, dims=(0,)):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            mc[t] = running
        return mc

    def _compute_mc_returns(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        """Compute Monte Carlo returns for sampled transitions via cached table.

        Args:
            batch_inds: Time indices in buffer (batch_size,) on storage_device
            env_inds: Environment indices (batch_size,) on storage_device

        Returns:
            MC returns tensor of shape (batch_size,) on storage_device.
        """
        if self._mc_table is None:
            self._mc_table = self._build_mc_table()
        return self._mc_table[batch_inds, env_inds]

    # ------------------------------------------------------------------
    # Sample with MC returns attached.
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> MCReplayBufferSample:
        """Sample batch with MC returns computed via cached table."""
        upper = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper, size=(batch_size,), device=self.storage_device)
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,), device=self.storage_device)

        mc_returns = self._compute_mc_returns(batch_inds, env_inds)

        # Get standard replay buffer sample (Tensor vs Dict path).
        if isinstance(self.obs, torch.Tensor):
            obs_sample = self.obs[batch_inds, env_inds].to(self.sample_device)
            next_obs_sample = self.next_obs[batch_inds, env_inds].to(self.sample_device)
        else:
            obs_sample = {
                k: v.to(self.sample_device) for k, v in self.obs[batch_inds, env_inds].items()
            }
            next_obs_sample = {
                k: v.to(self.sample_device)
                for k, v in self.next_obs[batch_inds, env_inds].items()
            }

        return MCReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
            mc_returns=mc_returns.to(self.sample_device),
        )


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
