"""Monte Carlo return computation for replay buffers.

Extends existing replay buffers (TensorReplayBuffer, DictReplayBuffer) with
on-the-fly MC return computation for Cal-QL. Tracks episode boundaries and
computes discounted returns when sampling batches.

Key features:
- Mixin pattern: works with both Tensor and Dict buffers
- Efficient episode boundary tracking using done flags
- On-the-fly MC return computation during sampling
- GPU-native: all computations stay on device
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
    - On-the-fly MC return computation during sampling
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
        # Call parent __init__
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=storage_device,
            sample_device=sample_device,
        )
        self.gamma = gamma

    def _compute_mc_returns(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        """Compute Monte Carlo returns for sampled transitions.

        For each sampled transition (t, env), computes:
            G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{T-t}*r_T
        where T is the end of the episode (done=1).

        Args:
            batch_inds: Time indices in buffer (batch_size,)
            env_inds: Environment indices (batch_size,)

        Returns:
            MC returns tensor of shape (batch_size,)
        """
        batch_size = batch_inds.shape[0]
        mc_returns = torch.zeros(batch_size, device=self.storage_device)

        # Process each sample individually to handle episode boundaries
        for i in range(batch_size):
            t_idx = batch_inds[i].item()
            env_idx = env_inds[i].item()

            # Find episode end (next done=1 or buffer end)
            upper = self.per_env_buffer_size if self.full else self.pos

            # Compute discounted return from t_idx to episode end
            mc_return = 0.0
            discount = 1.0

            for step in range(t_idx, upper):
                reward = self.rewards[step, env_idx].item()
                mc_return += discount * reward
                discount *= self.gamma

                # Stop at episode boundary
                if self.dones[step, env_idx].item() > 0.5:
                    break

            # If we wrapped around (circular buffer), continue from start
            if self.full and step == upper - 1 and self.dones[step, env_idx].item() < 0.5:
                for step in range(0, t_idx):
                    reward = self.rewards[step, env_idx].item()
                    mc_return += discount * reward
                    discount *= self.gamma

                    if self.dones[step, env_idx].item() > 0.5:
                        break

            mc_returns[i] = mc_return

        return mc_returns

    def sample(self, batch_size: int) -> MCReplayBufferSample:
        """Sample batch with MC returns computed on-the-fly."""
        upper = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper, size=(batch_size,), device=self.storage_device)
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,), device=self.storage_device)

        # Compute MC returns
        mc_returns = self._compute_mc_returns(batch_inds, env_inds)

        # Get standard replay buffer sample
        if hasattr(self, 'obs') and isinstance(self.obs, torch.Tensor):
            # TensorReplayBuffer path
            obs_sample = self.obs[batch_inds, env_inds].to(self.sample_device)
            next_obs_sample = self.next_obs[batch_inds, env_inds].to(self.sample_device)
        else:
            # DictReplayBuffer path
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
