"""Fixed-size per-rollout-window storage for DPPO's denoising chains.

Matches ``3rd_party/dppo/agent/finetune/train_ppo_diffusion_agent.py:83-91,
225-232`` exactly: alongside the standard ``RolloutBuffer``, DPPO needs the
full sampled DDPM chain (for the fine-tuned tail only) and each step's
per-denoising-step log-probs, neither of which fit ``RolloutBuffer``'s
``(T, N, ...)`` transition layout. A standalone tensor holder, not a
``BaseReplayBuffer`` subclass -- no sampling, no Dict-obs generality, no
``without_replace`` mode are needed here, matching "no abstractions for
single-use code". Same reset-fill-consume lifecycle as ``RolloutBuffer``:
``reset()`` at the start of each rollout window, ``add()`` once per rollout
step (auto-incrementing ``pos``, mirroring ``RolloutBuffer.add()``), read
back in full once the window is complete.
"""
from __future__ import annotations

import torch


class DiffusionChainBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        ft_denoising_steps: int,
        horizon_steps: int,
        action_dim: int,
        device: torch.device | str = "cuda",
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if ft_denoising_steps <= 0:
            raise ValueError(
                f"ft_denoising_steps must be positive, got {ft_denoising_steps}."
            )
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.ft_denoising_steps = ft_denoising_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.pos = 0
        self.full = False

        self.chains = torch.zeros(
            (num_steps, num_envs, ft_denoising_steps + 1, horizon_steps, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.old_log_probs = torch.zeros(
            (num_steps, num_envs, ft_denoising_steps, horizon_steps, action_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def add(self, chain: torch.Tensor, log_probs: torch.Tensor) -> None:
        """``chain``: (num_envs, ft_denoising_steps+1, horizon_steps,
        action_dim). ``log_probs``: (num_envs, ft_denoising_steps,
        horizon_steps, action_dim)."""
        if self.pos >= self.num_steps:
            raise RuntimeError(
                "DiffusionChainBuffer is full; call reset() before adding more."
            )
        self.chains[self.pos] = chain.to(self.device)
        self.old_log_probs[self.pos] = log_probs.to(self.device)
        self.pos += 1
        self.full = self.pos == self.num_steps
