"""Tests for RecedingHorizonPolicy: chunk-then-slice rollout wrapper."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.policies.diffusion_policy import DiffusionPolicy
from rl_garden.policies.receding_horizon import RecedingHorizonPolicy


class _StubChunkedPolicy:
    """Deterministic stand-in: predict() returns arange-valued chunks and
    counts calls, so tests can assert exactly when resampling happens."""

    def __init__(self, horizon_steps: int, cond_steps: int, action_dim: int = 1) -> None:
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.action_dim = action_dim
        self.call_count = 0

    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        del obs, deterministic
        self.call_count += 1
        batch = 2
        chunk = torch.arange(self.horizon_steps, dtype=torch.float32)
        chunk = chunk.view(1, self.horizon_steps, 1).expand(batch, -1, self.action_dim).clone()
        # Tag which call produced this chunk, so tests can distinguish
        # "still draining the first chunk" from "resampled".
        chunk += self.call_count * 100
        return chunk


def test_drains_n_action_steps_before_resampling():
    stub = _StubChunkedPolicy(horizon_steps=8, cond_steps=1, action_dim=1)
    wrapper = RecedingHorizonPolicy(stub, n_action_steps=3)

    for _ in range(3):
        wrapper.predict(torch.zeros(2, 4))
    assert stub.call_count == 1  # one chunk covered all 3 actions

    wrapper.predict(torch.zeros(2, 4))
    assert stub.call_count == 2  # 4th call exhausts n_action_steps=3, resamples


def test_action_sequence_matches_execution_window():
    stub = _StubChunkedPolicy(horizon_steps=6, cond_steps=2, action_dim=1)
    wrapper = RecedingHorizonPolicy(stub, n_action_steps=2)
    start = stub.cond_steps - 1  # = 1

    a0 = wrapper.predict(torch.zeros(2, 4))
    a1 = wrapper.predict(torch.zeros(2, 4))
    assert torch.allclose(a0[:, 0], torch.full((2,), float(start) + 100))
    assert torch.allclose(a1[:, 0], torch.full((2,), float(start + 1) + 100))

    # Third call exhausts the window (n_action_steps=2) -> resample -> index
    # `start` again, tagged with the new call count.
    a2 = wrapper.predict(torch.zeros(2, 4))
    assert torch.allclose(a2[:, 0], torch.full((2,), float(start) + 200))


def test_reset_chunk_forces_immediate_resample():
    stub = _StubChunkedPolicy(horizon_steps=8, cond_steps=1, action_dim=1)
    wrapper = RecedingHorizonPolicy(stub, n_action_steps=5)

    wrapper.predict(torch.zeros(2, 4))
    assert stub.call_count == 1
    wrapper.reset_chunk()
    wrapper.predict(torch.zeros(2, 4))
    assert stub.call_count == 2


def test_n_action_steps_bounds_validated():
    stub = _StubChunkedPolicy(horizon_steps=6, cond_steps=2, action_dim=1)
    max_valid = stub.horizon_steps - (stub.cond_steps - 1)  # 5
    RecedingHorizonPolicy(stub, n_action_steps=max_valid)  # ok, boundary
    try:
        RecedingHorizonPolicy(stub, n_action_steps=max_valid + 1)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        RecedingHorizonPolicy(stub, n_action_steps=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_wraps_real_diffusion_policy_end_to_end():
    torch.manual_seed(0)
    obs_dim, action_dim = 3, 2
    policy = DiffusionPolicy(
        observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        horizon_steps=4,
        cond_steps=1,
        denoising_steps=5,
        mlp_dims=[16, 16, 16],
    )
    wrapper = RecedingHorizonPolicy(policy, n_action_steps=2)
    obs = torch.randn(3, obs_dim)
    with torch.no_grad():
        for _ in range(5):
            action = wrapper.predict(obs, deterministic=True)
            assert action.shape == (3, action_dim)
            assert (action >= -1.0 - 1e-5).all() and (action <= 1.0 + 1e-5).all()
