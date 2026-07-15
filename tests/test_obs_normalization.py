"""Unit tests for ObsNormalizingMixin and its runner hooks.

Covers the mixin's numerical correctness in isolation, plus verifies that
the ``hasattr``-gated hooks added to ``run_offline``/``run_off2on`` are true
no-ops for algorithms that don't opt in (e.g. IQL/BC), so existing offline
algorithms see zero behavior change from Part 0's infrastructure.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl_garden.common.obs_normalization import ObsNormalizingMixin


class _DummyNormalizingModule(nn.Module, ObsNormalizingMixin):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self._register_obs_normalizer(obs_dim)


def test_buffers_registered_with_identity_defaults():
    mod = _DummyNormalizingModule(obs_dim=5)
    assert torch.equal(mod.obs_mean, torch.zeros(5))
    assert torch.equal(mod.obs_std, torch.ones(5))
    assert "obs_mean" in dict(mod.named_buffers())
    assert "obs_std" in dict(mod.named_buffers())


def test_fit_obs_normalizer_computes_dataset_mean_std():
    mod = _DummyNormalizingModule(obs_dim=3)
    obs = torch.randn(1000, 3) * 2.0 + torch.tensor([1.0, -2.0, 0.5])
    mod.fit_obs_normalizer(obs)
    assert torch.allclose(mod.obs_mean, obs.mean(0), atol=1e-5)
    assert torch.allclose(mod.obs_std, obs.std(0) + 1e-3, atol=1e-5)


def test_normalize_obs_applies_mean_std():
    mod = _DummyNormalizingModule(obs_dim=2)
    mod.obs_mean.copy_(torch.tensor([1.0, -1.0]))
    mod.obs_std.copy_(torch.tensor([2.0, 0.5]))
    obs = torch.tensor([[3.0, 0.0]])
    normalized = mod._normalize_obs(obs)
    assert torch.allclose(normalized, torch.tensor([[1.0, 2.0]]))


def test_round_trips_through_state_dict():
    mod = _DummyNormalizingModule(obs_dim=2)
    mod.fit_obs_normalizer(torch.randn(50, 2) * 3.0 + 1.0)
    state = mod.state_dict()

    fresh = _DummyNormalizingModule(obs_dim=2)
    fresh.load_state_dict(state)
    assert torch.equal(mod.obs_mean, fresh.obs_mean)
    assert torch.equal(mod.obs_std, fresh.obs_std)


def test_run_offline_hook_is_noop_for_algorithms_without_normalizer():
    """IQL/BC/CQL/CalQL have no ``fit_obs_normalizer`` method, so the
    ``hasattr``-gated hook added to ``run_offline`` never fires for them."""
    from rl_garden.algorithms import BC, OfflineEnvSpec
    from gymnasium import spaces
    import numpy as np

    env = OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    agent = BC(env=env, buffer_device="cpu", device="cpu")
    assert not hasattr(agent, "fit_obs_normalizer")
