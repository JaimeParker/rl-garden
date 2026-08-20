from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms._chunked_rollout import ChunkedRolloutMixin


class _BaseHarness:
    """Minimal stand-in for the `off_policy.py` hooks `ChunkedRolloutMixin`
    overrides, isolated from the full `OffPolicyAlgorithm`."""

    def __init__(self, num_envs: int, horizon_length: int, action_dim: int):
        self.num_envs = num_envs
        self.horizon_length = horizon_length
        self.device = torch.device("cpu")
        self.env = SimpleNamespace(
            single_action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        )
        self.explore_calls = 0

    def _explore_action(self, obs):
        self.explore_calls += 1
        return torch.full(
            (self.num_envs,) + self.env.single_action_space.shape, -1.0
        )

    def _obs_to_policy_device(self, obs):
        return obs

    def _on_env_reset(self, obs) -> None:
        pass

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        pass

    def _rollout_action(self, obs, learning_has_started: bool):
        actions = self._explore_action(obs)
        return actions, actions, None


class _Harness(ChunkedRolloutMixin, _BaseHarness):
    def __init__(self, num_envs: int, horizon_length: int, action_dim: int):
        super().__init__(num_envs, horizon_length, action_dim)
        self._init_chunked_rollout()
        self.sample_calls = 0

    def _sample_action_chunk(self, obs) -> torch.Tensor:
        self.sample_calls += 1
        base = float(self.sample_calls * 100)
        chunk = torch.arange(self.horizon_length, dtype=torch.float32).view(1, -1, 1) + base
        act_dim = self.env.single_action_space.shape[0]
        return chunk.expand(self.num_envs, self.horizon_length, act_dim).clone()


def _done(num_envs: int, true_indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(num_envs, dtype=torch.bool)
    for i in true_indices:
        mask[i] = True
    return mask


def test_replans_once_every_horizon_length_steps():
    h = _Harness(num_envs=1, horizon_length=3, action_dim=2)
    h._on_env_reset(obs=None)

    obs = None
    for _ in range(9):  # 3 full chunks
        actions, env_actions, ctx = h._rollout_action(obs, learning_has_started=True)
        assert torch.equal(actions, env_actions)
        h._post_rollout_step(ctx, _done(1, []), _done(1, []), infos={})

    assert h.sample_calls == 3
    assert h.explore_calls == 0


def test_bypasses_queue_before_learning_started():
    h = _Harness(num_envs=2, horizon_length=3, action_dim=2)
    h._on_env_reset(obs=None)

    for _ in range(5):
        h._rollout_action(obs=None, learning_has_started=False)

    assert h.sample_calls == 0
    assert h.explore_calls == 5


def test_staggered_reset_forces_immediate_per_env_replan():
    h = _Harness(num_envs=2, horizon_length=3, action_dim=1)
    h._on_env_reset(obs=None)

    # Step 1: both envs refill (cursor starts at horizon_length for both).
    a1, _, ctx = h._rollout_action(obs=None, learning_has_started=True)
    h._post_rollout_step(ctx, _done(2, []), _done(2, []), infos={})
    assert h.sample_calls == 1
    assert torch.allclose(a1, torch.tensor([[100.0], [100.0]]))  # chunk[0] of call #1

    # Step 2: env 0 terminates; env 1 does not.
    a2, _, ctx = h._rollout_action(obs=None, learning_has_started=True)
    assert torch.allclose(a2, torch.tensor([[101.0], [101.0]]))  # chunk[1] of call #1, both envs
    h._post_rollout_step(ctx, _done(2, [0]), _done(2, []), infos={})

    # Step 3: env 0's cursor was forced to horizon_length by the termination
    # -> triggers a fresh chunk (call #2) for BOTH envs (single batched
    # forward pass), but only env 0's queue slot is actually overwritten;
    # env 1 keeps consuming its original (call #1) cached chunk uninterrupted.
    a3, _, ctx = h._rollout_action(obs=None, learning_has_started=True)
    assert h.sample_calls == 2
    assert torch.allclose(a3[0], torch.tensor([200.0]))  # env 0: chunk[0] of call #2
    assert torch.allclose(a3[1], torch.tensor([102.0]))  # env 1: chunk[2] of call #1, undisturbed
    h._post_rollout_step(ctx, _done(2, []), _done(2, []), infos={})

    # Step 4: env 1 now exhausts its original chunk (cursor reaches
    # horizon_length) -> triggers call #3 for env 1; env 0 continues call #2.
    a4, _, ctx = h._rollout_action(obs=None, learning_has_started=True)
    assert h.sample_calls == 3
    assert torch.allclose(a4[0], torch.tensor([201.0]))  # env 0: chunk[1] of call #2
    assert torch.allclose(a4[1], torch.tensor([300.0]))  # env 1: chunk[0] of call #3
