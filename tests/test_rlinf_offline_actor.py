"""RLinf-independent tests for rl_garden.integrations.rlinf.offline_actor.

RLinf isn't required to run these: they cover the algorithm-dispatch table,
the fail-loudly Cal-QL rejection, and the D4RL-batch-to-rl-garden-batch
conversion helper -- all plain Python/torch, no ``rlinf`` import. Tests that
actually construct ``RLGardenOfflineActor`` (which requires RLinf) are not
here; see docs/design/rlinf-integration.md's verification tiers.
"""
from __future__ import annotations

import pytest
import torch

from rl_garden.algorithms import AWAC, BC, CQL, IQL, TD3BC
from rl_garden.integrations.rlinf.offline_actor import (
    _ALGORITHMS,
    _dataset_batch_to_sample,
    resolve_algorithm,
)


def test_algorithm_dispatch_covers_offline_uniform_contract():
    assert _ALGORITHMS == {
        "bc": BC,
        "iql": IQL,
        "cql": CQL,
        "awac": AWAC,
        "td3_bc": TD3BC,
    }


@pytest.mark.parametrize("name", ["bc", "iql", "cql", "awac", "td3_bc"])
def test_resolve_algorithm_returns_expected_class(name):
    assert resolve_algorithm(name) is _ALGORITHMS[name]


def test_resolve_algorithm_rejects_calql_by_name():
    with pytest.raises(ValueError, match="Cal-QL"):
        resolve_algorithm("calql")


def test_resolve_algorithm_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_algorithm("not_a_real_algorithm")


def test_dataset_batch_to_sample_inverts_masks_to_dones():
    batch = {
        "observations": torch.zeros(4, 3),
        "next_observations": torch.ones(4, 3),
        "actions": torch.full((4, 2), 0.5),
        "rewards": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        # masks = 1 - terminals: last transition is terminal (mask 0).
        "masks": torch.tensor([1.0, 1.0, 1.0, 0.0]),
    }
    sample = _dataset_batch_to_sample(batch)
    assert torch.equal(sample.obs, batch["observations"])
    assert torch.equal(sample.next_obs, batch["next_observations"])
    assert torch.equal(sample.actions, batch["actions"])
    assert torch.equal(sample.rewards, batch["rewards"])
    assert torch.equal(sample.dones, torch.tensor([0.0, 0.0, 0.0, 1.0]))
