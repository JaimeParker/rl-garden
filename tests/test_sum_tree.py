from __future__ import annotations

import torch

from rl_garden.buffers.sum_tree import SumTree


def test_update_propagates_to_root_sum():
    tree = SumTree(capacity=8, alpha=1.0, beta=1.0, device=torch.device("cpu"))
    indices = torch.arange(8)
    td_errors = torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    tree.update(indices, td_errors)

    leaf_priorities = (td_errors.abs() + tree.eps) ** tree.alpha
    assert torch.allclose(tree.total, leaf_priorities.sum().to(torch.float64), atol=1e-6)


def test_update_is_idempotent_for_repeated_calls():
    tree = SumTree(capacity=4, alpha=0.9, beta=0.6, device=torch.device("cpu"))
    tree.update(torch.arange(4), torch.tensor([1.0, 1.0, 1.0, 1.0]))
    first_total = float(tree.total.item())
    tree.update(torch.tensor([2]), torch.tensor([1.0]))
    assert float(tree.total.item()) == first_total


def test_sample_matches_priority_distribution_statistically():
    torch.manual_seed(0)
    tree = SumTree(capacity=4, alpha=1.0, beta=0.0, device=torch.device("cpu"), eps=0.0)
    # Priorities proportional to [1, 2, 3, 4] -- index 3 should be sampled ~4x as
    # often as index 0.
    tree.update(torch.arange(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))

    generator = torch.Generator().manual_seed(0)
    leaf_indices, _ = tree.sample(20_000, generator=generator)
    counts = torch.bincount(leaf_indices, minlength=4).float()
    empirical = counts / counts.sum()
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]) / 10.0

    assert torch.allclose(empirical, expected, atol=0.02)


def test_sample_is_weights_bounded_by_one_via_local_min():
    """IS weight = (priority/batch_min_priority)^-beta; since priority >= the
    batch-local min, weights are bounded ABOVE by 1 (not below)."""
    torch.manual_seed(0)
    tree = SumTree(capacity=4, alpha=1.0, beta=0.5, device=torch.device("cpu"))
    tree.update(torch.arange(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))

    _, is_weights = tree.sample(1000, generator=torch.Generator().manual_seed(1))
    assert torch.all(is_weights <= 1.0 + 1e-4)


def test_set_uninitialized_uses_current_max_priority():
    tree = SumTree(capacity=4, alpha=1.0, beta=1.0, device=torch.device("cpu"), eps=0.0)
    tree.update(torch.tensor([0, 1]), torch.tensor([1.0, 5.0]))
    max_before = tree.tree[tree._leaf_offset : tree._leaf_offset + 4].max().item()

    tree.set_uninitialized(torch.tensor([2, 3]))

    leaf_priorities = tree.tree[tree._leaf_offset : tree._leaf_offset + 4]
    assert torch.allclose(leaf_priorities[2], torch.tensor(max_before, dtype=torch.float64))
    assert torch.allclose(leaf_priorities[3], torch.tensor(max_before, dtype=torch.float64))


def test_set_uninitialized_defaults_to_one_when_tree_empty():
    tree = SumTree(capacity=4, alpha=1.0, beta=1.0, device=torch.device("cpu"))
    tree.set_uninitialized(torch.tensor([0]))
    assert float(tree.total.item()) == 1.0


def test_capacity_non_power_of_two_builds_valid_tree():
    tree = SumTree(capacity=5, alpha=1.0, beta=1.0, device=torch.device("cpu"), eps=0.0)
    tree.update(torch.arange(5), torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(tree.total, torch.tensor(5.0, dtype=torch.float64))
    leaf_indices, _ = tree.sample(100, generator=torch.Generator().manual_seed(0))
    assert torch.all(leaf_indices < 5)
    assert torch.all(leaf_indices >= 0)
