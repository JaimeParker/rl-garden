"""Vectorized proportional priority sum-tree (Schaul et al. 2016), torch-native.

Ported from the R2D2 reference clone's numpy ``PriorityTree``
(``3rd_party/R2D2/priority_tree.py``), generalized to run entirely as torch tensor
ops on the buffer's storage device -- no numpy handoffs in the replay hot path.
"""
from __future__ import annotations

from typing import Optional

import torch

from rl_garden.buffers.base import restored_storage


class SumTree:
    """Flat, complete-binary-tree-backed sum-tree over ``capacity`` leaves.

    ``update``/``sample`` operate in one shot over the whole batch -- no Python
    loop over individual leaves. A small ``eps`` is added before exponentiating in
    ``update`` so a legitimately-zero TD-error sample never acquires priority 0
    and gets permanently excluded from sampling (not present in the R2D2
    reference, added here as a safety fix, not a fidelity deviation).
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta: float,
        device: torch.device,
        eps: float = 1e-3,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = torch.device(device)

        num_layers = 1
        while capacity > 2 ** (num_layers - 1):
            num_layers += 1
        self.num_layers = num_layers
        self._leaf_offset = 2 ** (num_layers - 1) - 1
        self.tree = torch.zeros(
            2**num_layers - 1, dtype=torch.float64, device=self.device
        )

    def _propagate(self, idxes: torch.Tensor) -> None:
        """``idxes``: absolute tree indices of leaves just written. Walks to the
        root, recomputing each parent as the sum of its two children. Duplicate
        indices at a given level always compute the same value from already-fixed
        children, so repeated writes under ``tensor[idxes] = values`` are safe
        without a ``np.unique``-style dedup pass (unlike the numpy reference)."""
        for _ in range(self.num_layers - 1):
            idxes = (idxes - 1) // 2
            self.tree[idxes] = self.tree[2 * idxes + 1] + self.tree[2 * idxes + 2]

    def update(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """``indices``: leaf-relative (``0..capacity-1``) LongTensor. ``td_errors``:
        same shape, raw (signed) TD errors -- priority is computed here as
        ``(|td_error| + eps) ** alpha``."""
        priorities = (
            td_errors.detach().abs().to(torch.float64) + self.eps
        ) ** self.alpha
        idxes = indices.to(device=self.device, dtype=torch.long) + self._leaf_offset
        self.tree[idxes] = priorities
        self._propagate(idxes)

    def set_uninitialized(
        self, indices: torch.Tensor, priority: Optional[float] = None
    ) -> None:
        """Seed freshly-created leaves at ``indices`` with the current max leaf
        priority (or 1.0 if the tree is entirely empty), so new experience is
        guaranteed a chance to be sampled before its first TD-error is known.
        Pass an explicit ``priority`` to override this default."""
        if priority is None:
            max_p = self.tree[self._leaf_offset : self._leaf_offset + self.capacity].max()
            fill = float(max_p.item()) if max_p > 0 else 1.0
        else:
            fill = float(priority)
        idxes = indices.to(device=self.device, dtype=torch.long) + self._leaf_offset
        self.tree[idxes] = fill
        self._propagate(idxes)

    def sample(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        *,
        num_stratified: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(leaf_indices, is_weights)``, both length ``num_samples``.

        The first ``num_stratified`` (default: all) samples take one from each
        of that many equal slices of the total priority, in ascending leaf
        order; the rest are independent."""
        p_sum = self.tree[0]
        if p_sum <= 0:
            raise RuntimeError(
                "SumTree.sample called with zero total priority -- no transitions "
                "have been added/updated yet."
            )
        if num_stratified is None or num_stratified == num_samples:
            interval = p_sum / num_samples
            offsets = (
                torch.arange(num_samples, dtype=torch.float64, device=self.device)
                * interval
            )
        else:
            slice_width = p_sum / max(num_stratified, 1)
            independent = num_samples - num_stratified
            interval = torch.cat((slice_width.expand(num_stratified), p_sum.expand(independent)))
            offsets = torch.cat(
                (
                    torch.arange(num_stratified, dtype=torch.float64, device=self.device)
                    * slice_width,
                    torch.zeros(independent, dtype=torch.float64, device=self.device),
                )
            )
        jitter = (
            torch.rand(
                num_samples, generator=generator, device=self.device, dtype=torch.float64
            )
            * interval
        )
        prefix_sums = offsets + jitter

        idxes = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        for _ in range(self.num_layers - 1):
            left_child = idxes * 2 + 1
            left_value = self.tree[left_child]
            go_left = prefix_sums < left_value
            idxes = torch.where(go_left, left_child, idxes * 2 + 2)
            went_right = idxes % 2 == 0
            prefix_sums = torch.where(
                went_right, prefix_sums - self.tree[idxes - 1], prefix_sums
            )

        leaf_indices = idxes - self._leaf_offset
        return leaf_indices, self.importance_weights(leaf_indices)

    def importance_weights(self, leaf_indices: torch.Tensor) -> torch.Tensor:
        """``(priority / smallest priority among leaf_indices) ** -beta``: the
        importance weights of a batch drawn in proportion to priority."""
        priorities = self.tree[leaf_indices + self._leaf_offset]
        min_p = priorities.min().clamp_min(self.eps)
        return (priorities / min_p).pow(-self.beta).to(torch.float32)

    def load_tree(self, tree: torch.Tensor) -> None:
        """Adopt the node values of a tree of the same capacity."""
        self.tree = restored_storage(self.tree, tree, "priority_tree")

    @property
    def total(self) -> torch.Tensor:
        return self.tree[0]
