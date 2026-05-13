"""Cal-QL algorithm layer.

Cal-QL extends CQL by lower-bounding OOD Q-values with Monte Carlo returns from
the replay sample. The rest of the SAC/REDQ/CQL update path is inherited from
``CQL``.
"""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.algorithms.cql import CQL, _CQLRolloutTrainingShell
from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer


class CalQLCore:
    """Shared Cal-QL replay and lower-bound behavior."""

    def _init_calql_params(
        self,
        *,
        use_calql: bool = True,
        calql_bound_random_actions: bool = False,
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
    ) -> None:
        self.use_calql = use_calql
        self.calql_bound_random_actions = calql_bound_random_actions
        self.sparse_reward_mc = sparse_reward_mc
        self.sparse_negative_reward = sparse_negative_reward
        self.success_threshold = success_threshold

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "use_calql": self.use_calql,
            "calql_bound_random_actions": self.calql_bound_random_actions,
            "sparse_reward_mc": self.sparse_reward_mc,
            "sparse_negative_reward": self.sparse_negative_reward,
            "success_threshold": self.success_threshold,
        }

    def _build_replay_buffer(self):
        return MCTensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
            sparse_reward_mc=self.sparse_reward_mc,
            sparse_negative_reward=self.sparse_negative_reward,
            success_threshold=self.success_threshold,
        )

    def _calql_lower_bound(
        self,
        q_ood: torch.Tensor,
        mc_returns: torch.Tensor,
        n_samples: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, float]:
        """Apply the Cal-QL infinite-horizon lower bound to OOD Q-values."""
        mc_returns_b1 = mc_returns.reshape(batch_size, 1)

        if self.calql_bound_random_actions:
            mc_lower_bound = mc_returns_b1.expand(batch_size, n_samples)
        else:
            fake = torch.full(
                (batch_size, self.cql_n_actions),
                float("-inf"),
                device=self.device,
                dtype=mc_returns_b1.dtype,
            )
            real = mc_returns_b1.expand(batch_size, 2 * self.cql_n_actions)
            mc_lower_bound = torch.cat([fake, real], dim=1)
        mc_lower_bound = mc_lower_bound.unsqueeze(0)

        num_vals = q_ood.numel()
        bound_rate = (q_ood < mc_lower_bound).sum().item() / max(num_vals, 1)
        return torch.maximum(q_ood, mc_lower_bound), bound_rate


class _CalQLRolloutTrainingShell(CalQLCore, _CQLRolloutTrainingShell):
    """Internal rollout/eval shell that wires ``CalQLCore`` into ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** It exists only to back
       :class:`~rl_garden.algorithms.WSRL` by attaching the Cal-QL loss core
       to an off-policy rollout/replay/eval loop. For standalone offline
       Cal-QL pretraining use :class:`CalQL`. The shape and arguments of this
       shell may change without notice.
    """

    def __init__(
        self,
        *args: Any,
        use_calql: bool = True,
        calql_bound_random_actions: bool = False,
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self._init_calql_params(
            use_calql=use_calql,
            calql_bound_random_actions=calql_bound_random_actions,
            sparse_reward_mc=sparse_reward_mc,
            sparse_negative_reward=sparse_negative_reward,
            success_threshold=success_threshold,
        )
        super().__init__(*args, **kwargs)
        self.use_calql = use_calql


class CalQL(CalQLCore, CQL):
    """Pure offline CQL with Cal-QL MC lower bounds."""

    def __init__(
        self,
        *args: Any,
        use_calql: bool = True,
        calql_bound_random_actions: bool = False,
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self._init_calql_params(
            use_calql=use_calql,
            calql_bound_random_actions=calql_bound_random_actions,
            sparse_reward_mc=sparse_reward_mc,
            sparse_negative_reward=sparse_negative_reward,
            success_threshold=success_threshold,
        )
        super().__init__(*args, **kwargs)
        self.use_calql = use_calql
