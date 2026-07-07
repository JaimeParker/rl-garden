"""Cal-QL algorithm layer.

Cal-QL extends CQL by lower-bounding OOD Q-values with Monte Carlo returns from
the replay sample. The rest of the SAC/REDQ/CQL update path is inherited from
``CQL``.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.cql import CQL, _CQLRolloutTrainingShell
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.buffers.mc_buffer import MCDictReplayBuffer, MCTensorReplayBuffer


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
        obs_space = self.env.single_observation_space
        kwargs = {
            "observation_space": obs_space,
            "action_space": self.env.single_action_space,
            "num_envs": self.num_envs,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "storage_device": self.buffer_device,
            "sample_device": self.device,
            "sparse_reward_mc": self.sparse_reward_mc,
            "sparse_negative_reward": self.sparse_negative_reward,
            "success_threshold": self.success_threshold,
        }
        if isinstance(obs_space, spaces.Dict):
            return MCDictReplayBuffer(**kwargs)
        return MCTensorReplayBuffer(**kwargs)

    def _calql_lower_bound(
        self,
        q_ood: torch.Tensor,
        mc_returns: torch.Tensor,
        n_samples: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        bound_rate = (q_ood < mc_lower_bound).float().mean().detach()
        return torch.maximum(q_ood, mc_lower_bound), bound_rate


class _CalQLRolloutTrainingShell(Off2OnReplayMixin, CalQLCore, _CQLRolloutTrainingShell):
    """Internal rollout/eval shell that wires ``CalQLCore`` into ``OffPolicyAlgorithm``.

    Generic offline->online transition mechanics (replay-buffer switching,
    mixed-batch sampling, checkpoint/probe/logging plumbing) are inherited
    from ``Off2OnReplayMixin``. This class adds only what's Cal-QL-specific:
    the online CQL-alpha override (``online_cql_alpha``/``online_use_cql_loss``,
    via ``_apply_online_regularizer_override``) and the Q-value offline probe
    (``_offline_probe_metrics``, which needs ``_critic_forward``/``_target_q``).

    .. warning::
       **Do not instantiate this class directly.** It exists only to back
       :class:`~rl_garden.algorithms.WSRL` and
       :class:`~rl_garden.algorithms.Off2OnCalQL` by attaching the Cal-QL loss
       core to an off-policy rollout/replay/eval loop. For standalone offline
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
        online_cql_alpha: float = 0.0,
        online_use_cql_loss: bool = False,
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
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
        self.online_cql_alpha = online_cql_alpha
        self.online_use_cql_loss = online_use_cql_loss
        self._init_off2on_params(offline_sampling=offline_sampling)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "online_cql_alpha": self.online_cql_alpha,
            "online_use_cql_loss": self.online_use_cql_loss,
        }

    def _apply_online_regularizer_override(self, online_replay_mode: str) -> None:
        self.use_cql_loss = self.online_use_cql_loss
        self.cql_alpha = self.online_cql_alpha
        if self.use_cql_loss and online_replay_mode == "empty":
            warnings.warn(
                "switch_to_online_mode: use_cql_loss=True with "
                "online_replay_mode='empty' is not a configuration the WSRL or "
                "Cal-QL papers cover. CQL conservatism is calibrated against the "
                "offline data distribution; clearing the buffer removes that "
                "support and leaves CQL fighting policy gradients with high-variance "
                "OOD estimates over warmup-only data. Pass --online_use_cql_loss "
                "False for paper-aligned WSRL, or --online_replay_mode mixed/append "
                "to retain offline data for Cal-QL.",
                UserWarning,
                stacklevel=2,
            )
        if self.logger:
            self.logger.add_summary("cql/online_use_cql_loss", self.use_cql_loss)
            self.logger.add_summary("cql/online_cql_alpha", self.cql_alpha)
            self.logger.add_summary("cql/online_backup_entropy", self.backup_entropy)

        # torch.compile traces cql_alpha/use_cql_loss as constants; retrace
        # after this online-side change so the compiled critic loss matches.
        if self.use_compile and self._eager_critic_loss is not None:
            if self.logger:
                self.logger.add_summary("cql/recompile_at_online_step", self._global_step)
            self._apply_compile()

    def _offline_probe_metrics(self) -> dict[str, float]:
        if self._offline_probe_batch is None:
            return {}
        with torch.no_grad():
            data = self._offline_probe_batch
            q_pred = self._critic_forward(data.obs, data.actions, target=False)
            target_q = self._target_q(data)
            target_q_expanded = target_q.unsqueeze(0).repeat(self.n_critics, 1, 1)
            td_mse = F.mse_loss(q_pred, target_q_expanded)
        return {
            "offline_probe/predicted_q": float(q_pred.mean().item()),
            "offline_probe/target_q": float(target_q.mean().item()),
            "offline_probe/td_rmse": float(torch.sqrt(td_mse).item()),
        }


class CalQL(CalQLCore, CQL):
    """Pure offline CQL with Cal-QL MC lower bounds."""

    _compatible_checkpoint_algorithms = ("CalQL", "CQL")

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
