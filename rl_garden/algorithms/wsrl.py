"""WSRL warm-start flow built on Cal-QL."""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rl_garden.algorithms.calql import _CalQLRolloutTrainingShell
from rl_garden.buffers.mc_buffer import MCReplayBufferSample
from rl_garden.common.logger import Logger


class WSRL(_CalQLRolloutTrainingShell):
    """Cal-QL plus offline→online replay switching and WSRL logging."""

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        # Buffer and training
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "always",
        # Optimizers
        policy_lr: float = 1e-4,
        q_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        cql_alpha_lr: float = 3e-4,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        use_compile: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
        # Entropy
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        backup_entropy: bool = False,
        # Network architecture
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[str] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        # Q-ensemble (REDQ)
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        # CQL parameters
        use_cql_loss: bool = True,
        cql_n_actions: int = 10,
        cql_alpha: float = 5.0,
        cql_autotune_alpha: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        cql_target_action_gap: float = 1.0,
        cql_importance_sample: bool = True,
        cql_max_target_backup: bool = True,
        cql_temp: float = 1.0,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_action_sample_method: str = "uniform",
        # Cal-QL parameters
        use_calql: bool = True,
        calql_bound_random_actions: bool = False,
        # WSRL phase control
        use_td_loss: bool = True,
        online_cql_alpha: Optional[float] = None,
        online_use_cql_loss: Optional[bool] = None,
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        # Sparse-reward MC
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
        # General
        policy_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            policy_lr=policy_lr,
            q_lr=q_lr,
            alpha_lr=alpha_lr,
            cql_alpha_lr=cql_alpha_lr,
            policy_frequency=policy_frequency,
            target_network_frequency=target_network_frequency,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            use_compile=use_compile,
            compile_mode=compile_mode,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            use_cql_loss=use_cql_loss,
            cql_n_actions=cql_n_actions,
            cql_alpha=cql_alpha,
            cql_autotune_alpha=cql_autotune_alpha,
            cql_alpha_lagrange_init=cql_alpha_lagrange_init,
            cql_target_action_gap=cql_target_action_gap,
            cql_importance_sample=cql_importance_sample,
            cql_max_target_backup=cql_max_target_backup,
            cql_temp=cql_temp,
            cql_clip_diff_min=cql_clip_diff_min,
            cql_clip_diff_max=cql_clip_diff_max,
            cql_action_sample_method=cql_action_sample_method,
            use_calql=use_calql,
            calql_bound_random_actions=calql_bound_random_actions,
            sparse_reward_mc=sparse_reward_mc,
            sparse_negative_reward=sparse_negative_reward,
            success_threshold=success_threshold,
            use_td_loss=use_td_loss,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self.online_cql_alpha = online_cql_alpha
        self.online_use_cql_loss = online_use_cql_loss
        self.offline_sampling: Literal["with_replace", "without_replace"] = offline_sampling
        self._online_start_step: int | None = None
        self._offline_probe_batch: MCReplayBufferSample | None = None
        self.offline_replay_buffer: Optional[Any] = None
        self.offline_data_ratio: float = 0.0

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "online_cql_alpha": self.online_cql_alpha,
            "online_use_cql_loss": self.online_use_cql_loss,
            "offline_sampling": self.offline_sampling,
        }

    def _clear_replay_buffer(self) -> int:
        previous_len = len(self.replay_buffer)
        self.replay_buffer.pos = 0
        self.replay_buffer.full = False
        if hasattr(self.replay_buffer, "_mc_table"):
            self.replay_buffer._mc_table = None
        return previous_len

    def _sample_batch(self, batch_size: int) -> MCReplayBufferSample:
        in_offline_phase = self._online_start_step is None
        if in_offline_phase and self.offline_sampling == "without_replace":
            return self.replay_buffer.sample_without_repeat(batch_size)
        if (
            not in_offline_phase
            and self.offline_replay_buffer is not None
            and self.offline_data_ratio > 0.0
        ):
            return self._sample_mixed_batch(batch_size)
        return self.replay_buffer.sample(batch_size)

    def _sample_train_batch(self, batch_size: int) -> MCReplayBufferSample:
        return self._sample_batch(batch_size)

    def _sample_mixed_batch(self, batch_size: int) -> MCReplayBufferSample:
        ratio = self.offline_data_ratio
        n_online = batch_size - int(round(batch_size * ratio))
        online_size = len(self.replay_buffer)
        if online_size == 0:
            return self.offline_replay_buffer.sample(batch_size)
        n_offline = batch_size - n_online
        if n_online == 0:
            return self.offline_replay_buffer.sample(batch_size)
        if n_offline == 0:
            return self.replay_buffer.sample(batch_size)

        online_sample = self.replay_buffer.sample(n_online)
        offline_sample = self.offline_replay_buffer.sample(n_offline)
        return self._concat_replay_samples(online_sample, offline_sample)

    @staticmethod
    def _concat_replay_samples(
        a: MCReplayBufferSample, b: MCReplayBufferSample
    ) -> MCReplayBufferSample:
        def _cat(x, y):
            if isinstance(x, dict):
                return {k: torch.cat([x[k], y[k]], dim=0) for k in x}
            return torch.cat([x, y], dim=0)

        return MCReplayBufferSample(
            obs=_cat(a.obs, b.obs),
            next_obs=_cat(a.next_obs, b.next_obs),
            actions=_cat(a.actions, b.actions),
            rewards=_cat(a.rewards, b.rewards),
            dones=_cat(a.dones, b.dones),
            mc_returns=_cat(a.mc_returns, b.mc_returns),
        )

    @staticmethod
    def canonical_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
        out = dict(metrics)
        success = metrics.get("success_at_end", metrics.get("success_once"))
        if success is not None:
            out["normalized_score"] = float(success) * 100.0
        return out

    @staticmethod
    def _update_metric_tags(metrics: dict[str, float]) -> dict[str, float]:
        """Map raw metric keys to namespaced paths using Logger's mapping."""
        tagged: dict[str, float] = {}
        namespaces = Logger.OFFLINE_METRIC_NAMESPACES

        for key, value in metrics.items():
            # Use explicit mapping or default to losses namespace
            full_path = namespaces.get(key, f"losses/{key}")
            tagged[full_path] = value

        # Derived metric: TD RMSE from TD loss
        if "td_loss" in metrics:
            tagged["q/td_rmse"] = float(np.sqrt(max(metrics["td_loss"], 0.0)))
        return tagged

    def set_offline_probe_batch(self, batch: MCReplayBufferSample | None) -> None:
        self._offline_probe_batch = batch

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
            "q/offline_probe/predicted": float(q_pred.mean().item()),
            "q/offline_probe/target": float(target_q.mean().item()),
            "q/offline_probe/td_rmse": float(torch.sqrt(td_mse).item()),
        }

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in self.canonical_eval_metrics(metrics).items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _log_rollout_metric(self, key: str, value: float, step: int) -> None:
        if self.logger is None:
            return
        self.logger.add_scalar(f"train/{key}", value, step)
        if key in {"success_at_end", "success_once"}:
            self.logger.add_scalar("train/normalized_score", value * 100.0, step)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for tag, value in self._update_metric_tags(metrics).items():
            self.logger.add_scalar(tag, value, step)

        online_start = self._online_start_step
        is_online = online_start is not None and step >= online_start
        offline_step = min(step, online_start) if online_start is not None else step
        online_step = max(0, step - online_start) if online_start is not None else 0
        self.logger.add_scalar("phase/is_online", float(is_online), step)
        self.logger.add_scalar("phase/offline_step", float(offline_step), step)
        self.logger.add_scalar("phase/online_step", float(online_step), step)

        if is_online:
            for tag, value in self._offline_probe_metrics().items():
                self.logger.add_scalar(tag, value, step)

    def switch_to_online_mode(
        self,
        online_replay_mode: Literal["empty", "append", "mixed"] = "append",
        offline_data_ratio: float = 0.0,
    ) -> None:
        self._online_start_step = self._global_step
        if self.online_use_cql_loss is not None:
            self.use_cql_loss = self.online_use_cql_loss
        if self.online_cql_alpha is not None:
            self.cql_alpha = self.online_cql_alpha

        if not (0.0 <= offline_data_ratio <= 1.0):
            raise ValueError(f"offline_data_ratio must be in [0, 1]; got {offline_data_ratio}.")
        self.offline_replay_buffer = None
        self.offline_data_ratio = 0.0
        cleared_transitions = 0
        if online_replay_mode == "empty":
            cleared_transitions = self._clear_replay_buffer()
        elif online_replay_mode == "append":
            pass
        elif online_replay_mode == "mixed":
            self.offline_replay_buffer = self.replay_buffer
            self.replay_buffer = self._build_replay_buffer()
            self.offline_data_ratio = offline_data_ratio
        else:
            raise ValueError(f"Unknown online_replay_mode: {online_replay_mode!r}")

        if self.logger:
            self.logger.add_summary("wsrl/online_start_step", self._global_step)
            self.logger.add_summary("wsrl/online_use_cql_loss", self.use_cql_loss)
            self.logger.add_summary("wsrl/online_cql_alpha", self.cql_alpha)
            self.logger.add_summary("wsrl/online_replay_mode", online_replay_mode)
            self.logger.add_summary("wsrl/online_replay_cleared", online_replay_mode == "empty")
            if online_replay_mode == "empty":
                self.logger.add_summary("wsrl/online_replay_size_before_clear", cleared_transitions)
            if online_replay_mode == "mixed":
                self.logger.add_summary("wsrl/offline_data_ratio", offline_data_ratio)

        if self.use_compile and self._eager_critic_loss is not None:
            self._apply_compile()
