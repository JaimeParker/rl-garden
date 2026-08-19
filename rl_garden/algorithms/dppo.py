"""DPPO: Diffusion PPO fine-tuning (phase 2, after ``DiffusionBC``).

Ported from ``3rd_party/dppo/agent/finetune/train_ppo_diffusion_agent.py`` +
``model/diffusion/diffusion_ppo.py::PPODiffusion.loss``, verified against
source directly. DDPM only; the reference's optional BC-regularization loss
term (``use_bc_loss``) is out of scope for this port -- it requires a full
non-subsampled ``get_logprobs`` pass over a *second*, freshly-sampled chain
per training step (roughly doubling the update's cost) for a term the
reference itself treats as optional and off in its own example configs.
Likewise the reference's ``entropy_loss = -eta.mean()`` term is omitted: for
DDPM, ``eta`` is hardcoded to ``torch.ones_like(mu)`` (a DDIM-only quantity),
making ``entropy_loss`` an algebraic constant with zero gradient -- and the
reference's own ``ent_coef`` defaults to 0 regardless.

``OnPolicyAlgorithm.learn()`` (the rollout loop, GAE call, checkpointing) is
reused completely unmodified -- see ``DPPOPolicy``'s module docstring and
this session's design notes: DPPO's critic is a pure function of ``obs``
(exactly like standard PPO's), so ``_rollout_step`` can compute real
bootstrap values inline, and ``compute_returns_and_advantage`` needs no
override. Only ``_rollout_step`` (DDPM sampling + chain-buffer fill) and
``_snapshot_window_initial_hidden`` (chain-buffer reset, hooked onto the one
call site that runs right after ``rollout_buffer.reset()``) are overridden;
``train()`` replaces PPO's clipped-surrogate update with DPPO's
denoising-step-aware one.

Action chunking (``horizon_steps``/``act_steps``) is handled entirely by
``rl_garden.envs.wrappers.ActionChunkWrapper`` around the env passed in --
this algorithm does not know or care that the env is chunked; it only knows
``self.env.single_action_space.shape == (act_steps, action_dim)``.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.buffers.diffusion_chain_buffer import DiffusionChainBuffer
from rl_garden.buffers.rollout_buffer import RolloutBuffer
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.logger import Logger
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.networks import Activation, KernelInit
from rl_garden.policies.dppo_policy import DPPOPolicy


class DPPOCore:
    """Shared DPPO loss/hyperparameter logic."""

    def _init_dppo_params(
        self,
        *,
        horizon_steps: int = 4,
        act_steps: int = 4,
        denoising_steps: int = 20,
        ft_denoising_steps: int = 10,
        actor_mlp_dims: Optional[Sequence[int]] = None,
        actor_activation_fn: Optional[Activation] = "relu",
        actor_residual_style: bool = True,
        critic_mlp_dims: Optional[Sequence[int]] = None,
        critic_activation_fn: Optional[Activation] = "mish",
        critic_residual_style: bool = True,
        time_dim: int = 16,
        kernel_init: Optional[KernelInit] = None,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        critic_warmup_updates: int = 0,
        update_epochs: int = 5,
        update_batch_size: int = 50_000,
        norm_adv: bool = True,
        gamma_denoising: float = 0.99,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.01,
        clip_ploss_coef_rate: float = 3.0,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        vf_coef: float = 0.5,
        target_kl: Optional[float] = 1.0,
        reward_horizon: Optional[int] = None,
    ) -> None:
        if not (1 <= act_steps <= horizon_steps):
            raise ValueError(f"act_steps must be in [1, horizon_steps], got {act_steps}.")
        if not (1 <= ft_denoising_steps <= denoising_steps):
            raise ValueError(
                f"ft_denoising_steps must be in [1, denoising_steps], got {ft_denoising_steps}."
            )
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.denoising_steps = denoising_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.actor_mlp_dims = list(actor_mlp_dims) if actor_mlp_dims is not None else [512, 512, 512]
        self.actor_activation_fn = actor_activation_fn
        self.actor_residual_style = actor_residual_style
        self.critic_mlp_dims = list(critic_mlp_dims) if critic_mlp_dims is not None else [256, 256, 256]
        self.critic_activation_fn = critic_activation_fn
        self.critic_residual_style = critic_residual_style
        self.time_dim = time_dim
        self.kernel_init = kernel_init
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.critic_warmup_updates = critic_warmup_updates
        self.update_epochs = update_epochs
        self.update_batch_size = update_batch_size
        self.norm_adv = norm_adv
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.vf_coef = vf_coef
        self.target_kl = target_kl
        self.reward_horizon = reward_horizon if reward_horizon is not None else act_steps

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer", "critic_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_steps": self.horizon_steps,
            "act_steps": self.act_steps,
            "denoising_steps": self.denoising_steps,
            "ft_denoising_steps": self.ft_denoising_steps,
            "actor_mlp_dims": self.actor_mlp_dims,
            "critic_mlp_dims": self.critic_mlp_dims,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "lr_scheduler_states": [
                sched.state_dict() if sched is not None else None
                for sched in self._lr_schedulers
            ]
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        for sched, sched_state in zip(
            self._lr_schedulers, state.get("lr_scheduler_states", [])
        ):
            if sched is not None and sched_state is not None:
                sched.load_state_dict(sched_state)


class DPPO(DPPOCore, OnPolicyAlgorithm):
    _compatible_checkpoint_algorithms = ("DPPO",)

    def __init__(
        self,
        env: Any,
        bc_checkpoint: Optional[str] = None,
        eval_env: Optional[Any] = None,
        num_steps: int = 50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon_steps: int = 4,
        act_steps: int = 4,
        denoising_steps: int = 20,
        ft_denoising_steps: int = 10,
        actor_mlp_dims: Optional[Sequence[int]] = None,
        actor_activation_fn: Optional[Activation] = "relu",
        actor_residual_style: bool = True,
        critic_mlp_dims: Optional[Sequence[int]] = None,
        critic_activation_fn: Optional[Activation] = "mish",
        critic_residual_style: bool = True,
        time_dim: int = 16,
        kernel_init: Optional[KernelInit] = None,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        critic_warmup_updates: int = 0,
        update_epochs: int = 5,
        update_batch_size: int = 50_000,
        norm_adv: bool = True,
        gamma_denoising: float = 0.99,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.01,
        clip_ploss_coef_rate: float = 3.0,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        vf_coef: float = 0.5,
        target_kl: Optional[float] = 1.0,
        reward_horizon: Optional[int] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        obs_space = env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"DPPO is state-only (Box observations); got {type(obs_space)}."
            )
        super().__init__(
            env=env,
            eval_env=eval_env,
            num_steps=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            finite_horizon_gae=finite_horizon_gae,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._init_dppo_params(
            horizon_steps=horizon_steps,
            act_steps=act_steps,
            denoising_steps=denoising_steps,
            ft_denoising_steps=ft_denoising_steps,
            actor_mlp_dims=actor_mlp_dims,
            actor_activation_fn=actor_activation_fn,
            actor_residual_style=actor_residual_style,
            critic_mlp_dims=critic_mlp_dims,
            critic_activation_fn=critic_activation_fn,
            critic_residual_style=critic_residual_style,
            time_dim=time_dim,
            kernel_init=kernel_init,
            denoised_clip_value=denoised_clip_value,
            randn_clip_value=randn_clip_value,
            final_action_clip_value=final_action_clip_value,
            min_sampling_denoising_std=min_sampling_denoising_std,
            min_logprob_denoising_std=min_logprob_denoising_std,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            critic_warmup_updates=critic_warmup_updates,
            update_epochs=update_epochs,
            update_batch_size=update_batch_size,
            norm_adv=norm_adv,
            gamma_denoising=gamma_denoising,
            clip_ploss_coef=clip_ploss_coef,
            clip_ploss_coef_base=clip_ploss_coef_base,
            clip_ploss_coef_rate=clip_ploss_coef_rate,
            clip_vloss_coef=clip_vloss_coef,
            clip_advantage_lower_quantile=clip_advantage_lower_quantile,
            clip_advantage_upper_quantile=clip_advantage_upper_quantile,
            vf_coef=vf_coef,
            target_kl=target_kl,
            reward_horizon=reward_horizon,
        )

        expected_action_shape = (self.act_steps,) + env.single_action_space.shape[1:]
        if env.single_action_space.shape != expected_action_shape:
            raise ValueError(
                "DPPO expects env.single_action_space to already be chunked by "
                "ActionChunkWrapper(env, act_steps=...); got shape "
                f"{env.single_action_space.shape}, expected {expected_action_shape}."
            )

        self._setup_model()
        if bc_checkpoint is not None:
            checkpoint = load_checkpoint_file(bc_checkpoint, map_location=self.device)
            ema_net_state_dict = checkpoint["state"]["extra"]["ema_net_state_dict"]
            self.policy.load_actor_weights(ema_net_state_dict)

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        raw_action_space = spaces.Box(
            low=self.env.single_action_space.low[0],
            high=self.env.single_action_space.high[0],
            shape=self.env.single_action_space.shape[1:],
            dtype=self.env.single_action_space.dtype,
        )
        self.policy = DPPOPolicy(
            observation_space=obs_space,
            action_space=raw_action_space,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            denoising_steps=self.denoising_steps,
            ft_denoising_steps=self.ft_denoising_steps,
            actor_mlp_dims=self.actor_mlp_dims,
            actor_activation_fn=self.actor_activation_fn,
            actor_residual_style=self.actor_residual_style,
            critic_mlp_dims=self.critic_mlp_dims,
            critic_activation_fn=self.critic_activation_fn,
            critic_residual_style=self.critic_residual_style,
            time_dim=self.time_dim,
            kernel_init=self.kernel_init,
            denoised_clip_value=self.denoised_clip_value,
            randn_clip_value=self.randn_clip_value,
            final_action_clip_value=self.final_action_clip_value,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
            min_logprob_denoising_std=self.min_logprob_denoising_std,
        ).to(self.device)

        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_ft.parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )
        self.critic_optimizer = make_optimizer(
            list(self.policy.critic.parameters()),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )
        self._lr_schedulers = [
            make_lr_scheduler(
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.actor_optimizer, self.critic_optimizer)
        ]

        self.rollout_buffer = RolloutBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        self._chain_buffer = DiffusionChainBuffer(
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            ft_denoising_steps=self.ft_denoising_steps,
            horizon_steps=self.horizon_steps,
            action_dim=self.policy.action_dim,
            device=self.device,
        )

    # --- rollout ---

    def _snapshot_window_initial_hidden(self, hidden, next_done: torch.Tensor):
        self._chain_buffer.reset()
        return hidden

    def _rollout_step(self, obs, hidden, episode_starts: torch.Tensor):
        del episode_starts
        with torch.no_grad():
            obs_t = self._obs_to_policy_device(obs)
            chain, action_chunk = self.policy.sample_rollout_chain(obs_t)
            cond = self.policy._cond(obs_t)
            old_log_probs = self.policy.get_logprobs(cond, chain)
            values = self.policy.predict_values(obs_t)
        self._chain_buffer.add(chain, old_log_probs)
        actions = action_chunk[:, : self.act_steps]
        log_probs = old_log_probs.mean(dim=(1, 2, 3))
        entropy = torch.zeros(self.num_envs, device=self.device)
        return actions, values, log_probs, entropy, hidden

    def _predict_last_values(self, obs, hidden) -> torch.Tensor:
        return self.policy.predict_values(obs)

    # --- update ---

    def train(self) -> dict[str, float]:
        self.policy.train()
        current_itr = self._global_update
        self._global_update += 1

        num_steps, num_envs = self.num_steps, self.num_envs
        k = self.ft_denoising_steps
        total = num_steps * num_envs

        obs_flat = flatten_leading_dims(self.rollout_buffer.obs)
        returns_flat = self.rollout_buffer.returns.reshape(total)
        values_flat = self.rollout_buffer.values.reshape(total)
        advantages_flat = self.rollout_buffer.advantages.reshape(total)
        chains_flat = self._chain_buffer.chains.reshape(
            total, k + 1, self.horizon_steps, self.policy.action_dim
        )
        old_logprobs_flat = self._chain_buffer.old_log_probs.reshape(
            total, k, self.horizon_steps, self.policy.action_dim
        )

        total_pairs = total * k
        metrics_sum: dict[str, float] = {}
        num_updates = 0
        train_actor = current_itr >= self.critic_warmup_updates

        for _ in range(self.update_epochs):
            perm = torch.randperm(total_pairs, device=self.device)
            num_batches = max(1, total_pairs // self.update_batch_size)
            stop = False
            for b in range(num_batches):
                idx = perm[b * self.update_batch_size : (b + 1) * self.update_batch_size]
                batch_inds, denoising_inds = torch.unravel_index(idx, (total, k))

                obs_b = index_obs(obs_flat, batch_inds)
                cond_b = self.policy._cond(obs_b)
                chains_prev_b = chains_flat[batch_inds, denoising_inds]
                chains_next_b = chains_flat[batch_inds, denoising_inds + 1]
                returns_b = returns_flat[batch_inds]
                values_b = values_flat[batch_inds]
                advantages_b = advantages_flat[batch_inds]
                logprobs_b = old_logprobs_flat[batch_inds, denoising_inds]

                loss, info = self._dppo_loss(
                    cond_b,
                    chains_prev_b,
                    chains_next_b,
                    denoising_inds,
                    returns_b,
                    values_b,
                    advantages_b,
                    logprobs_b,
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if train_actor:
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.actor_ft.parameters(), self.grad_clip_norm
                        )
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                for key, value in info.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                num_updates += 1

                if self.target_kl is not None and info["approx_kl"] > self.target_kl:
                    stop = True
                    break
            if stop:
                break

        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

        metrics = {key: value / num_updates for key, value in metrics_sum.items()}
        y_pred, y_true = values_flat, returns_flat
        var_y = torch.var(y_true)
        explained_var = (
            float("nan") if var_y == 0 else float((1 - torch.var(y_true - y_pred) / var_y).item())
        )
        metrics["explained_variance"] = explained_var
        return metrics

    def _dppo_loss(
        self,
        cond_b: dict,
        chains_prev_b: torch.Tensor,
        chains_next_b: torch.Tensor,
        denoising_inds_b: torch.Tensor,
        returns_b: torch.Tensor,
        values_b: torch.Tensor,
        advantages_b: torch.Tensor,
        logprobs_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        newlogprobs = self.policy.get_logprobs_subsample(
            cond_b, chains_prev_b, chains_next_b, denoising_inds_b
        )
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = logprobs_b.clamp(min=-5, max=2)

        newlogprobs = newlogprobs[:, : self.reward_horizon, :].mean(dim=(-1, -2))
        oldlogprobs = oldlogprobs[:, : self.reward_horizon, :].mean(dim=(-1, -2))

        if self.norm_adv:
            advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)
        adv_min = torch.quantile(advantages_b, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages_b, self.clip_advantage_upper_quantile)
        advantages_b = advantages_b.clamp(min=adv_min, max=adv_max)

        discount = self.gamma_denoising ** (self.ft_denoising_steps - denoising_inds_b.float() - 1)
        advantages_b = advantages_b * discount

        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        if self.ft_denoising_steps > 1:
            t = denoising_inds_b.float() / (self.ft_denoising_steps - 1)
            clip_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_coef = denoising_inds_b.float()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        pg_loss1 = -advantages_b * ratio
        pg_loss2 = -advantages_b * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalues = self.policy.critic(cond_b["state"]).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns_b) ** 2
            v_clipped = values_b + torch.clamp(
                newvalues - values_b, -self.clip_vloss_coef, self.clip_vloss_coef
            )
            v_loss_clipped = (v_clipped - returns_b) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalues - returns_b) ** 2).mean()

        loss = pg_loss + self.vf_coef * v_loss
        info = {
            "loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "clipfrac": clipfrac,
            "approx_kl": float(approx_kl.item()),
            "ratio": float(ratio.mean().item()),
        }
        return loss, info
