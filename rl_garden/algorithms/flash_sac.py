"""FlashSAC: Fast and Stable Off-Policy RL.

Port of Holiday-Robot/FlashSAC into the rl-garden OffPolicyAlgorithm hierarchy.
Key properties carried over from the original:
  - Categorical distributional critic (C51-style, 101 bins, CE loss)
  - Unit-norm weight constraint on every linear layer (called after each opt step)
  - Zeta noise repetition for action exploration
  - Optional reward normalization (running discounted-return variance)
  - Optional AMP (torch.autocast float16) and torch.compile
  - Asymmetric actor/critic observations

Reference: arxiv.org/abs/2409.08689
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.buffers.nstep_tensor_buffer import NStepTensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.reward_normalizer import RewardNormalizer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.policies.flash_sac_policy import FlashSACPolicy, FlashSACTemperature


# ---------------------------------------------------------------------------
# Compiled helpers (ported verbatim from FlashSAC/update.py)
# ---------------------------------------------------------------------------

@torch.compile
def _select_min_q_log_probs(
    next_qs: torch.Tensor,       # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Select log-probs from the critic with the lower Q-value expectation."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]  # (B, num_bins)
    return selected


@torch.compile
def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """Truncated Zeta(mu) CDF for noise repeat-count sampling."""
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / pmf.sum()
    return torch.cumsum(pmf, dim=0)


@torch.compile
def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """Sample a scalar integer in [1, len(cdf)] from the given CDF."""
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,            # (B,)
    discounts: torch.Tensor,         # (B,)  — gamma^n * (1-terminal)
    actor_entropy: torch.Tensor,     # (B,)
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    """Categorical Bellman projection using pre-computed n-step discounts.

    Uses ``discounts`` directly (encodes both gamma^n and terminal mask) instead
    of separate ``gamma`` and ``done`` arguments, so it works correctly with the
    n-step buffer that may stop early at non-terminal episode boundaries.
    """
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    discounts = discounts.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    target_bin_values = reward + discounts * (bin_values - actor_entropy)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    frac = b - lower.float()

    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    target_probs = torch.zeros(
        batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device
    )
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class FlashSAC(SACCore, OffPolicyAlgorithm):
    """FlashSAC algorithm for Box observations."""

    _compatible_checkpoint_algorithms = ("FlashSAC",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        # replay / training
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 1024,
        gamma: float = 0.99,
        tau: float = 0.005,
        training_freq: int = 512,
        utd: float = 1.0,
        n_step: int = 3,
        bootstrap_at_done: str = "truncated",
        # architecture
        actor_hidden_dim: int = 128,
        actor_num_blocks: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_blocks: int = 2,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        asymmetric_obs_dim: int = 0,
        # optimizers
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        alpha_lr: float = 1e-4,
        actor_update_period: int = 1,
        grad_clip_norm: Optional[float] = None,
        # entropy / temperature
        temp_initial_value: float = 0.01,
        target_entropy: float | str = "auto",
        # exploration (Zeta noise)
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        # reward normalization
        normalize_reward: bool = False,
        normalized_g_max: float = 10.0,
        # BC regularization
        bc_alpha: float = 0.0,
        # performance
        use_compile: bool = False,
        compile_mode: str = "default",
        use_amp: bool = False,
        # standard OffPolicyAlgorithm params
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
        initial_training_phase: Optional[InitialTrainingPhase] = None,
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
            initial_training_phase=initial_training_phase,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"FlashSAC requires a flat Box observation space, got {type(obs_space)}"
            )

        self.n_step = n_step
        self.actor_hidden_dim = actor_hidden_dim
        self.actor_num_blocks = actor_num_blocks
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_num_blocks = critic_num_blocks
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v
        self.asymmetric_obs_dim = asymmetric_obs_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.actor_update_period = actor_update_period
        self.grad_clip_norm = grad_clip_norm
        self.temp_initial_value = temp_initial_value
        self.target_entropy_arg = target_entropy
        self.actor_noise_zeta_mu = actor_noise_zeta_mu
        self.actor_noise_zeta_max = actor_noise_zeta_max
        self.normalize_reward = normalize_reward
        self.normalized_g_max = normalized_g_max
        self.bc_alpha = bc_alpha
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.use_amp = use_amp

        self._setup_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        self.policy = FlashSACPolicy(
            observation_space=obs_space,
            action_space=act_space,
            actor_hidden_dim=self.actor_hidden_dim,
            actor_num_blocks=self.actor_num_blocks,
            critic_hidden_dim=self.critic_hidden_dim,
            critic_num_blocks=self.critic_num_blocks,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
            asymmetric_obs_dim=self.asymmetric_obs_dim,
        ).to(self.device)

        # Normalize parameters after initialization
        self.policy.normalize_weights()

        use_fused = self.device.type == "cuda" and torch.cuda.is_available()

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.actor_lr, fused=use_fused
        )
        self.q_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.critic_lr, fused=use_fused
        )

        # Temperature
        self.temperature_module = FlashSACTemperature(self.temp_initial_value).to(self.device)
        self.alpha_optimizer = torch.optim.Adam(
            self.temperature_module.parameters(), lr=self.alpha_lr, fused=use_fused
        )

        if self.target_entropy_arg == "auto":
            self.target_entropy = float(-np.prod(act_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy_arg)

        # Replay buffer
        self.replay_buffer = NStepTensorReplayBuffer(
            observation_space=obs_space,
            action_space=act_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            nstep=self.n_step,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

        # Zeta noise state — shape (num_envs, action_dim)
        action_dim = int(act_space.shape[0])
        self._zeta_noise = torch.zeros(
            self.num_envs, action_dim, device=self.device
        )
        self._zeta_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._zeta_n = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self._zeta_cdf = _build_truncated_zeta_cdf(
            self.actor_noise_zeta_mu, self.actor_noise_zeta_max
        ).to(self.device)

        # Optional reward normalizer
        self.reward_normalizer: Optional[RewardNormalizer] = None
        if self.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=self.gamma,
                G_max=self.normalized_g_max,
                device=self.device,
            )

        # Optional compile
        if self.use_compile:
            self.policy.actor = torch.compile(self.policy.actor, mode=self.compile_mode)  # type: ignore[assignment]
            self.policy.critic = torch.compile(self.policy.critic, mode=self.compile_mode)  # type: ignore[assignment]
            self.policy.critic_target = torch.compile(self.policy.critic_target, mode=self.compile_mode)  # type: ignore[assignment]

        # AMP grad scaler (CUDA only)
        self._grad_scaler: Optional[torch.amp.GradScaler] = None
        if self.use_amp and self.device.type == "cuda":
            self._grad_scaler = torch.amp.GradScaler()

    # ------------------------------------------------------------------
    # Replay buffer — extra kwargs for episode_end field
    # ------------------------------------------------------------------

    def _replay_buffer_step_kwargs(
        self,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
    ) -> dict[str, Any]:
        return {"episode_end": (terminations | truncations)}

    # ------------------------------------------------------------------
    # Reward normalizer — updated online each env step
    # ------------------------------------------------------------------

    def _on_env_step(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
    ) -> None:
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(rewards, terminations, truncations)

    # ------------------------------------------------------------------
    # Rollout action — Zeta noise repeat
    # ------------------------------------------------------------------

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        phase = self._active_initial_training_phase()
        if phase is not None:
            actions = self._policy_action(obs)
            if phase.random_action_prob > 0.0:
                random_actions = self._explore_action(obs)
                mask_shape = (actions.shape[0],) + (1,) * (actions.ndim - 1)
                random_mask = (
                    torch.rand(mask_shape, device=actions.device) < phase.random_action_prob
                )
                actions = torch.where(random_mask, random_actions, actions)
            return actions, actions, None

        if not learning_has_started:
            actions = self._explore_action(obs)
            return actions, actions, None

        obs_dev = self._obs_to_policy_device(obs)
        with torch.no_grad():
            actor_obs = self.policy.actor_obs(obs_dev)
            mean, std = self.policy.actor.get_mean_and_std(actor_obs, training=False)

        # Zeta noise repeat: reinit when count exhausted or at start
        reinit = (self._zeta_count == 0) | (self._zeta_count >= self._zeta_n)
        new_noise = torch.randn_like(mean)
        new_n = _sample_integer_from_cdf(self._zeta_cdf)

        self._zeta_noise = torch.where(reinit.unsqueeze(-1), new_noise, self._zeta_noise)
        self._zeta_n = torch.where(reinit, new_n, self._zeta_n)
        self._zeta_count = torch.where(reinit, torch.zeros_like(self._zeta_count), self._zeta_count)

        actions = torch.tanh(mean + std * self._zeta_noise)
        self._zeta_count += 1

        return actions, actions, None

    # ------------------------------------------------------------------
    # SACCore compatibility: _current_alpha for Q-MC diagnostics
    # ------------------------------------------------------------------

    def _current_alpha(self) -> torch.Tensor:
        return self.temperature_module()

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        update_mask = self._training_update_mask()
        if not update_mask.update_actor and not update_mask.update_critic:
            return {}

        critic_losses_t: list[torch.Tensor] = []
        actor_losses_t: list[torch.Tensor] = []
        alphas_t: list[torch.Tensor] = []

        amp_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        )

        for _ in range(gradient_steps):
            self._global_update += 1
            data = self.replay_buffer.sample(self.batch_size)

            rewards = data.rewards
            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer.normalize_rewards(rewards)

            # ----------------------------------------------------------------
            # Critic update
            # ----------------------------------------------------------------
            if update_mask.update_critic:
                with torch.no_grad():
                    with amp_ctx:
                        actor_next_obs = self.policy.actor_obs(data.next_obs)
                        next_actions, next_info = self.policy.actor(actor_next_obs, training=False)
                        next_log_probs = next_info["log_prob"]

                        temp = self.temperature_module()
                        next_actor_entropy = temp * next_log_probs  # (B,)

                        # Cross-batch concat for BN statistics
                        obs_all = torch.cat([data.obs, data.next_obs], dim=0)          # (2B, obs)
                        act_all = torch.cat([data.actions, next_actions], dim=0)        # (2B, act)

                        qs_all, q_infos_all = self.policy.critic_target(
                            observations=obs_all, actions=act_all, training=True
                        )
                        next_qs = qs_all.chunk(2, dim=1)[1]           # (2, B)
                        next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]  # (2, B, bins)
                        next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)

                        target_probs = _compute_categorical_td_target(
                            target_log_probs=next_q_log_probs,
                            reward=rewards,
                            discounts=data.discounts,
                            actor_entropy=next_actor_entropy,
                            num_bins=self.num_bins,
                            min_v=self.min_v,
                            max_v=self.max_v,
                        )

                with amp_ctx:
                    pred_qs_all, pred_q_infos = self.policy.critic(
                        observations=obs_all, actions=act_all, training=True
                    )
                    pred_log_probs = pred_q_infos["log_prob"].chunk(2, dim=1)[0]  # (2, B, bins)
                    ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
                    critic_loss = ce_loss.mean()

                self.q_optimizer.zero_grad(set_to_none=True)
                if self.use_amp and self._grad_scaler is not None:
                    self._grad_scaler.scale(critic_loss).backward()
                    if self.grad_clip_norm is not None:
                        self._grad_scaler.unscale_(self.q_optimizer)
                        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_clip_norm)
                    self._grad_scaler.step(self.q_optimizer)
                    self._grad_scaler.update()
                else:
                    critic_loss.backward()
                    if self.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_clip_norm)
                    self.q_optimizer.step()

                self.policy.normalize_weights()

                if compute_info:
                    critic_losses_t.append(critic_loss.detach())

            # ----------------------------------------------------------------
            # Actor + temperature update
            # ----------------------------------------------------------------
            do_actor = (self._global_update % self.actor_update_period == 0)
            if do_actor and update_mask.update_actor:
                with amp_ctx:
                    actor_obs_cur = self.policy.actor_obs(data.obs)
                    actor_obs_next = self.policy.actor_obs(data.next_obs)
                    actor_obs_all = torch.cat([actor_obs_cur, actor_obs_next], dim=0)
                    actions_all, info_all = self.policy.actor(actor_obs_all, training=True)
                    log_probs_all = info_all["log_prob"]

                    actions_cur = actions_all.chunk(2, dim=0)[0]
                    log_probs_cur = log_probs_all.chunk(2, dim=0)[0]

                    # disable critic gradients (prevent CUDA graph corruption)
                    self.policy.critic.requires_grad_(False)
                    qs_cur, _ = self.policy.critic(
                        observations=data.obs, actions=actions_cur, training=False
                    )
                    min_q = torch.minimum(qs_cur[0], qs_cur[1])
                    self.policy.critic.requires_grad_(True)

                    temp = self.temperature_module()
                    actor_loss = (log_probs_cur * temp.detach() - min_q).mean()

                    if self.bc_alpha > 0:
                        q_abs = min_q.abs().mean().detach()
                        bc_loss = ((actions_cur - data.actions) ** 2).mean()
                        actor_loss = actor_loss + self.bc_alpha * q_abs * bc_loss

                    entropy = -log_probs_cur.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                if self.use_amp and self._grad_scaler is not None:
                    self._grad_scaler.scale(actor_loss).backward()
                    if self.grad_clip_norm is not None:
                        self._grad_scaler.unscale_(self.actor_optimizer)
                        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_clip_norm)
                    self._grad_scaler.step(self.actor_optimizer)
                    self._grad_scaler.update()
                else:
                    actor_loss.backward()
                    if self.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_clip_norm)
                    self.actor_optimizer.step()

                self.policy.normalize_weights()

                # Temperature update — matches FlashSAC original (no negation)
                alpha_loss = temp * (entropy.detach() - self.target_entropy)
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()

                if compute_info:
                    actor_losses_t.append(actor_loss.detach())
                    alphas_t.append(temp.detach())

            # Target network update every step
            self._target_update()

        if not compute_info:
            return {}

        out: dict[str, float] = {}
        if critic_losses_t:
            out["critic/loss"] = float(torch.stack(critic_losses_t).mean().item())
        if actor_losses_t:
            out["actor/loss"] = float(torch.stack(actor_losses_t).mean().item())
        if alphas_t:
            out["temperature/value"] = float(torch.stack(alphas_t).mean().item())
        return out

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "n_step": self.n_step,
            "actor_hidden_dim": self.actor_hidden_dim,
            "actor_num_blocks": self.actor_num_blocks,
            "critic_hidden_dim": self.critic_hidden_dim,
            "critic_num_blocks": self.critic_num_blocks,
            "num_bins": self.num_bins,
            "min_v": self.min_v,
            "max_v": self.max_v,
            "asymmetric_obs_dim": self.asymmetric_obs_dim,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "alpha_lr": self.alpha_lr,
            "temp_initial_value": self.temp_initial_value,
            "target_entropy": self.target_entropy,
            "normalize_reward": self.normalize_reward,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "target_entropy": self.target_entropy,
            "temperature": self.temperature_module.state_dict(),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])
        if "temperature" in state:
            self.temperature_module.load_state_dict(state["temperature"])

    def _optimizer_state_dicts(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def _load_optimizer_state_dicts(self, states: dict[str, Any]) -> None:
        if "actor_optimizer" in states:
            self.actor_optimizer.load_state_dict(states["actor_optimizer"])
        if "q_optimizer" in states:
            self.q_optimizer.load_state_dict(states["q_optimizer"])
        if "alpha_optimizer" in states:
            self.alpha_optimizer.load_state_dict(states["alpha_optimizer"])
