"""``TDMPC2``: implicit world model + CEM/MPPI planner, ported from
``3rd_party/tdmpc2/tdmpc2/{tdmpc2.py,trainer/online_trainer.py}``.

Scope for this port (see ``docs/superpowers/specs`` design discussion this
was brainstormed from):

- **Single-task only.** No task embedding / action masking / per-task
  discount -- nothing else in rl-garden needs multitask training yet.
- **``num_envs == 1`` / ``num_eval_envs == 1`` required.** The CEM planner
  already rolls out ``num_samples`` (hundreds) of trajectories per env step;
  multiplying that by vectorized envs is an untested configuration upstream
  never validated. Vectorized rollout is a deliberate future extension, not
  attempted here.
- **``episodic=False`` only.** ``EpisodeSliceBuffer`` rejects any sampled
  window reaching a true terminal transition (see its module docstring), so
  a termination classifier trained on those windows would never see a
  positive example. Upstream's own default is ``episodic: false`` (episodes
  end by truncation only), which this restriction doesn't affect at all.
- **Joint world-model optimizer kept, as upstream does** (encoder + latent
  projection + dynamics + reward + termination + Q share one Adam; ``pi`` has
  its own): every other rl-garden algorithm keeps separate optimizers per
  network, but TD-MPC2's consistency loss backprops through all of those
  heads via one shared latent rollout in a single ``backward()`` call --
  splitting that into per-network optimizers would need multiple backward
  passes or manual gradient accumulation, diverging from upstream and adding
  bug surface for no behavioral benefit.

This class inherits ``BaseAlgorithm`` directly (not ``OffPolicyAlgorithm``):
its rollout is single-env, one-episode-at-a-time, with exactly one gradient
step per env step past ``learning_starts`` (plus a "pretrain on seed data"
burst exactly at the ``learning_starts`` boundary, matching upstream) --
structurally nothing like ``OffPolicyAlgorithm``'s vectorized, multi-step
rollout-then-batch-update loop.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.algorithms.tdmpc2 import math_utils
from rl_garden.algorithms.tdmpc2 import planner as planner_mod
from rl_garden.algorithms.tdmpc2.planner import PlannerConfig
from rl_garden.algorithms.tdmpc2.policy import TDMPC2Policy
from rl_garden.algorithms.tdmpc2.world_model import WorldModel
from rl_garden.buffers.episode_slice_buffer import EpisodeSliceBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import CombinedExtractor, ImageEncoderFactory, default_image_encoder_factory
from rl_garden.encoders.flatten import FlattenExtractor


def _compute_discount(
    episode_length: int, discount_denom: float, discount_min: float, discount_max: float
) -> float:
    frac = episode_length / discount_denom
    return min(max((frac - 1) / frac, discount_min), discount_max)


class TDMPC2(BaseAlgorithm):
    _compatible_checkpoint_algorithms = ("TDMPC2",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        episode_length: int = 100,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        seed_steps: Optional[int] = None,
        # planning
        use_planner: bool = True,
        horizon: int = 3,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        iterations: int = 6,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        # architecture
        latent_dim: int = 512,
        mlp_dim: int = 512,
        simnorm_dim: int = 8,
        num_q: int = 5,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        dropout: float = 0.01,
        episodic: bool = False,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        entropy_coef: float = 1e-4,
        # optimization
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        grad_clip_norm: float = 20.0,
        tau: float = 0.01,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        termination_coef: float = 1.0,
        discount_denom: float = 5.0,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        # Dict-obs (pixel) encoder kwargs -- same shape as SAC's, ignored for
        # Box observations.
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        image_augmentation: Optional[str] = None,
        random_shift_pad: Optional[int] = None,
        image_augmentation_seed: Optional[int] = None,
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
        super().__init__(env=env, eval_env=eval_env, seed=seed, device=device, logger=logger)

        if self.env.num_envs != 1:
            raise ValueError(
                f"TDMPC2 requires env.num_envs == 1 (vectorized rollout is not "
                f"supported in this port), got {self.env.num_envs}."
            )
        if self.eval_env is not None and self.eval_env.num_envs != 1:
            raise ValueError(
                f"TDMPC2 requires eval_env.num_envs == 1, got {self.eval_env.num_envs}."
            )
        if episodic:
            raise NotImplementedError(
                "episodic=True is not supported: EpisodeSliceBuffer rejects any "
                "sampled window that reaches a true terminal transition (see its "
                "module docstring), so the termination classifier would never "
                "see a positive training example. Use the default episodic=False."
            )

        self.episode_length = episode_length
        self.buffer_size = buffer_size
        self.buffer_device = buffer_device
        self.batch_size = batch_size
        self.seed_steps = seed_steps

        self.use_planner = use_planner
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_pi_trajs = num_pi_trajs
        self.iterations = iterations
        self.min_std = min_std
        self.max_std = max_std
        self.temperature = temperature

        self.latent_dim = latent_dim
        self.mlp_dim = mlp_dim
        self.simnorm_dim = simnorm_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.dropout = dropout
        self.episodic = episodic
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.entropy_coef = entropy_coef

        self.lr = lr
        self.enc_lr_scale = enc_lr_scale
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.rho = rho
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.termination_coef = termination_coef
        self.discount_denom = discount_denom
        self.discount_min = discount_min
        self.discount_max = discount_max

        obs_space = self.env.single_observation_space
        self._is_dict_obs = isinstance(obs_space, spaces.Dict)
        if not self._is_dict_obs and not isinstance(obs_space, spaces.Box):
            raise TypeError(f"TDMPC2 supports Box or Dict observation spaces, got {type(obs_space)}")
        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_keys = image_keys if image_keys is not None else ("rgb", "depth")
        self._state_key = state_key if state_key is not None else "state"
        self._use_proprio = use_proprio if use_proprio is not None else True
        self._proprio_latent_dim = proprio_latent_dim if proprio_latent_dim is not None else 64
        self._image_fusion_mode = image_fusion_mode if image_fusion_mode is not None else "stack_channels"
        self._enable_stacking = enable_stacking if enable_stacking is not None else False
        self._image_augmentation = image_augmentation if image_augmentation is not None else "none"
        self._random_shift_pad = random_shift_pad if random_shift_pad is not None else 4
        self._image_augmentation_seed = image_augmentation_seed

        self.std_log = std_log
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.num_eval_steps = num_eval_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.save_replay_buffer = save_replay_buffer
        self.save_final_checkpoint = save_final_checkpoint
        self._last_checkpoint_step = -1

        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        return CombinedExtractor if self._is_dict_obs else FlattenExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        if not self._is_dict_obs:
            return {}
        return {
            "image_keys": self._image_keys,
            "state_key": self._state_key,
            "image_encoder_factory": self._image_encoder_factory,
            "proprio_latent_dim": self._proprio_latent_dim,
            "use_proprio": self._use_proprio,
            "fusion_mode": self._image_fusion_mode,
            "enable_stacking": self._enable_stacking,
            "image_augmentation": self._image_augmentation,
            "random_shift_pad": self._random_shift_pad,
            "augmentation_seed": self._image_augmentation_seed,
        }

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        cls = self._default_features_extractor_class()
        return cls(observation_space=self.env.single_observation_space, **self._default_features_extractor_kwargs())

    def _setup_model(self) -> None:
        action_dim = int(np.prod(self.env.single_action_space.shape))
        features_extractor = self._build_features_extractor()

        world_model = WorldModel(
            encoder=features_extractor,
            action_dim=action_dim,
            latent_dim=self.latent_dim,
            mlp_dim=self.mlp_dim,
            simnorm_dim=self.simnorm_dim,
            num_q=self.num_q,
            num_bins=self.num_bins,
            vmin=self.vmin,
            vmax=self.vmax,
            dropout=self.dropout,
            episodic=self.episodic,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            tau=self.tau,
        ).to(self.device)

        self.planner_cfg = PlannerConfig(
            action_dim=action_dim,
            horizon=self.horizon,
            num_samples=self.num_samples,
            num_elites=self.num_elites,
            num_pi_trajs=self.num_pi_trajs,
            iterations=self.iterations,
            min_std=self.min_std,
            max_std=self.max_std,
            temperature=self.temperature,
        )
        self.discount = _compute_discount(
            self.episode_length, self.discount_denom, self.discount_min, self.discount_max
        )
        self.learning_starts = (
            self.seed_steps if self.seed_steps is not None else max(1_000, 5 * self.episode_length)
        )

        self.policy = TDMPC2Policy(
            world_model, self.planner_cfg, self.discount, use_planner=self.use_planner
        ).to(self.device)

        enc_params = list(world_model.encoder.parameters()) + list(world_model._latent_proj.parameters())
        other_params = (
            list(world_model._dynamics.parameters())
            + list(world_model._reward.parameters())
            + list(world_model._Q.parameters())
        )
        if world_model._termination is not None:
            other_params += list(world_model._termination.parameters())
        self.world_optimizer = torch.optim.Adam(
            [
                {"params": enc_params, "lr": self.lr * self.enc_lr_scale},
                {"params": other_params, "lr": self.lr},
            ]
        )
        self.pi_optimizer = torch.optim.Adam(world_model._pi.parameters(), lr=self.lr, eps=1e-5)

        self.replay_buffer = EpisodeSliceBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=self.buffer_size,
            horizon=self.horizon,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _random_action(self) -> torch.Tensor:
        shape = self.env.action_space.shape
        return 2 * torch.rand(shape, dtype=torch.float32, device=self.device) - 1

    # ------------------------------------------------------------------
    # Gradient step
    # ------------------------------------------------------------------

    def _encode_window(self, obs_window: Obs, window_len: int, batch_size: int) -> torch.Tensor:
        flat = flatten_leading_dims(obs_window)
        z_flat = self.policy.world_model.encode(flat)
        return z_flat.reshape(window_len, batch_size, -1)

    def _td_target(
        self, next_z: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor
    ) -> torch.Tensor:
        world_model = self.policy.world_model
        action, _ = world_model.pi(next_z)
        return reward + self.discount * (1 - terminated) * world_model.Q(
            next_z, action, return_type="min", target=True
        )

    def _update_pi(self, zs: torch.Tensor) -> dict[str, float]:
        world_model = self.policy.world_model
        action, info = world_model.pi(zs)
        qs = world_model.Q(zs, action, return_type="avg", detach=True)
        world_model.scale.update(qs[0])
        qs = world_model.scale(qs)

        rho_pows = self.rho ** torch.arange(zs.shape[0], device=zs.device)
        pi_loss = (
            -(self.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho_pows
        ).mean()

        self.pi_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(world_model._pi.parameters(), self.grad_clip_norm)
        self.pi_optimizer.step()

        return {
            "pi_loss": float(pi_loss.detach()),
            "pi_grad_norm": float(pi_grad_norm),
            "pi_scale": float(world_model.scale.value.item()),
        }

    def _gradient_step(self) -> dict[str, float]:
        world_model = self.policy.world_model
        sample = self.replay_buffer.sample(self.batch_size)
        obs, action, reward, terminated = sample.obs, sample.action, sample.reward, sample.terminated
        horizon, batch_size = action.shape[0], action.shape[1]
        reward = reward.unsqueeze(-1)
        terminated_f = terminated.float().unsqueeze(-1)

        with torch.no_grad():
            next_obs = index_obs(obs, slice(1, None))
            next_z = self._encode_window(next_obs, horizon, batch_size)
            td_targets = self._td_target(next_z, reward, terminated_f)

        world_model.train()
        zs = torch.empty(horizon + 1, batch_size, self.latent_dim, device=self.device)
        obs0 = index_obs(obs, 0)
        z = world_model.encode(obs0)
        zs[0] = z
        consistency_loss = torch.zeros((), device=self.device)
        for t in range(horizon):
            z = world_model.next(z, action[t])
            consistency_loss = consistency_loss + F.mse_loss(z, next_z[t]) * self.rho**t
            zs[t + 1] = z

        _zs = zs[:-1]
        qs = world_model.Q(_zs, action, return_type="all")
        reward_preds = world_model.reward(_zs, action)
        termination_pred = world_model.termination(zs[1:], unnormalized=True) if self.episodic else None

        reward_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        for t in range(horizon):
            reward_loss = reward_loss + math_utils.soft_ce(
                reward_preds[t], reward[t], self.num_bins, self.vmin, self.vmax, world_model.bin_size
            ).mean() * self.rho**t
            for qi in range(self.num_q):
                value_loss = value_loss + math_utils.soft_ce(
                    qs[qi, t], td_targets[t], self.num_bins, self.vmin, self.vmax, world_model.bin_size
                ).mean() * self.rho**t

        consistency_loss = consistency_loss / horizon
        reward_loss = reward_loss / horizon
        value_loss = value_loss / (horizon * self.num_q)
        if self.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated_f)
        else:
            termination_loss = torch.zeros((), device=self.device)

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.termination_coef * termination_loss
            + self.value_coef * value_loss
        )

        self.world_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(), self.grad_clip_norm)
        self.world_optimizer.step()

        pi_info = self._update_pi(zs.detach())
        world_model.soft_update_target_Q()
        world_model.eval()

        self._global_update += 1
        info = {
            "consistency_loss": float(consistency_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "termination_loss": float(termination_loss.detach()) if self.episodic else 0.0,
            "total_loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm),
        }
        info.update(pi_info)
        return info

    # ------------------------------------------------------------------
    # Eval hooks (see BaseAlgorithm._evaluate() -- reused unmodified)
    # ------------------------------------------------------------------

    def _eval_start_hook(self) -> None:
        self.policy.reset_episode()

    def _eval_step_hook(
        self,
        obs_before,
        critic_action: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos: dict,
    ) -> None:
        del obs_before, critic_action, rewards, infos
        done = bool((terminations | truncations).any().item())
        self.policy.notify_step_done(done)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("world_optimizer", "pi_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "episode_length": self.episode_length,
            "buffer_size": self.buffer_size,
            "buffer_device": self.buffer_device,
            "batch_size": self.batch_size,
            "horizon": self.horizon,
            "num_samples": self.num_samples,
            "num_elites": self.num_elites,
            "num_pi_trajs": self.num_pi_trajs,
            "iterations": self.iterations,
            "latent_dim": self.latent_dim,
            "mlp_dim": self.mlp_dim,
            "num_q": self.num_q,
            "num_bins": self.num_bins,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "episodic": self.episodic,
            "discount_denom": self.discount_denom,
            "discount_min": self.discount_min,
            "discount_max": self.discount_max,
            "learning_starts": self.learning_starts,
        }

    def _checkpoint_path(self, name: str) -> Path:
        assert self.checkpoint_dir is not None
        return Path(self.checkpoint_dir) / name

    def _save_checkpoint(self, name: str) -> None:
        if self.checkpoint_dir is None:
            return
        self.save(self._checkpoint_path(name), include_replay_buffer=self.save_replay_buffer)

    def _maybe_save_periodic_checkpoint(self, previous_step: int) -> None:
        if self.checkpoint_dir is None or self.checkpoint_freq <= 0:
            return
        if self._global_step // self.checkpoint_freq <= previous_step // self.checkpoint_freq:
            return
        if self._global_step == self._last_checkpoint_step:
            return
        self._save_checkpoint(f"checkpoint_{self._global_step}.pt")
        self._last_checkpoint_step = self._global_step

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int) -> "TDMPC2":
        self._on_training_start(total_timesteps)
        obs, _ = self.env.reset(seed=self.seed)

        train_prev_mean: Optional[torch.Tensor] = None
        train_t0 = True
        cumulative: dict[str, float] = defaultdict(float)

        while self._global_step < total_timesteps:
            previous_step = self._global_step

            if self.eval_freq > 0 and self._global_step % self.eval_freq == 0:
                eval_metrics = self._evaluate()
                if self.logger is not None:
                    for key, value in eval_metrics.items():
                        self.logger.add_scalar(f"eval/{key}", value, self._global_step)
                if self.std_log:
                    print(
                        "[eval] "
                        f"step={self._global_step}/{total_timesteps} "
                        f"{ {k: round(v, 4) for k, v in eval_metrics.items()} }",
                        flush=True,
                    )

            obs_device = self._obs_to_policy_device(obs)
            if self._global_step < self.learning_starts:
                action = self._random_action()
            elif self.use_planner:
                action, train_prev_mean = planner_mod.plan(
                    self.policy.world_model,
                    obs_device,
                    train_prev_mean,
                    self.discount,
                    self.planner_cfg,
                    train_t0,
                    eval_mode=False,
                )
                action = action.unsqueeze(0)
            else:
                with torch.no_grad():
                    z = self.policy.world_model.encode(obs_device)
                    action, _ = self.policy.world_model.pi(z)
            train_t0 = False

            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            if bool(terminated.any().item()) and not self.episodic:
                raise ValueError(
                    "Termination detected but episodic=False -- construct TDMPC2 with "
                    "episodic=True to train on tasks with true terminal states (note: "
                    "not currently supported by EpisodeSliceBuffer, see its docstring)."
                )
            episode_end = terminated | truncated
            self.replay_buffer.add(obs, next_obs, action, reward, terminated, episode_end)

            done = bool(episode_end.any().item())
            if done:
                train_prev_mean = None
                train_t0 = True

            self._global_step += 1
            obs = next_obs

            if self._global_step >= self.learning_starts:
                num_updates = self.learning_starts if self._global_step == self.learning_starts else 1
                info: dict[str, float] = {}
                for _ in range(num_updates):
                    info = self._gradient_step()
                should_log = self.log_freq > 0 and self._global_step % self.log_freq == 0
                if should_log:
                    if self.logger is not None:
                        self.logger.log_metrics({f"train/{k}": v for k, v in info.items()}, self._global_step)
                    if self.std_log:
                        progress = 100.0 * self._global_step / total_timesteps
                        print(
                            "[train] "
                            f"step={self._global_step}/{total_timesteps} ({progress:.2f}%) "
                            f"total_loss={info.get('total_loss', float('nan')):.4f} "
                            f"pi_loss={info.get('pi_loss', float('nan')):.4f}",
                            flush=True,
                        )

            self._maybe_save_periodic_checkpoint(previous_step)

        if self.checkpoint_dir is not None and self.save_final_checkpoint:
            self._save_checkpoint("final.pt")

        return self
