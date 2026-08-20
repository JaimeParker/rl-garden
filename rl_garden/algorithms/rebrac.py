"""ReBRAC: TD3-BC with a second BC penalty applied to the critic's bootstrap.

Ported from ``3rd_party/CORL/algorithms/offline/rebrac.py``
(arXiv 2305.09836). A genuine literature-subtyping relationship to the
already-shipped ``TD3BC`` (``rl_garden/algorithms/td3_bc.py``) -- ReBRAC's
own paper frames itself as a refinement of the TD3+BC lineage -- so
``ReBRACCore(TD3BCCore)`` reuses that class's network construction
(``TD3BCPolicy``), ``_critic_loss`` static method, and
``policy_freq``-delayed actor/target-update structure directly.
``TD3BCCore.train()`` has no hook seams (monolithic, like
``FQLCore.train()``), so ``train()`` below is a full override, matching the
precedent ``ACFQLCore.train()`` already set for overriding a hookless
parent method.

Formulas verified against ``rebrac.py`` directly:

- **Critic** (``update_critic``, ``rebrac.py:477-523``): identical TD3-style
  target to ``TD3BCCore``'s (deterministic target actor + clipped policy
  noise, min over the critic ensemble), but with a BC penalty against the
  *dataset's actual next-step action* subtracted from the bootstrap before
  it's used: ``next_q = target_critic(next_obs,next_action).min(0) -
  critic_bc_coef * ((next_action - next_actions)**2).sum(-1)``. This needs
  ``next_actions`` -- the action the dataset's behavior policy actually
  took at ``next_obs`` -- which ``TensorReplayBuffer`` doesn't carry; see
  ``ReBRACReplayBuffer`` (``rl_garden/buffers/rebrac_replay_buffer.py``).
  The resulting ``critic_loss`` formula is unchanged from ``TD3BCCore``'s
  own (``((q-target_q)**2).mean(over batch).sum(over critics)``) -- reused
  verbatim via ``TD3BCCore._critic_loss``.
- **Actor** (``update_actor``, ``rebrac.py:425-474``), computed *after* the
  critic step with the freshly-updated critic (same ordering
  ``TD3BCCore.train()`` already uses): ``actor_loss = (actor_bc_coef *
  ((action-actions)**2).sum(-1) - lmbda * q_values).mean()``, where
  ``lmbda = stop_grad(1/abs(q_values).mean())`` if ``normalize_q`` else
  ``1``. Two real divergences from ``TD3BCCore``'s own actor loss, both
  verified against source, not assumed: (1) the BC penalty sums over the
  action dimension before averaging over the batch
  (``bc_penalty.sum(-1).mean()``), not ``F.mse_loss``'s mean-over-everything
  -- numerically different by a factor of ``action_dim``; (2) ``q_values``
  is ``min`` over the **full critic ensemble** (``rebrac.py:441``), not
  ``TD3BCCore``'s own ``q_values_all(...)[0]`` (first critic only).
- **``valid``/terminal handling**: unchanged from ``TD3BCCore`` --
  ``target_q = rewards + (1-dones)*gamma*next_q``, no chunking involved.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms.td3_bc import TD3BCCore
from rl_garden.buffers.rebrac_replay_buffer import ReBRACReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.utils import polyak_update
from rl_garden.networks import KernelInit
from rl_garden.networks.actor_critic import BackboneType


class ReBRACCore(TD3BCCore):
    """Shared ReBRAC loss/network logic. See module docstring."""

    def _init_rebrac_params(
        self,
        *,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_bc_coef: float = 1.0,
        critic_bc_coef: float = 1.0,
        normalize_q: bool = True,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        # TD3BCCore._init_td3bc_params owns tau/lrs/net_arch/n_critics/layer
        # norms/etc.; its own `alpha` field is unused here (ReBRAC replaces
        # it with actor_bc_coef/critic_bc_coef below) -- left at its default,
        # never read by this class's train().
        self._init_td3bc_params(
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        if actor_bc_coef < 0:
            raise ValueError(f"actor_bc_coef must be >= 0, got {actor_bc_coef}.")
        if critic_bc_coef < 0:
            raise ValueError(f"critic_bc_coef must be >= 0, got {critic_bc_coef}.")
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef
        self.normalize_q = normalize_q

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = super()._checkpoint_metadata()
        meta.pop("alpha", None)
        return {
            **meta,
            "actor_bc_coef": self.actor_bc_coef,
            "critic_bc_coef": self.critic_bc_coef,
            "normalize_q": self.normalize_q,
        }

    def _build_replay_buffer(self) -> ReBRACReplayBuffer:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                "ReBRAC only supports Box observation spaces, got " + str(type(obs_space))
            )
        return ReBRACReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        counts: dict[str, int] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            obs_features = self.policy.extract_features(data.obs)
            with torch.no_grad():
                next_features = self.policy.extract_features(data.next_obs)
                noise = (torch.randn_like(data.actions) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = (self.policy.actor_target(next_features) + noise).clamp(
                    self._action_low, self._action_high
                )
                critic_bc_penalty = (next_action - data.next_actions).pow(2).sum(
                    -1, keepdim=True
                )
                target_q_all = self.policy.q_values_all(
                    next_features, next_action, target=True
                )
                next_q = (
                    target_q_all.min(dim=0).values - self.critic_bc_coef * critic_bc_penalty
                )
                target_q = data.rewards.unsqueeze(-1) + self.gamma * (
                    1.0 - data.dones.unsqueeze(-1)
                ) * next_q

            q_all = self.policy.q_values_all(obs_features, data.actions, target=False)
            critic_loss = self._critic_loss(q_all, target_q)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            for key, value in (
                ("critic_loss", float(critic_loss.detach().item())),
                ("critic_bc_penalty", float(critic_bc_penalty.detach().mean().item())),
            ):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

            if self._global_update % self.policy_freq == 0:
                features_detached = obs_features.detach()
                pi_action = self.policy.actor(features_detached)
                actor_bc_penalty = (pi_action - data.actions).pow(2).sum(-1, keepdim=True)
                # min over the FULL critic ensemble (rebrac.py:441) -- unlike
                # TD3BCCore's own actor loss, which uses q_values_all(...)[0]
                # (first critic only). Uses the freshly-updated critic, same
                # ordering TD3BCCore.train() already relies on.
                q_values = self.policy.q_values_all(
                    features_detached, pi_action, target=False
                ).min(dim=0).values
                if self.normalize_q:
                    lmbda = (1.0 / q_values.abs().mean()).detach()
                else:
                    lmbda = torch.ones((), device=q_values.device, dtype=q_values.dtype)
                actor_loss = (
                    self.actor_bc_coef * actor_bc_penalty - lmbda * q_values
                ).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self._clip_grad_norm(self.policy.actor_parameters())
                self.actor_optimizer.step()
                if self._lr_schedulers[1] is not None:
                    self._lr_schedulers[1].step()

                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.policy.actor.parameters(),
                    self.policy.actor_target.parameters(),
                    self.tau,
                )

                for key, value in (
                    ("actor_loss", float(actor_loss.detach().item())),
                    ("actor_bc_penalty", float(actor_bc_penalty.detach().mean().item())),
                    ("lmbda", float(lmbda.detach().item())),
                ):
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class ReBRAC(ReBRACCore, OfflineRLAlgorithm):
    """Offline ReBRAC: TD3-BC + a critic-side BC penalty against the
    dataset's actual next-step action. See module docstring."""

    _compatible_checkpoint_algorithms = ("ReBRAC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 1024,
        gamma: float = 0.99,
        offline_sampling: str = "with_replace",
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_bc_coef: float = 1.0,
        critic_bc_coef: float = 1.0,
        normalize_q: bool = True,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 0,
        num_eval_steps: int = 50,
        eval_env: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            batch_size=batch_size,
            gamma=gamma,
            offline_sampling=offline_sampling,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            eval_env=eval_env,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._init_rebrac_params(
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            actor_bc_coef=actor_bc_coef,
            critic_bc_coef=critic_bc_coef,
            normalize_q=normalize_q,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"ReBRAC supports only Box observation spaces, got {type(obs_space)}"
            )

        self._setup_model()
