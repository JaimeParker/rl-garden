"""IDQL (Hansen-Estruch et al. 2023, arXiv:2304.10573): IQL-style expectile
value/critic regression paired with a diffusion actor instead of a Gaussian
one. State-based (Box observations) only, matching ``DiffusionMLP``'s scope.

Standalone -- does not subclass ``IQLCore`` (`rl_garden/algorithms/iql.py`).
``IQLCore``'s value/critic math is literally the same math IDQL needs, but
its ``_init_iql_params``/`_setup_model`` are ~110 lines tightly coupled to
the Gaussian actor (`actor_distribution`, `std_parameterization`,
`log_std_*`, `behavior_log_prob`) that would need overriding wholesale
anyway -- the genuinely reusable pieces (the 3-line expectile-loss formula,
a target-min-Q lookup, a polyak update) are cheap enough to duplicate here
that inheriting the mixin would mean "inherits half a mixin, fights the
other half." One real consumer today (this file), not two, so no shared
abstraction -- same reasoning already applied to ``ExPLORe`` not hooking
into ``PriorDataReplayMixin``. ``iql.py`` is not touched.

Known upstream-config ambiguity, not resolved: the reference launcher's
exact ``actor_architecture`` (``mlp`` vs ``ln_resnet``) for its released
runs could not be confirmed from the files read (``ln_resnet``-only kwargs
are present in the launcher config without the flag itself being set
there). Moot for this port since the actor is built on
``rl_garden.networks.DiffusionMLP``, not a resnet-based reverse encoder.

``actor_objective`` defaults to ``"bc"`` (unweighted diffusion BC) --
matches the actual upstream released config, not the paper's headline
advantage-weighted variant; ``soft_adv``/``hard_adv``/``exp_adv`` are
exposed as opt-in. ``predict(deterministic=...)`` reuses rl-garden's
existing ``deterministic`` convention for the reference's two inference
variants: ``True`` -> ``eval_actions``-style hard argmax-Q over ``N``
diffusion samples; ``False`` -> ``sample_implicit_policy``-style
expectile-weighted stochastic resample.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.idql_policy import IDQLPolicy

ActorObjective = Literal["bc", "soft_adv", "hard_adv", "exp_adv"]


class IDQL(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("IDQL",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        expectile: float = 0.7,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_tau: float = 0.001,
        actor_objective: ActorObjective = "bc",
        policy_temperature: float = 3.0,
        critic_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        diffusion_mlp_dims: Sequence[int] = (256, 256),
        denoising_steps: int = 5,
        schedule: Literal["cosine", "vp", "linear"] = "vp",
        n_action_samples: int = 64,
        critic_value_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        offline_sampling: str = "with_replace",
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
        self.expectile = expectile
        self.tau = tau
        self.actor_tau = actor_tau
        self.actor_objective = actor_objective
        self.policy_temperature = policy_temperature
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.value_hidden_dims = tuple(value_hidden_dims)
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size
        self.critic_use_layer_norm = critic_use_layer_norm
        self.value_use_layer_norm = value_use_layer_norm
        self.diffusion_mlp_dims = tuple(diffusion_mlp_dims)
        self.denoising_steps = denoising_steps
        self.schedule = schedule
        self.n_action_samples = n_action_samples
        self.critic_value_lr = critic_value_lr
        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm

        if not isinstance(self.env.single_observation_space, spaces.Box):
            raise TypeError(
                f"IDQL supports Box observation spaces only, got "
                f"{type(self.env.single_observation_space)}"
            )
        self._setup_model()

    def _setup_model(self) -> None:
        features_extractor = FlattenExtractor(self.env.single_observation_space)
        self.policy = IDQLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            critic_hidden_dims=self.critic_hidden_dims,
            value_hidden_dims=self.value_hidden_dims,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            critic_use_layer_norm=self.critic_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            diffusion_mlp_dims=self.diffusion_mlp_dims,
            denoising_steps=self.denoising_steps,
            schedule=self.schedule,
            n_action_samples=self.n_action_samples,
            expectile=self.expectile,
        ).to(self.device)

        self.critic_value_optimizer = make_optimizer(
            list(self.policy.critic_value_and_encoder_parameters()),
            lr=self.critic_value_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.actor_optimizer = make_optimizer(
            list(self.policy.net_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = TensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )
        self._lr_schedulers = [
            make_lr_scheduler(
                self.critic_value_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            ),
            make_lr_scheduler(
                self.actor_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            ),
        ]

    def _sample_train_batch(self, batch_size: int):
        return self.replay_buffer.sample(batch_size)

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return weight * diff.pow(2)

    def _actor_weight(self, adv: torch.Tensor) -> torch.Tensor:
        if self.actor_objective == "bc":
            return torch.ones_like(adv)
        if self.actor_objective == "soft_adv":
            return torch.where(adv > 0, self.expectile, 1.0 - self.expectile)
        if self.actor_objective == "hard_adv":
            return torch.where(adv >= -0.01, 1.0, 0.0)
        if self.actor_objective == "exp_adv":
            return torch.exp(adv * self.policy_temperature).clamp(max=100.0)
        raise ValueError(f"Unknown actor_objective: {self.actor_objective!r}")

    def _compute_losses(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        features = self.policy.extract_features(data.obs, stop_gradient=False)

        with torch.no_grad():
            target_q_for_value = self.policy.min_q_value(
                features.detach(),
                data.actions,
                subsample_size=self.critic_subsample_size,
                target=True,
            )
        values = self.policy.value(features)
        value_loss = self._expectile_loss(target_q_for_value - values).mean()

        q_pred = self.policy.q_values_all(features, data.actions, target=False)
        with torch.no_grad():
            next_features = self.policy.extract_features(data.next_obs, stop_gradient=False)
            next_v = self.policy.value(next_features)
            target_q = (
                data.rewards.unsqueeze(-1)
                + self.gamma * (1.0 - data.dones.unsqueeze(-1)) * next_v
            )
        critic_loss = F.mse_loss(q_pred, target_q.unsqueeze(0).expand_as(q_pred))

        with torch.no_grad():
            adv = (target_q_for_value - values).squeeze(-1)
            weight = self._actor_weight(adv)
        actor_loss = self.policy.diffusion_loss(data.obs, data.actions, weight=weight)

        total_loss = value_loss + critic_loss + actor_loss
        metrics = {
            "loss": float(total_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "q": float(q_pred.detach().mean().item()),
            "target_q": float(target_q.detach().mean().item()),
            "v": float(values.detach().mean().item()),
            "adv": float(adv.detach().mean().item()),
        }
        return total_loss, metrics

    def _polyak_update(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(
                self.policy.critic.parameters(), self.policy.critic_target.parameters()
            ):
                p_targ.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_targ in zip(
                self.policy.net.parameters(), self.policy.target_net.parameters()
            ):
                p_targ.data.mul_(1.0 - self.actor_tau).add_(p.data, alpha=self.actor_tau)

    def _clip_grad_norm(self) -> None:
        if self.grad_clip_norm is None:
            return
        params = list(self.policy.critic_value_and_encoder_parameters()) + list(
            self.policy.net_parameters()
        )
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            self.critic_value_optimizer.zero_grad(set_to_none=True)
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._compute_losses(data)
            loss.backward()
            self._clip_grad_norm()
            self.critic_value_optimizer.step()
            self.actor_optimizer.step()
            for sched in self._lr_schedulers:
                if sched is not None:
                    sched.step()
            self._polyak_update()

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        del compute_info
        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_value_optimizer", "actor_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "expectile": self.expectile,
            "tau": self.tau,
            "actor_tau": self.actor_tau,
            "actor_objective": self.actor_objective,
            "policy_temperature": self.policy_temperature,
            "critic_hidden_dims": self.critic_hidden_dims,
            "value_hidden_dims": self.value_hidden_dims,
            "n_critics": self.n_critics,
            "critic_subsample_size": self.critic_subsample_size,
            "diffusion_mlp_dims": self.diffusion_mlp_dims,
            "denoising_steps": self.denoising_steps,
            "schedule": self.schedule,
            "n_action_samples": self.n_action_samples,
        }
