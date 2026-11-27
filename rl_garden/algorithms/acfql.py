"""ACFQL: Q-chunking's action-chunked, offline-to-online FQL
(``3rd_party/qc/agents/acfql.py``).

Ported as a new algorithm built directly on ``FQLCore`` + ``Off2OnReplayMixin``
+ ``OffPolicyAlgorithm``, following the ``CQLCore``/``_CQLRolloutTrainingShell``/
``CalQLCore``/``_CalQLRolloutTrainingShell``/``Off2OnCalQL`` precedent
(``cql.py``/``calql.py``/``off2on_calql.py``) for "an offline algorithm gains
an online off2on phase" -- ``FQL`` itself is ``OfflineRLAlgorithm``-based (no
rollout loop at all), so ACFQL cannot simply subclass it; it reuses
``FQLCore``'s network/loss *shape* while wiring into the online-capable
``OffPolicyAlgorithm`` shell instead. The CLI entrypoint reuses the
already-generic ``run_off2on`` runner (``rl_garden/training/off2on/runner.py``)
unmodified -- it already does exactly what QC's own ``main.py`` does (load an
offline dataset into ``agent.replay_buffer``, run offline gradient steps,
switch to online mode, continue via ``learn()``), so no new CLI orchestration
is needed, only a ``build_acfql`` callback (see
``rl_garden/training/off2on/entrypoints/acfql.py``).

``FQLCore.train()`` is fully monolithic (no SACCore-style hook seams to
override individually -- confirmed by reading it directly), so
``ACFQLCore.train()`` is a full override, not a partial one. What changes
from plain ``FQLCore.train()`` (verified against ``acfql.py`` directly):

- **Chunked critic target**: ``target_q = rewards + discounts * next_q``
  using the buffer's pre-accumulated H-step discounted reward/discount. A
  window that runs past its episode's end has zero weight in the critic
  loss, the mean still over the whole batch (``acfql.py:45``).
- **``next_action`` goes through the actor_type-aware dispatch**
  (``self.policy.predict(...)``), not a hardcoded ``actor_onestep_flow``
  call -- ``acfql.py``'s own ``critic_loss`` calls ``self.sample_actions``
  (the same dispatching method used everywhere), so ``best-of-n``'s
  expensive multi-sample selection also runs inside the critic's bootstrap,
  not just at rollout/eval time.
- **Chunked, valid-masked BC-flow loss**: ``(pred_vel - vel_target)`` reshaped
  to ``(B, horizon_length, action_dim)``, multiplied by
  ``data.valid[..., None]`` before ``.mean()`` (still divides by the full
  element count, not ``sum(valid)`` -- matches ``acfql.py:73-79`` exactly):
  per position, as an action past the episode's end belongs to another one.
- **``actor_type`` gates the distill/Q terms**: ``"best-of-n"`` sets
  ``distill_loss = q_loss = 0`` (no gradient into ``actor_onestep_flow`` at
  all, matching ``acfql.py:83-100``'s branch exactly) -- ``"distill-ddpg"``
  (default) keeps plain FQL's three-term loss shape unchanged.
- **``alpha`` default is 100.0**, not plain FQL's 10.0 (``acfql.py:329``).

Everything else (``_optimizer_names``, checkpoint metadata shape, network
construction via ``ACFQLPolicy``, ``EnsembleQCritic``, Polyak update,
gradient clipping) reuses ``FQLCore``'s existing methods directly, extended
only where the chunked action space requires it.

Action-space tiling, chunked replay construction, training-batch flattening,
evaluation queues and horizon metadata live in ``ActionChunkingMixin``.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F

from rl_garden.algorithms.chunked_rollout import ChunkedRolloutMixin
from rl_garden.algorithms.fql import FQLCore
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.common.utils import polyak_update
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.fql import ACFQLParams
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.observations import ObsGroups, resolve_obs_groups
from rl_garden.policies.acfql_policy import ACFQLPolicy, EncoderSharing


class ACFQLCore(FQLCore):
    """Chunked FQL loss/network overrides. See module docstring."""

    def _init_acfql_params(
        self,
        params: ACFQLParams,
        *,
        horizon_length: int,
    ) -> None:
        self._init_action_chunking(horizon_length)

    def _setup_model(self) -> None:
        observation_space = self.env.single_observation_space
        extractor_kwargs = self._policy_extractor_kwargs(observation_space)
        if self.encoder_sharing == "separate":
            actor_keys = resolve_obs_groups(
                self.observation_encoders.schema, self.obs_groups
            )["actor"].keys
            actor_bc_flow_encoder = build_observation_encoder(
                observation_space, self.encoder_config, keys=actor_keys
            )
        else:
            actor_bc_flow_encoder = None
        self.policy = ACFQLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            critic_use_group_norm=self.critic_use_group_norm,
            num_groups=self.num_groups,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            activation_fn=self.activation_fn,
            actor_bc_flow_encoder=actor_bc_flow_encoder,
            actor_type=self.actor_type,
            actor_num_samples=self.actor_num_samples,
            flow_steps=self.flow_steps,
            q_agg=self.q_agg,
            **extractor_kwargs,
        ).to(self.device)

        self.critic_optimizer = make_optimizer(
            list(self.policy.critic_and_encoder_parameters()),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers = [
            make_lr_scheduler(
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.critic_optimizer, self.actor_optimizer)
        ]

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        counts: dict[str, int] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)
            batch_size = data.actions.shape[0]
            flat_actions = data.actions
            full_action_dim = flat_actions.shape[-1]
            action_dim = full_action_dim // self.horizon_length

            obs_features = self.policy.extract_critic_features(data.obs)

            # --- critic ---
            with torch.no_grad():
                next_features_critic = self.policy.extract_critic_features(data.next_obs)
                next_action = self.policy.predict(data.next_obs, deterministic=False)
                target_q_all = self.policy.q_values_all(
                    next_features_critic, next_action, target=True
                )
                next_q = self._aggregate_target_q(target_q_all)
                target_q = (
                    data.rewards.unsqueeze(-1) + data.discounts.unsqueeze(-1) * next_q
                )

            complete_window = data.valid[:, -1].to(target_q.dtype).unsqueeze(-1)
            q_all = self.policy.q_values_all(obs_features, flat_actions, target=False)
            critic_loss = ((q_all - target_q).square() * complete_window).mean()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            # --- actor ---
            bc_features, onestep_features, q_features = self.policy.extract_actor_loss_features(
                data.obs, critic_features=obs_features
            )
            device, dtype = bc_features.device, bc_features.dtype

            x_0 = torch.randn(batch_size, full_action_dim, device=device, dtype=dtype)
            t = torch.rand(batch_size, 1, device=device, dtype=dtype)
            x_t = (1 - t) * x_0 + t * flat_actions
            vel_target = flat_actions - x_0
            pred_vel = self.policy.actor_bc_flow(bc_features, x_t, t)
            bc_flow_loss = (
                (pred_vel - vel_target).reshape(batch_size, self.horizon_length, action_dim) ** 2
                * data.valid.unsqueeze(-1)
            ).mean()

            if self.actor_type == "distill-ddpg":
                noises = torch.randn(batch_size, full_action_dim, device=device, dtype=dtype)
                with torch.no_grad():
                    target_flow_actions = self.policy.compute_flow_actions(
                        bc_features, noises, self.flow_steps
                    )
                actor_actions = self.policy.actor_onestep_flow(onestep_features, noises)
                distill_loss = F.mse_loss(actor_actions, target_flow_actions)

                clipped_actions = actor_actions.clamp(
                    self.policy.action_low, self.policy.action_high
                )
                q_all_pi = self.policy.q_values_all(q_features, clipped_actions, target=False)
                q_pi = q_all_pi.mean(dim=0)
                q_loss = -q_pi.mean()
                if self.normalize_q_loss:
                    lam = (1.0 / q_pi.abs().mean()).detach()
                    q_loss = lam * q_loss
            else:  # "best-of-n": no actor_onestep_flow gradient (acfql.py:83-100)
                distill_loss = torch.zeros((), device=device, dtype=dtype)
                q_loss = torch.zeros((), device=device, dtype=dtype)

            actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_grad_norm(self.policy.actor_parameters())
            self.actor_optimizer.step()
            if self._lr_schedulers[1] is not None:
                self._lr_schedulers[1].step()

            polyak_update(
                self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau
            )

            for key, value in (
                ("critic_loss", float(critic_loss.detach().item())),
                ("actor_loss", float(actor_loss.detach().item())),
                ("bc_flow_loss", float(bc_flow_loss.detach().item())),
                ("distill_loss", float(distill_loss.detach().item())),
                ("q_loss", float(q_loss.detach().item())),
            ):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class _ACFQLRolloutTrainingShell(Off2OnReplayMixin, ACFQLCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell wiring ``ACFQLCore`` into ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** Use :class:`ACFQL`. The
       shape and arguments of this shell may change without notice -- mirrors
       ``_CQLRolloutTrainingShell``/``_CalQLRolloutTrainingShell``'s existing
       precedent for this internal-extension-point pattern.
    """

    params_class = ACFQLParams

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        params: Optional[ACFQLParams] = None,
        horizon_length: int = 5,
        buffer_size: int = 1000000,
        buffer_device: str = 'cuda',
        learning_starts: int = 4000,
        batch_size: int = 256,
        gamma: float = 0.99,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = 'truncated',
        online_episodes_per_iteration: Optional[int] = None,
        stats_window_size: Optional[int] = None,
        actor_use_group_norm: bool = False,
        encoder_sharing: Optional[EncoderSharing] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        offline_sampling: Literal['with_replace', 'without_replace'] = 'with_replace',
        seed: int = 1,
        device: str | torch.device = 'auto',
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
        initial_training_phase: Optional[InitialTrainingPhase] = None,
        **overrides,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=params.tau,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            online_episodes_per_iteration=online_episodes_per_iteration,
            stats_window_size=stats_window_size,
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
        self._init_acfql_params(
            params,
            horizon_length=horizon_length,
        )
        self._init_fql_params(
            params,
            actor_use_group_norm=actor_use_group_norm,
            alpha=params.alpha,
            critic_dropout_rate=params.critic_dropout_rate,
            critic_encoder_config=critic_encoder_config,
            critic_use_group_norm=params.critic_use_group_norm,
            encoder_config=encoder_config,
            encoder_sharing=encoder_sharing,
            num_groups=params.num_groups,
            obs_groups=obs_groups,
            reference_compatible=False,
        )
        self._init_off2on_params(offline_sampling=offline_sampling)
        self._setup_model()


class ACFQL(ChunkedRolloutMixin, _ACFQLRolloutTrainingShell):
    """Q-chunking's action-chunked, offline-to-online FQL. See module docstring."""

    _compatible_checkpoint_algorithms = ("ACFQL",)

    params_class = ACFQLParams

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
