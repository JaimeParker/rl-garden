"""MeanFlow Behavioral Cloning (MeanFlowBC) for offline datasets.

Pure offline algorithm: trains an actor to imitate expert actions via the
MeanFlow identity loss (no Q-function, no separate distillation phase).
Structurally a near-verbatim copy of ``FlowBC`` (``rl_garden/algorithms/
flow_bc.py``), with ``MeanFlowBCPolicy`` (``MeanFlowActorField``-based)
swapped in for ``FlowBCPolicy``. See ``rl_garden/networks/mean_flow_field.py``
for the ported algorithm and its time-convention derivation.

Observation encoding is schema-driven via ``ObservationEncoderMixin``
(``encoder_config``/``obs_groups``, resolved in ``_setup_model``), matching
``SAC``'s convention -- no critic, so ``encoder_sharing`` is fixed to
``"shared"`` (never exposed as a constructor kwarg): the encoder is trained
end-to-end by the actor loss, the only loss that ever touches it, and
``"shared_critic_grad"`` would otherwise stop-gradient it (see
``BasePolicy.extract_actor_features``'s formula, which detaches whenever
``critic_extractor is None`` under that sharing mode).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional, Sequence

import torch

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.mean_flow_field import MeanFlowMode
from rl_garden.observations import ObsGroups
from rl_garden.policies.mean_flow_bc_policy import MeanFlowBCPolicy


class MeanFlowBC(OfflineRLAlgorithm):
    """MeanFlow Behavioral Cloning: single-stage, one-step-sampleable
    imitation via the MeanFlow identity loss."""

    _compatible_checkpoint_algorithms = ("MeanFlowBC",)
    _SUPPORTED_POLICY_KWARGS = frozenset(
        {"actor_extractor_class", "actor_extractor_kwargs"}
    )
    # See module docstring: no critic, so extract_actor_features must never
    # stop-gradient.
    encoder_sharing = "shared"

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        offline_sampling: str = "with_replace",
        actor_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        net_arch: Optional[Sequence[int]] = None,
        num_sample_steps: int = 1,
        mode: MeanFlowMode = "i-meanflow",
        time_dist_mu: float = 0.4,
        time_dist_sigma: float = 1.0,
        adaptive_l2_gamma: float = 0.0,
        adaptive_l2_c: float = 1e-2,
        actor_use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = None,
        activation_fn: Optional[Activation] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        image_augmentation_seed: Optional[int] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
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
            gamma=0.99,  # not used by MeanFlowBC; inherited field
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
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.net_arch: list[int] = (
            list(net_arch) if net_arch is not None else [512, 512, 512, 512]
        )
        self.num_sample_steps = num_sample_steps
        self.mode = mode
        self.time_dist_mu = time_dist_mu
        self.time_dist_sigma = time_dist_sigma
        self.adaptive_l2_gamma = adaptive_l2_gamma
        self.adaptive_l2_c = adaptive_l2_c
        self.actor_use_layer_norm = actor_use_layer_norm
        self.kernel_init = kernel_init
        self.activation_fn = activation_fn

        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self._image_augmentation_seed = image_augmentation_seed

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()

    # --- checkpoint ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
            "actor_lr": self.actor_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "net_arch": self.net_arch,
            "num_sample_steps": self.num_sample_steps,
            "mode": self.mode,
            "time_dist_mu": self.time_dist_mu,
            "time_dist_sigma": self.time_dist_sigma,
            "adaptive_l2_gamma": self.adaptive_l2_gamma,
            "adaptive_l2_c": self.adaptive_l2_c,
            "encoder_sharing": self.encoder_sharing,
            "encoder_sharing_origin": self.encoder_sharing_origin,
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
            "image_augmentation_seed": self._image_augmentation_seed,
        }
        return meta

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

    # --- model setup ---

    def _setup_model(self) -> None:
        extractor_kwargs = self._policy_extractor_kwargs(
            self.env.single_observation_space,
            augmentation_seed=self._image_augmentation_seed,
        )
        self.policy = MeanFlowBCPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            actor_extractor=extractor_kwargs["actor_extractor"],
            net_arch=self.net_arch,
            use_layer_norm=self.actor_use_layer_norm,
            kernel_init=self.kernel_init,
            activation_fn=self.activation_fn,
            num_sample_steps=self.num_sample_steps,
            mode=self.mode,
            time_dist_mu=self.time_dist_mu,
            time_dist_sigma=self.time_dist_sigma,
            adaptive_l2_gamma=self.adaptive_l2_gamma,
            adaptive_l2_c=self.adaptive_l2_c,
            encoder_sharing=extractor_kwargs["encoder_sharing"],
        ).to(self.device)

        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers = [
            make_lr_scheduler(
                self.actor_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
        ]

    # --- training ---

    def _compute_losses(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        actor_loss, aux = self.policy.mean_flow_loss(data.obs, data.actions)
        metrics = {
            "loss": float(actor_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "fm_loss": float(aux["fm_loss"].item()),
            "mf_loss": float(aux["mf_loss"].item()),
            "mf_v_mse": float(aux["mf_v_mse"].item()),
        }
        return actor_loss, metrics

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch()

            self.actor_optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._compute_losses(data)
            loss.backward()
            self._clip_grad_norm()
            self.actor_optimizer.step()
            self._step_schedulers()

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    # --- helpers (mirrored from BC/FlowBC) ---

    def _sample_train_batch(self):
        if self.offline_sampling == "with_replace":
            return self.replay_buffer.sample(self.batch_size)
        if self.offline_sampling == "without_replace":
            sample = getattr(self.replay_buffer, "sample_without_repeat", None)
            if sample is None:
                raise ValueError(
                    "offline_sampling='without_replace' requires a replay buffer "
                    "with sample_without_repeat()."
                )
            return sample(self.batch_size)
        raise ValueError(f"Unknown offline_sampling: {self.offline_sampling!r}")

    def _step_schedulers(self) -> None:
        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

    def _clip_grad_norm(self) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.actor_parameters()), self.grad_clip_norm
        )

    def _normalize_policy_kwargs(
        self, policy_kwargs: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        normalized = dict(policy_kwargs or {})
        unsupported = sorted(set(normalized) - self._SUPPORTED_POLICY_KWARGS)
        if unsupported:
            raise ValueError(
                "Unsupported policy_kwargs keys: "
                + ", ".join(unsupported)
                + ". Supported keys are: "
                + ", ".join(sorted(self._SUPPORTED_POLICY_KWARGS))
                + "."
            )
        return normalized

    def _build_replay_buffer(self):
        # obs_space is always Dict (boundary normalization is unconditional).
        return ReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )
