"""Teacher-student policy distillation: on-policy action regression.

Ports rsl_rl's (ETH Zurich/NVIDIA) teacher-student distillation pattern --
used for sim-to-real legged-robot locomotion, where a privileged-observation
"teacher" is trained first and a realistic-observation "student" is then
distilled from it via on-policy rollout -- as a framework-level mechanism
that sits *beside* PPO, not on top of it, and that accepts a frozen teacher
from any rl-garden algorithm (any ``BasePolicy``), not just PPO.

Why this is not just ``PPO`` with a different loss, and not a
``RolloutBuffer``/``OnPolicyAlgorithm.learn()`` reuse: distillation trains no
critic and has no reward-based objective -- the loss is pure action
regression (student's action vs. the frozen teacher's action on the same
observation). ``OnPolicyAlgorithm.learn()`` unconditionally calls
``_predict_last_values()``/``compute_returns_and_advantage()`` (GAE) and
``RolloutBuffer`` unconditionally stores ``values``/``log_probs``/
``advantages``/``returns`` -- none of which distillation has any use for.
Faking zero-valued placeholders through that path would be dead complexity
for no benefit, so ``PolicyDistillation`` overrides ``learn()`` entirely
(same overall shape -- rollout collection -> train() -> logging/checkpoint --
reusing every generic inherited piece from ``OnPolicyAlgorithm``/
``BaseAlgorithm``) and uses its own ``DistillationRolloutBuffer`` instead of
``RolloutBuffer``. This mirrors this codebase's own precedent:
``DAgger(DemoInterventionMixin, BC)`` similarly overrides ``learn()``/
``train()`` wholesale because ``OfflineRLAlgorithm.learn()``'s pure
gradient-step loop doesn't fit env-interleaved training either.

The student is a plain ``BCPolicy`` (actor-only, no critic -- exactly the
shape a distillation student needs, zero new policy class required), built
via the same Dict-obs dispatch (``CombinedExtractor``) that ``BC``/
``FlowBC``/``VisionDiffusionBC`` each already copy independently, sliced to
``student_obs_keys``. The teacher is accepted as an already-built,
already-frozen ``BasePolicy`` object -- ``PolicyDistillation`` never
constructs or loads it itself, which is what actually makes "any rl-garden
algorithm can be the teacher" true: every algorithm's policy already
implements the same ``predict(obs, deterministic)`` contract.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.buffers.distillation_rollout_buffer import DistillationRolloutBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_optimizer
from rl_garden.common.types import Obs
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.bc_policy import BCPolicy


def _select_keys(obs: dict, keys: Sequence[str]) -> dict:
    return {key: obs[key] for key in keys}


class PolicyDistillation(OnPolicyAlgorithm):
    """On-policy teacher-student distillation: regress a student's action
    onto a frozen teacher's action, collected via the student's own rollout.

    TODO(dagger-variant): a future variant could mix in the teacher's action
    as the *executed* rollout action per a beta schedule (see the comment at
    the action-selection line in ``learn()`` below) instead of always
    executing the student's own action.
    """

    _compatible_checkpoint_algorithms = ("PolicyDistillation",)

    def __init__(
        self,
        env: Any,
        teacher_policy: BasePolicy,
        teacher_obs_keys: Sequence[str],
        student_obs_keys: Sequence[str],
        eval_env: Optional[Any] = None,
        *,
        num_steps: int = 50,
        num_learning_epochs: int = 5,
        num_minibatches: int = 4,
        loss_type: Literal["mse", "huber"] = "mse",
        actor_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        max_grad_norm: Optional[float] = None,
        net_arch: Optional[Sequence[int]] = None,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        actor_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        tanh_squash: bool = True,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        # gamma/gae_lambda are OnPolicyAlgorithm constructor requirements
        # that distillation never reads (no reward-based objective, no GAE
        # -- see module docstring). finite_horizon_gae likewise unused.
        super().__init__(
            env=env,
            eval_env=eval_env,
            num_steps=num_steps,
            gamma=1.0,
            gae_lambda=1.0,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            finite_horizon_gae=False,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_final_checkpoint=save_final_checkpoint,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Dict):
            raise TypeError(
                "PolicyDistillation requires a Dict observation space "
                "(needs both teacher_obs_keys and student_obs_keys as Dict "
                f"keys), got {type(obs_space)}."
            )
        missing = [
            key
            for key in (*teacher_obs_keys, *student_obs_keys)
            if key not in obs_space.spaces
        ]
        if missing:
            raise ValueError(
                "teacher_obs_keys/student_obs_keys reference unknown obs "
                f"keys: {missing}. Available: {sorted(obs_space.spaces)}."
            )
        self.teacher_obs_keys = list(teacher_obs_keys)
        self.student_obs_keys = list(student_obs_keys)

        self.teacher_policy = teacher_policy.to(self.device)
        self.teacher_policy.eval()
        for param in self.teacher_policy.parameters():
            param.requires_grad_(False)

        self.num_learning_epochs = num_learning_epochs
        self.num_minibatches = num_minibatches
        self.loss_fn = F.mse_loss if loss_type == "mse" else F.smooth_l1_loss
        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.max_grad_norm = max_grad_norm
        self.net_arch: list[int] = list(net_arch) if net_arch is not None else [256, 256]
        self.actor_use_layer_norm = actor_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.std_parameterization = std_parameterization
        self.tanh_squash = tanh_squash

        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_keys = image_keys if image_keys is not None else ("rgb", "depth")
        self._state_key = state_key if state_key is not None else "state"
        self._use_proprio = use_proprio if use_proprio is not None else True
        self._proprio_latent_dim = (
            proprio_latent_dim if proprio_latent_dim is not None else 64
        )
        self._image_fusion_mode = (
            image_fusion_mode if image_fusion_mode is not None else "stack_channels"
        )
        self._enable_stacking = enable_stacking if enable_stacking is not None else False

        self._setup_model()

    # --- checkpoint ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("policy_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **BaseAlgorithm._checkpoint_metadata(self),
            "num_steps": self.num_steps,
            "teacher_obs_keys": self.teacher_obs_keys,
            "student_obs_keys": self.student_obs_keys,
            "num_learning_epochs": self.num_learning_epochs,
            "num_minibatches": self.num_minibatches,
            "actor_lr": self.actor_lr,
            "net_arch": self.net_arch,
        }

    # --- model setup ---

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        student_obs_space = spaces.Dict(
            {key: obs_space[key] for key in self.student_obs_keys}
        )
        self._student_obs_space = student_obs_space

        features_extractor = CombinedExtractor(
            observation_space=student_obs_space,
            image_keys=self._image_keys,
            state_key=self._state_key,
            image_encoder_factory=self._image_encoder_factory,
            proprio_latent_dim=self._proprio_latent_dim,
            use_proprio=self._use_proprio,
            fusion_mode=self._image_fusion_mode,
            enable_stacking=self._enable_stacking,
        )
        self.policy = BCPolicy(
            observation_space=student_obs_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            use_layer_norm=self.actor_use_layer_norm,
            use_group_norm=self.actor_use_group_norm,
            num_groups=self.num_groups,
            dropout_rate=self.actor_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            std_parameterization=self.std_parameterization,
            tanh_squash=self.tanh_squash,
        ).to(self.device)

        self.policy_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.distillation_buffer = DistillationRolloutBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
        )

    # --- evaluation: student only sees its own obs slice ---

    def _eval_action(self, obs: Obs) -> torch.Tensor:
        with torch.no_grad():
            student_obs = _select_keys(
                self._obs_to_policy_device(obs), self.student_obs_keys
            )
            return self.policy.predict(student_obs, deterministic=True)

    # --- training ---

    def train(self) -> dict[str, float]:
        metrics_sum: dict[str, float] = {}
        num_updates = 0
        minibatch_size = max(1, self.distillation_buffer.buffer_size // self.num_minibatches)
        self.policy.train()
        for _ in range(self.num_learning_epochs):
            for batch in self.distillation_buffer.get(minibatch_size):
                self._global_update += 1
                num_updates += 1
                student_obs = _select_keys(batch.obs, self.student_obs_keys)
                # Recomputed fresh from current student parameters each
                # gradient step -- the teacher's action was stored once at
                # rollout time (it's frozen, so recomputing it would give the
                # same result at extra cost; see DistillationRolloutBuffer's
                # docstring).
                pred_action = self.policy.predict(student_obs, deterministic=True)
                loss = self.loss_fn(pred_action, batch.teacher_actions)

                self.policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy.actor_parameters()), self.max_grad_norm
                    )
                self.policy_optimizer.step()

                metrics_sum["behavior"] = metrics_sum.get("behavior", 0.0) + float(
                    loss.detach().item()
                )
        return {key: value / num_updates for key, value in metrics_sum.items()}

    # --- outer loop (see module docstring for why this can't be inherited) ---

    def learn(self, total_timesteps: int) -> "PolicyDistillation":
        obs, _ = self.env.reset(seed=self.seed)
        cumulative: dict[str, float] = defaultdict(float)

        while self._global_step < total_timesteps:
            previous_step = self._global_step
            if self.eval_freq > 0 and self._global_update % self.eval_freq == 0:
                stime = time.perf_counter()
                eval_metrics = self._evaluate()
                if self.logger is not None:
                    self._log_eval_metrics(eval_metrics, self._global_step)
                    self.logger.add_scalar(
                        "time/eval_time", time.perf_counter() - stime, self._global_step
                    )
                if self.std_log:
                    eval_return = self._first_metric(eval_metrics, ("return",))
                    eval_success = self._first_metric(
                        eval_metrics, ("success_at_end", "success_once")
                    )
                    print(
                        "[eval] "
                        f"step={self._global_step}/{total_timesteps} "
                        f"return={self._fmt_metric(eval_return)} "
                        f"success_at_end={self._fmt_metric(eval_success)}",
                        flush=True,
                    )

            self.distillation_buffer.reset()
            rollout_t = time.perf_counter()
            rollout_episode_metrics: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.num_steps):
                self._global_step += self.num_envs
                device_obs = self._obs_to_policy_device(obs)
                student_obs = _select_keys(device_obs, self.student_obs_keys)
                teacher_obs = _select_keys(device_obs, self.teacher_obs_keys)
                with torch.no_grad():
                    # Student always executes its own action -- no
                    # beta-mixing.
                    # TODO(dagger-variant): mix in teacher_action as the
                    # *executed* action per a beta schedule here, instead of
                    # always executing the student's own action.
                    action = self.policy.predict(student_obs, deterministic=False)
                    teacher_action = self.teacher_policy.predict(
                        teacher_obs, deterministic=True
                    ).detach()

                next_obs, _rewards, terminations, truncations, infos = self.env.step(
                    action
                )
                next_done = torch.logical_or(terminations, truncations).to(self.device)
                self.distillation_buffer.add(obs, teacher_action, next_done)

                if "final_info" in infos:
                    fi = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for key, value in fi["episode"].items():
                        done_values = value[done_mask]
                        if done_values.numel() == 0:
                            continue
                        mean_value = float(done_values.float().mean().item())
                        self._log_rollout_metric(key, mean_value, self._global_step)
                        rollout_episode_metrics[key].append(mean_value)
                obs = next_obs
            rollout_time = time.perf_counter() - rollout_t
            cumulative["rollout_time"] += rollout_time

            update_t = time.perf_counter()
            losses = self.train()
            post_update_hook = getattr(self.env, "post_update_sync", None)
            if post_update_hook is not None:
                post_update_hook()
            update_time = time.perf_counter() - update_t
            cumulative["update_time"] += update_time

            should_log = (
                self.log_freq > 0
                and (self._global_step - self.batch_size) // self.log_freq
                < self._global_step // self.log_freq
            )
            if should_log:
                rollout_fps = (
                    self.batch_size / rollout_time if rollout_time > 0 else float("nan")
                )
                if self.logger is not None:
                    self._log_update_metrics(losses, self._global_step)
                    self.logger.add_scalar(
                        "time/update_time", update_time, self._global_step
                    )
                    self.logger.add_scalar(
                        "time/rollout_time", rollout_time, self._global_step
                    )
                    self.logger.add_scalar(
                        "time/rollout_fps", rollout_fps, self._global_step
                    )
                    for key, value in cumulative.items():
                        self.logger.add_scalar(f"time/total_{key}", value, self._global_step)
                if self.std_log:
                    episode_means = {
                        key: float(sum(values) / len(values))
                        for key, values in rollout_episode_metrics.items()
                        if len(values) > 0
                    }
                    rollout_return = self._first_metric(episode_means, ("return",))
                    rollout_success = self._first_metric(
                        episode_means, ("success_at_end", "success_once")
                    )
                    progress = 100.0 * self._global_step / total_timesteps
                    print(
                        "[train] "
                        f"step={self._global_step}/{total_timesteps} ({progress:.2f}%) "
                        f"return={self._fmt_metric(rollout_return)} "
                        f"success_at_end={self._fmt_metric(rollout_success)} "
                        f"fps={self._fmt_metric(rollout_fps)}",
                        flush=True,
                    )
            self._maybe_save_periodic_checkpoint(previous_step)

        if self.checkpoint_dir is not None and self.save_final_checkpoint:
            self._save_checkpoint("final.pt")
        return self
