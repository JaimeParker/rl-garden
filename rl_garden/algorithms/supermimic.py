"""``SuperMimic`` (arXiv 2610.28480, Algorithm A.1): online model-based RL with
a differentiable explicit-state world model and a deterministic policy trained
by backpropagation through imagined rollouts.

Each iteration collects ``sim_steps_per_iteration`` vectorized env steps with
the deterministic policy plus Gaussian action noise (the executed, noisy
action is what the replay buffer stores), then runs
``updates_per_iteration`` updates. One update is:

1. World model (Eq. 1): ``wm_batches`` windows of ``wm_window`` steps, one
   Adam step each on ``StateSpaceWorldModel.model_loss`` (k-step
   autoregressive loss plus the learned heads' terms).
2. Actor (Eq. 2): ``horizon`` imagined steps from replay start rows with
   their in-episode observation history; ``a_t = pi(o_hist, g_t)`` (no
   noise), ``s_{t+1} = f(s_t, a_t)``, the next observation comes from the
   world model's observation head plus the task's uniform sensor noise. The
   start history is used as stored (it already carries the env's sensor
   noise), so every frame is noised exactly once.
   With soft termination ``d_{t+1}`` and survival ``c_0 = 1``,
   ``c_{t+1} = c_t (1 - d_{t+1})``:
   ``J = sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)``. The world model
   and critic are frozen (``requires_grad`` off) but the graph runs through
   them. Gradients are clipped by global norm, then element-wise; a step with
   a non-finite gradient norm is skipped and counted.
3. Critic (Eq. 3): on the same (detached) imagined trajectory,
   ``R_t = r_{t+1} + gamma (1 - d_{t+1}) ((1 - lam) V_{t+1} + lam R_{t+1})``,
   ``R_H = V_H``; ``critic_epochs`` epochs over ``critic_minibatches``
   shuffled minibatches, MSE, no target network.

The first ``warmup_updates`` updates train only the world model (an
``InitialTrainingPhase`` with actor and critic off). Learning rates follow a
cosine decay from ``lr`` to ``lr * lr_final_ratio`` over
``lr_decay_updates`` updates, the same value for all three optimizers.

Update-to-data ratio: rl-garden's ``utd`` is per collected transition. The
paper's 0.4 updates per vectorized env step is expressed here as
``sim_steps_per_iteration=5, updates_per_iteration=2``;
``grad_steps_per_iteration`` is set to ``updates_per_iteration`` exactly.

Observation roles (Dict keys): ``state`` is the policy observation;
``state_sim`` the modeled state (default: ``state``); ``state_ctx`` the
per-episode context, ``state_goal`` the goal and ``state_time`` the time
index, each optional. Without an ``AnalyticMDP`` the goal stays at its start
value during imagination (the learned heads cannot advance it).

Heads: ``heads="learned"`` uses learned reward and termination heads and a
learned observation head (identity when there is no ``state_sim`` key);
``heads="analytic"`` takes all three from ``env.unwrapped.analytic_mdp`` and
requires the ``state_time`` key, from which the analytic heads read the time
index (a time-independent MDP still emits a zero column).

``SuperMimicPolicy`` is a plain ``nn.Module`` outside the ``BasePolicy``
contract (same class of exception as ``PWMPolicy``): the critic reads the
world model's explicit state features rather than an observation, the
actor's imagined observations come from the world model's observation head,
and no observation encoder is trained. It holds the world model, so the model
is saved under the checkpoint's ``policy`` namespace.
"""
from __future__ import annotations

import contextlib
import math
from collections import defaultdict
from typing import Any, Iterator, Literal, Optional

import torch
import torch.nn as nn

from rl_garden.algorithms.model_based import ModelBasedAlgorithm
from rl_garden.buffers.state_sequence_replay_buffer import (
    HistorySample,
    StateSequenceReplayBuffer,
)
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.networks.mlp import create_mlp
from rl_garden.networks.returns import lambda_return
from rl_garden.networks.running_norm import RunningNorm
from rl_garden.world_models.base import State
from rl_garden.world_models.batch import SequenceBatch
from rl_garden.world_models.state_space import StateCodec, StateSpaceWorldModel

OBS_KEY = "state"
STATE_KEY = "state_sim"
CTX_KEY = "state_ctx"
GOAL_KEY = "state_goal"
TIME_KEY = "state_time"
# Imagination start rejection sampling (analytic MDP ``valid_start``).
_MAX_START_SAMPLING_ROUNDS = 10


@contextlib.contextmanager
def _parameters_frozen(*modules: nn.Module) -> Iterator[None]:
    """Temporarily ``requires_grad_(False)`` on every parameter of ``modules``
    (restored on exit). Graphs built inside still differentiate through the
    modules' activations, but never accumulate into their parameters."""
    flags = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def _select_history(sample: HistorySample, index: torch.Tensor) -> HistorySample:
    return HistorySample(
        obs={key: value[:, index] for key, value in sample.obs.items()},
        prev_action=sample.prev_action[index],
        obs_valid=sample.obs_valid[:, index],
    )


def _concat_history(samples: list[HistorySample], batch_size: int) -> HistorySample:
    return HistorySample(
        obs={key: torch.cat([s.obs[key] for s in samples], dim=1)[:, :batch_size] for key in samples[0].obs},
        prev_action=torch.cat([s.prev_action for s in samples], dim=0)[:batch_size],
        obs_valid=torch.cat([s.obs_valid for s in samples], dim=1)[:, :batch_size],
    )


def _push_frame(history: torch.Tensor, restart: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Rows flagged in ``restart`` are first refilled with ``frame`` (first-
    frame replication), then every row shifts left and appends ``frame``.
    ``history`` ``(N, L, D)``, ``restart`` ``(N,)`` bool, ``frame`` ``(N, D)``."""
    frame = frame.unsqueeze(1)
    history = torch.where(restart.view(-1, 1, 1), frame.expand_as(history), history)
    return torch.cat([history[:, 1:], frame], dim=1)


class SuperMimicPolicy(nn.Module):
    """Goal encoder (``tanh`` latent), observation-history decoder (the
    deterministic actor), state-value critic and their input normalizers,
    plus the world model. Outside the ``BasePolicy`` contract (module docstring)."""

    def __init__(
        self,
        *,
        world_model: StateSpaceWorldModel,
        obs_dim: int,
        goal_dim: int,
        ctx_dim: int,
        action_dim: int,
        critic_extra_dim: int,
        obs_history_len: int = 1,
        latent_dim: int = 32,
        goal_encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        decoder_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        critic_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.ctx_dim = ctx_dim
        self.action_dim = action_dim
        self.obs_history_len = obs_history_len

        self.goal_encoder = (
            create_mlp(goal_dim, latent_dim, goal_encoder_dims, activation_fn=nn.ELU) if goal_dim > 0 else None
        )
        decoder_in_dim = (latent_dim if goal_dim > 0 else 0) + obs_history_len * obs_dim
        self.decoder = create_mlp(decoder_in_dim, action_dim, decoder_dims, activation_fn=nn.ELU)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()

        self.critic_in_dim = world_model.latent_dim + ctx_dim + goal_dim + critic_extra_dim
        self.critic = create_mlp(self.critic_in_dim, 1, critic_dims, activation_fn=nn.ELU)

        self.obs_norm = RunningNorm((obs_dim,))
        self.goal_norm = RunningNorm((goal_dim,)) if goal_dim > 0 else None
        self.critic_norm = RunningNorm((self.critic_in_dim,))

    def act(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """``obs_hist`` ``(B, L, obs_dim)``, ``goal`` ``(B, goal_dim)`` ->
        ``(B, action_dim)``, unsquashed."""
        parts = []
        if self.goal_encoder is not None:
            parts.append(torch.tanh(self.goal_encoder(self.goal_norm(goal))))
        parts.append(self.obs_norm(obs_hist).flatten(1))
        return self.decoder(torch.cat(parts, dim=-1))

    def value(self, critic_in: torch.Tensor) -> torch.Tensor:
        """``(..., critic_in_dim)`` -> ``(..., 1)``."""
        return self.critic(self.critic_norm(critic_in))

    def actor_parameters(self) -> list[nn.Parameter]:
        encoder = list(self.goal_encoder.parameters()) if self.goal_encoder is not None else []
        return encoder + list(self.decoder.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.critic.parameters())


class SuperMimicCore:
    """Mixin (no base class) with SuperMimic's updates and optimizers. Hosts
    call ``_init_supermimic_params(...)`` before ``_setup_model()``, set
    ``self.policy`` (``SuperMimicPolicy``), ``self.action_low``/``action_high``
    (``(action_dim,)`` tensors) and call ``_build_supermimic_optimizers()``.
    ``_sample_train_batch`` returns ``{"windows": [SequenceBatch],
    "starts": HistorySample}``."""

    def _init_supermimic_params(
        self,
        *,
        horizon: int = 32,
        wm_window: int = 8,
        wm_batches: int = 10,
        wm_batch_size: int = 2048,
        imagination_batch: int = 4096,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 2e-4,
        lr_final_ratio: float = 0.03,
        lr_decay_updates: int = 250_000,
        actor_betas: tuple[float, float] = (0.9, 0.95),
        wm_grad_norm: float = 1.0,
        actor_grad_norm: float = 1.0,
        actor_grad_value: float = 0.1,
        critic_grad_norm: float = 1.0,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
        action_noise_std: float = 0.1,
        obs_history_len: int = 1,
        warmup_updates: int = 500,
        amp_dtype: Optional[str] = None,
    ) -> None:
        positive = {
            "horizon": horizon,
            "wm_window": wm_window,
            "wm_batches": wm_batches,
            "wm_batch_size": wm_batch_size,
            "imagination_batch": imagination_batch,
            "lr_decay_updates": lr_decay_updates,
            "critic_epochs": critic_epochs,
            "critic_minibatches": critic_minibatches,
            "obs_history_len": obs_history_len,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")
        if warmup_updates < 0:
            raise ValueError(f"warmup_updates must be >= 0, got {warmup_updates}.")
        if not 0.0 <= lr_final_ratio <= 1.0:
            raise ValueError(f"lr_final_ratio must be in [0, 1], got {lr_final_ratio}.")
        if action_noise_std < 0:
            raise ValueError(f"action_noise_std must be >= 0, got {action_noise_std}.")
        if amp_dtype not in (None, "bf16"):
            raise ValueError(f"amp_dtype must be None or 'bf16', got {amp_dtype!r}.")
        self.horizon = horizon
        self.wm_window = wm_window
        self.wm_batches = wm_batches
        self.wm_batch_size = wm_batch_size
        self.imagination_batch = imagination_batch
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.lr_final_ratio = lr_final_ratio
        self.lr_decay_updates = lr_decay_updates
        self.actor_betas = tuple(actor_betas)
        self.wm_grad_norm = wm_grad_norm
        self.actor_grad_norm = actor_grad_norm
        self.actor_grad_value = actor_grad_value
        self.critic_grad_norm = critic_grad_norm
        self.critic_epochs = critic_epochs
        self.critic_minibatches = critic_minibatches
        self.action_noise_std = action_noise_std
        self.obs_history_len = obs_history_len
        self.warmup_updates = warmup_updates
        self.amp_dtype = amp_dtype

    # ------------------------------------------------------------------
    # Optimizers and schedule
    # ------------------------------------------------------------------

    def _build_supermimic_optimizers(self) -> None:
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=self.lr, betas=self.actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic_parameters(), lr=self.lr)

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("world_optimizer", "actor_optimizer", "critic_optimizer")

    def _set_learning_rates(self) -> float:
        """Cosine decay to ``lr * lr_final_ratio`` over ``lr_decay_updates``.
        ``_global_update`` is already incremented for the current update, so
        the first update uses the initial lr (as PWM)."""
        progress = min(max(self._global_update - 1, 0), self.lr_decay_updates) / self.lr_decay_updates
        ratio = self.lr_final_ratio + (1.0 - self.lr_final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.lr * ratio
        for optimizer in (self.world_optimizer, self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = lr
        return lr

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.amp_dtype == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Imagination (Eq. 2)
    # ------------------------------------------------------------------

    def _critic_input(self, state: State) -> torch.Tensor:
        """``[features(s), ctx, goal, mdp.critic_features]`` for any leading dims."""
        world_model = self.world_model
        s = state["s"]
        parts = [world_model.features(state).float(), state["ctx"].float(), state["goal"].float()]
        if world_model.mdp is not None:
            parts.append(world_model.mdp.critic_features(s, state["ctx"], state["t"]).float())
        return torch.cat(parts, dim=-1)

    def _add_observation_noise(self, obs: torch.Tensor) -> torch.Tensor:
        mdp = self.world_model.mdp
        noise = None if mdp is None else mdp.obs_noise
        if noise is None:
            return obs
        return obs + noise.to(obs) * (2.0 * torch.rand_like(obs) - 1.0)

    def _imagine(self, starts: HistorySample) -> dict[str, Any]:
        world_model, policy = self.world_model, self.policy
        with self._autocast(), _parameters_frozen(world_model, policy.critic):
            state = world_model.state_from_obs({key: value[-1] for key, value in starts.obs.items()})
            obs_hist = starts.obs[world_model.obs_key].float().transpose(0, 1)
            goal = state["goal"]
            states, actions = [state], []
            for _ in range(self.horizon):
                action = torch.clamp(policy.act(obs_hist, goal).float(), self.action_low, self.action_high)
                state = world_model.step(state, action)
                obs = self._add_observation_noise(world_model.observation(state, action).float())
                obs_hist = torch.cat([obs_hist[:, 1:], obs.unsqueeze(1)], dim=1)
                goal = state["goal"]
                states.append(state)
                actions.append(action)

            trajectory = {key: torch.stack([s[key] for s in states], dim=0) for key in states[0]}
            action_seq = torch.stack(actions, dim=0)
            prev_actions = torch.cat([starts.prev_action.float().unsqueeze(0), action_seq[:-1]], dim=0)
            before = {key: value[:-1] for key, value in trajectory.items()}
            after = {key: value[1:] for key, value in trajectory.items()}
            rewards = world_model.transition_reward(before, action_seq, after, prev_actions).squeeze(-1).float()
            dones = world_model.termination_prob(after).squeeze(-1).float()

            alive = torch.cumprod(1.0 - dones, dim=0)
            survival = torch.cat([torch.ones_like(alive[:1]), alive[:-1]], dim=0)
            discounts = self.gamma ** torch.arange(self.horizon, device=rewards.device, dtype=rewards.dtype)
            final_state = {key: value[-1] for key, value in trajectory.items()}
            value_h = policy.value(self._critic_input(final_state)).squeeze(-1).float()
            objective = (discounts.unsqueeze(-1) * survival * rewards).sum(dim=0)
            objective = objective + self.gamma**self.horizon * alive[-1] * value_h
        return {
            "loss": -objective.mean(),
            "rewards": rewards,
            "dones": dones,
            "states": trajectory,
            "survival_H": alive[-1].mean(),
            "value_H": value_h.mean(),
        }

    def _actor_step(self, rollout: dict[str, Any]) -> dict[str, float]:
        params = self.policy.actor_parameters()
        self.actor_optimizer.zero_grad(set_to_none=True)
        rollout["loss"].backward()
        grad_norm = nn.utils.clip_grad_norm_(params, self.actor_grad_norm)
        nn.utils.clip_grad_value_(params, self.actor_grad_value)
        finite = bool(torch.isfinite(grad_norm))
        if finite:
            self.actor_optimizer.step()
        return {
            "actor/loss": float(rollout["loss"].detach()),
            "actor/grad_norm": float(grad_norm),
            "actor/skipped": float(not finite),
            "imag/reward_mean": float(rollout["rewards"].detach().mean()),
            "imag/survival_H": float(rollout["survival_H"].detach()),
            "imag/value_H": float(rollout["value_H"].detach()),
        }

    # ------------------------------------------------------------------
    # Critic (Eq. 3)
    # ------------------------------------------------------------------

    def _critic_targets(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """``(H, B)`` lambda-returns; ``rewards``/``dones`` ``(H, B)`` hold
        ``r_{t+1}``/``d_{t+1}``, ``values`` ``(H + 1, B)``. ``lambda_return``
        reads row ``t + 1`` for target ``t``, hence the leading zero rows."""
        zero_row = torch.zeros_like(rewards[:1])
        return lambda_return(
            torch.cat([zero_row, rewards], dim=0),
            values,
            values,
            last=torch.zeros_like(values),
            term=torch.cat([zero_row, dones], dim=0),
            disc=self.gamma,
            lam=self.lam,
        )

    def _update_critic(self, critic_in: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """``critic_epochs`` epochs over ``critic_minibatches`` shuffled
        minibatches of the flattened ``(H * B)`` rows."""
        inputs = critic_in.reshape(-1, critic_in.shape[-1])
        flat_targets = targets.reshape(-1)
        params = self.policy.critic_parameters()
        losses, grad_norms, skipped = [], [], 0
        for _ in range(self.critic_epochs):
            permutation = torch.randperm(inputs.shape[0], device=inputs.device)
            for index in permutation.tensor_split(self.critic_minibatches):
                if index.numel() == 0:
                    continue
                with self._autocast():
                    prediction = self.policy.value(inputs[index]).squeeze(-1).float()
                loss = (prediction - flat_targets[index]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(params, self.critic_grad_norm)
                if bool(torch.isfinite(grad_norm)):
                    self.critic_optimizer.step()
                else:
                    skipped += 1
                losses.append(loss.detach())
                grad_norms.append(grad_norm.detach())
        return {
            "critic/loss": float(torch.stack(losses).mean()),
            "critic/grad_norm": float(torch.stack(grad_norms).mean()),
            "critic/skipped": float(skipped),
        }

    # ------------------------------------------------------------------
    # ModelBasedStepMixin hooks
    # ------------------------------------------------------------------

    def _update_model(self, batch: dict[str, Any]) -> tuple[dict[str, float], State]:
        self._set_learning_rates()
        params = list(self.world_model.parameters())
        records: dict[str, list[torch.Tensor]] = defaultdict(list)
        skipped = 0
        posterior: State = {}
        for window in batch["windows"]:
            with self._autocast():
                losses, posterior = self.world_model.model_loss(window)
            self.world_optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = nn.utils.clip_grad_norm_(params, self.wm_grad_norm)
            if bool(torch.isfinite(grad_norm)):
                self.world_optimizer.step()
            else:
                skipped += 1
            for name, key in (
                ("wm/loss", "total"),
                ("wm/state", "state"),
                ("wm/reward", "reward"),
                ("wm/observation", "observation"),
                ("wm/termination", "termination"),
                ("wm/err_last", "state_err_last"),
            ):
                records[name].append(losses[key].detach().float())
            records["wm/grad_norm"].append(grad_norm.detach().float())
        metrics = {name: float(torch.stack(values).mean()) for name, values in records.items()}
        metrics["wm/skipped"] = float(skipped)
        return metrics, posterior

    def _update_actor_critic(self, batch: dict[str, Any], posterior: State) -> dict[str, float]:
        del posterior  # imagination starts come from batch["starts"].
        if not self._training_update_mask().update_actor:
            return {}
        lr = self._set_learning_rates()
        rollout = self._imagine(batch["starts"])
        actor_metrics = self._actor_step(rollout)
        with torch.no_grad():
            critic_in = self._critic_input(rollout["states"])
            self.policy.critic_norm.update(critic_in)
            with self._autocast():
                values = self.policy.value(critic_in).squeeze(-1).float()
            targets = self._critic_targets(rollout["rewards"].detach(), rollout["dones"].detach(), values)
        critic_metrics = self._update_critic(critic_in[:-1], targets)
        return {**actor_metrics, **critic_metrics, "lr": lr}

    def _update_targets(self) -> None:
        """No target networks."""

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon": self.horizon,
            "wm_window": self.wm_window,
            "wm_batches": self.wm_batches,
            "wm_batch_size": self.wm_batch_size,
            "imagination_batch": self.imagination_batch,
            "gamma": self.gamma,
            "lam": self.lam,
            "lr": self.lr,
            "lr_final_ratio": self.lr_final_ratio,
            "lr_decay_updates": self.lr_decay_updates,
            "actor_betas": list(self.actor_betas),
            "wm_grad_norm": self.wm_grad_norm,
            "actor_grad_norm": self.actor_grad_norm,
            "actor_grad_value": self.actor_grad_value,
            "critic_grad_norm": self.critic_grad_norm,
            "critic_epochs": self.critic_epochs,
            "critic_minibatches": self.critic_minibatches,
            "action_noise_std": self.action_noise_std,
            "obs_history_len": self.obs_history_len,
            "warmup_updates": self.warmup_updates,
            "amp_dtype": self.amp_dtype,
        }


class SuperMimic(SuperMimicCore, ModelBasedAlgorithm):
    """Online SuperMimic on any rl-garden vector env (module docstring)."""

    _compatible_checkpoint_algorithms = ("SuperMimic",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        heads: Literal["learned", "analytic"] = "learned",
        codec: Optional[StateCodec] = None,
        sim_steps_per_iteration: int = 5,
        updates_per_iteration: int = 2,
        buffer_size: int = 4_096_000,
        buffer_device: str = "cuda",
        learning_starts: Optional[int] = None,
        wm_hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        head_hidden_dims: tuple[int, ...] = (256, 256),
        latent_dim: int = 32,
        goal_encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        decoder_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        critic_dims: tuple[int, ...] = (256, 128),
        horizon: int = 32,
        wm_window: int = 8,
        wm_batches: int = 10,
        wm_batch_size: int = 2048,
        imagination_batch: int = 4096,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 2e-4,
        lr_final_ratio: float = 0.03,
        lr_decay_updates: int = 250_000,
        actor_betas: tuple[float, float] = (0.9, 0.95),
        wm_grad_norm: float = 1.0,
        actor_grad_norm: float = 1.0,
        actor_grad_value: float = 0.1,
        critic_grad_norm: float = 1.0,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
        action_noise_std: float = 0.1,
        obs_history_len: int = 1,
        warmup_updates: int = 500,
        amp_dtype: Optional[str] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 0,
        num_eval_steps: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        if heads not in ("learned", "analytic"):
            raise ValueError(f"heads must be 'learned' or 'analytic', got {heads!r}.")
        if sim_steps_per_iteration < 1 or updates_per_iteration < 1:
            raise ValueError("sim_steps_per_iteration and updates_per_iteration must be >= 1.")
        num_envs = env.num_envs
        training_freq = num_envs * sim_steps_per_iteration
        super().__init__(
            env,
            eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=(wm_window + 2) * num_envs if learning_starts is None else learning_starts,
            batch_size=wm_batch_size,
            gamma=gamma,
            training_freq=training_freq,
            utd=updates_per_iteration / training_freq,
            bootstrap_at_done="truncated",
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
            initial_training_phase=InitialTrainingPhase(
                duration_steps=warmup_updates * training_freq,
                update_actor=False,
                update_critic=False,
                update_encoder=False,
            ),
        )
        self.grad_steps_per_iteration = updates_per_iteration
        self.sim_steps_per_iteration = sim_steps_per_iteration
        self.updates_per_iteration = updates_per_iteration
        self.heads = heads
        self.codec = codec
        self.wm_hidden_dims = tuple(wm_hidden_dims)
        self.head_hidden_dims = tuple(head_hidden_dims)
        self.latent_dim = latent_dim
        self.goal_encoder_dims = tuple(goal_encoder_dims)
        self.decoder_dims = tuple(decoder_dims)
        self.critic_dims = tuple(critic_dims)
        self._init_supermimic_params(
            horizon=horizon,
            wm_window=wm_window,
            wm_batches=wm_batches,
            wm_batch_size=wm_batch_size,
            imagination_batch=imagination_batch,
            gamma=gamma,
            lam=lam,
            lr=lr,
            lr_final_ratio=lr_final_ratio,
            lr_decay_updates=lr_decay_updates,
            actor_betas=actor_betas,
            wm_grad_norm=wm_grad_norm,
            actor_grad_norm=actor_grad_norm,
            actor_grad_value=actor_grad_value,
            critic_grad_norm=critic_grad_norm,
            critic_epochs=critic_epochs,
            critic_minibatches=critic_minibatches,
            action_noise_std=action_noise_std,
            obs_history_len=obs_history_len,
            warmup_updates=warmup_updates,
            amp_dtype=amp_dtype,
        )
        self._obs_hist: Optional[torch.Tensor] = None
        self._hist_reset: Optional[torch.Tensor] = None
        self._last_action: Optional[torch.Tensor] = None
        self._eval_hist: Optional[torch.Tensor] = None
        self._eval_hist_reset: Optional[torch.Tensor] = None
        self._eval_hist_rebuild = True
        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _resolve_analytic_mdp(self, obs_dim: int, goal_dim: int) -> Any:
        mdp = getattr(self.env.unwrapped, "analytic_mdp", None)
        if mdp is None:
            raise ValueError(
                "heads='analytic' needs the env to provide an AnalyticMDP as "
                "env.unwrapped.analytic_mdp; use heads='learned' otherwise."
            )
        if TIME_KEY not in self.env.single_observation_space.spaces:
            raise ValueError(
                f"heads='analytic' reads the time index from the {TIME_KEY!r} observation key, "
                f"which the env does not provide; a time-independent AnalyticMDP still needs the "
                f"env to emit a zero {TIME_KEY!r} column."
            )
        if (mdp.obs_dim, mdp.goal_dim) != (obs_dim, goal_dim):
            raise ValueError(
                f"AnalyticMDP dims (obs_dim={mdp.obs_dim}, goal_dim={mdp.goal_dim}) do not match "
                f"the observation space (state={obs_dim}, {GOAL_KEY}={goal_dim})."
            )
        return mdp

    def _setup_model(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        space = self.env.single_observation_space

        def width(key: str) -> int:
            return int(space[key].shape[-1]) if key in space.spaces else 0

        obs_dim = width(OBS_KEY)
        state_key = STATE_KEY if STATE_KEY in space.spaces else OBS_KEY
        state_dim = width(state_key)
        ctx_dim, goal_dim = width(CTX_KEY), width(GOAL_KEY)
        action_space = self.env.single_action_space
        action_dim = int(action_space.shape[-1])

        if self.heads == "analytic":
            mdp = self._resolve_analytic_mdp(obs_dim, goal_dim)
            head_modes = {"reward_head": "analytic", "observation_head": "analytic", "termination_head": "analytic"}
        else:
            mdp = None
            observation_head = "learned" if state_key == STATE_KEY else "identity"
            head_modes = {"reward_head": "learned", "observation_head": observation_head, "termination_head": "learned"}

        world_model = StateSpaceWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            obs_dim=obs_dim,
            ctx_dim=ctx_dim,
            goal_dim=goal_dim,
            codec=self.codec,
            mdp=mdp,
            hidden_dims=self.wm_hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            state_key=STATE_KEY,
            obs_key=OBS_KEY,
            ctx_key=CTX_KEY,
            goal_key=GOAL_KEY,
            time_key=TIME_KEY,
            **head_modes,
        )
        critic_extra_dim = 0
        if mdp is not None:
            probe = mdp.critic_features(
                torch.zeros(1, state_dim, device=self.device),
                torch.zeros(1, ctx_dim, device=self.device),
                torch.zeros(1, dtype=torch.long, device=self.device),
            )
            critic_extra_dim = int(probe.shape[-1])

        self.policy = SuperMimicPolicy(
            world_model=world_model,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            ctx_dim=ctx_dim,
            action_dim=action_dim,
            critic_extra_dim=critic_extra_dim,
            obs_history_len=self.obs_history_len,
            latent_dim=self.latent_dim,
            goal_encoder_dims=self.goal_encoder_dims,
            decoder_dims=self.decoder_dims,
            critic_dims=self.critic_dims,
        ).to(self.device)
        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.replay_buffer = StateSequenceReplayBuffer(
            observation_space=space,
            action_space=action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            cross_episode=False,
            horizon=self.wm_window,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )
        self._build_supermimic_optimizers()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _policy_goal(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs[OBS_KEY]
        if self.policy.goal_dim == 0:
            return state.new_zeros(state.shape[0], 0).float()
        return obs[GOAL_KEY].float()

    def _noisy_policy_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            action = self.policy.act(self._obs_hist, self._policy_goal(obs))
            noisy = action + self.action_noise_std * torch.randn_like(action)
            return torch.clamp(noisy, self.action_low, self.action_high)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "heads": self.heads,
            "sim_steps_per_iteration": self.sim_steps_per_iteration,
            "updates_per_iteration": self.updates_per_iteration,
        }

    def _on_env_reset(self, obs) -> None:
        frame = self._obs_to_policy_device(obs)[OBS_KEY].float()
        self._obs_hist = frame.unsqueeze(1).repeat(1, self.obs_history_len, 1)
        self._hist_reset = torch.zeros(frame.shape[0], dtype=torch.bool, device=self.device)

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        del learning_has_started  # no random-exploration phase.
        obs = self._obs_to_policy_device(obs)
        self._obs_hist = _push_frame(self._obs_hist, self._hist_reset, obs[OBS_KEY].float())
        self._hist_reset = torch.zeros_like(self._hist_reset)
        action = self._noisy_policy_action(obs)
        self._last_action = action
        return action, action, None

    def _explore_action(self, obs) -> torch.Tensor:
        return self._noisy_policy_action(self._obs_to_policy_device(obs))

    def _replay_buffer_add_kwargs(
        self,
        action_context: Optional[dict[str, Any]],
        obs,
        next_obs,
        real_next_obs,
        infos,
        need_final_obs: torch.Tensor,
    ) -> dict[str, Any]:
        """Updates the running normalizers from the collected transition (the
        only rollout hook that sees both ``obs`` and ``real_next_obs``); adds
        no replay kwargs."""
        del action_context, next_obs, infos, need_final_obs
        obs = self._obs_to_policy_device(obs)
        self.world_model.update_normalizers(obs, self._last_action, self._obs_to_policy_device(real_next_obs))
        self.policy.obs_norm.update(obs[OBS_KEY].float())
        if self.policy.goal_norm is not None:
            self.policy.goal_norm.update(obs[GOAL_KEY].float())
        return {}

    def _post_rollout_step(
        self,
        action_context: Optional[dict[str, Any]],
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos,
    ) -> None:
        del action_context, infos
        self._hist_reset = self._hist_reset | (terminations | truncations).to(self.device).bool()

    # ------------------------------------------------------------------
    # Training batch
    # ------------------------------------------------------------------

    def _valid_starts(self, starts: HistorySample) -> torch.Tensor:
        t = self.world_model.state_from_obs({key: value[-1] for key, value in starts.obs.items()})["t"]
        return self.world_model.mdp.valid_start(t, self.horizon).bool()

    def _sample_imagination_starts(self) -> HistorySample:
        """Start rows with history; with an analytic MDP, rows whose
        ``valid_start`` is false are rejected (2x oversampling rounds)."""
        starts = self.replay_buffer.sample_history(self.imagination_batch, self.obs_history_len)
        if self.heads != "analytic":
            return starts
        valid = self._valid_starts(starts)
        if bool(valid.all()):
            return starts
        kept = [_select_history(starts, valid.nonzero(as_tuple=True)[0])]
        count = int(valid.sum())
        for _ in range(_MAX_START_SAMPLING_ROUNDS):
            if count >= self.imagination_batch:
                break
            candidates = self.replay_buffer.sample_history(2 * self.imagination_batch, self.obs_history_len)
            valid = self._valid_starts(candidates)
            kept.append(_select_history(candidates, valid.nonzero(as_tuple=True)[0]))
            count += int(valid.sum())
        if count < self.imagination_batch:
            raise RuntimeError(
                f"Found only {count} of {self.imagination_batch} valid imagination starts after "
                f"{_MAX_START_SAMPLING_ROUNDS} rounds; the replay buffer holds too few rows the "
                f"AnalyticMDP accepts for a {self.horizon}-step imagination."
            )
        return _concat_history(kept, self.imagination_batch)

    def _sample_train_batch(self) -> dict[str, Any]:
        windows = []
        for _ in range(self.wm_batches):
            sample = self.replay_buffer.sample(self.wm_batch_size)
            windows.append(
                SequenceBatch(obs=sample.obs, action=sample.action, reward=sample.reward, terminated=sample.terminated)
            )
        return {"windows": windows, "starts": self._sample_imagination_starts()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_start_hook(self) -> None:
        self._eval_hist_rebuild = True

    def _eval_action(self, obs) -> torch.Tensor:
        obs = self._obs_to_policy_device(obs)
        frame = obs[OBS_KEY].float()
        if self._eval_hist_rebuild or self._eval_hist is None:
            self._eval_hist = frame.unsqueeze(1).repeat(1, self.obs_history_len, 1)
            self._eval_hist_reset = torch.zeros(frame.shape[0], dtype=torch.bool, device=self.device)
            self._eval_hist_rebuild = False
        self._eval_hist = _push_frame(self._eval_hist, self._eval_hist_reset, frame)
        self._eval_hist_reset = torch.zeros_like(self._eval_hist_reset)
        with torch.no_grad():
            action = self.policy.act(self._eval_hist, self._policy_goal(obs))
        return torch.clamp(action, self.action_low, self.action_high)

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
        self._eval_hist_reset = self._eval_hist_reset | (terminations | truncations).to(self.device).bool()
