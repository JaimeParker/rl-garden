"""``PWM``: Policy learning with large World Models (Georgiev et al. 2024,
arXiv 2407.02466), offline multitask path (``3rd_party/PWM/src/pwm/algorithms
/pwm.py`` ``PWM.update`` driven by ``scripts/train_multitask.py``).

A SHAC-style first-order actor-critic on a **frozen** pretrained
``MultitaskWorldModel`` bound to one task (``BoundTaskWorldModel``). Each
gradient step:

1. Start states: ``obs[0]`` of a window sampled from the task's offline
   episodes, encoded under ``no_grad`` (``_update_model``).
2. Actor (``_actor_loss``): differentiable imagination over ``horizon``
   steps, ``a_t = tanh(actor(z_t.detach()).rsample()) * action_mask``; loss
   ``-(sum_t gamma^t r_t + gamma^H min_k V_k(z_H))`` per sample, divided by
   ``sqrt(ret_rms.var + 1e-5)`` and averaged. ``r_t`` is the model reward in
   symlog space (``reward_value(..., symlog_space=True)``).
3. Critic: truncated TD(lambda) targets from the same imagined rollout using
   the pre-update critic's min-ensemble values, MSE on all critics,
   ``critic_iterations`` epochs over ``critic_batches`` chronological
   minibatches. No target network.

Deviations from the official code: truncated TD(lambda) critic target instead of
the code's one-step TD; ``gamma^H`` bootstrap instead of ``gamma^(H+1)``; padded
action dims are masked to 0 during imagination; the world model is rl-garden's
``MultitaskWorldModel``.

``PWMPolicy`` is a plain ``nn.Module`` outside the ``BasePolicy`` contract
(same class of exception as ``MultitaskTDMPC2Policy``): its input is the
world-model latent, not an observation, so there is no observation/action
space or ``actor_extractor`` to declare, and no ``encoder_sharing``
attribute (no observation encoder is trained). The world model is NOT part
of the policy: it is frozen, loaded from ``world_model_checkpoint`` and kept
out of every optimizer and out of the checkpoint (its source is recorded in
the checkpoint metadata instead).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.offline_model_based import OfflineModelBasedAlgorithm
from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer
from rl_garden.buffers.mmap_storage import MmapMode
from rl_garden.common.checkpoint import load_filtered_state_dict
from rl_garden.common.logger import Logger
from rl_garden.networks.returns import lambda_return
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.bound_task import BoundTaskWorldModel
from rl_garden.world_models.imagine import imagine
from rl_garden.world_models.multitask_latent_consistency import MultitaskWorldModel

WORLD_MODEL_FILENAME = "world_model.pt"
WORLD_MODEL_MANIFEST_FILENAME = "manifest.json"
# PWM's world model is trained with a 16-step consistency rollout and rho 0.99;
# a much shorter-sighted model is out of distribution for the actor's imagination.
MIN_WORLD_MODEL_RHO = 0.9


# ---------------------------------------------------------------------------
# World-model loading (TDMPC2Multitask checkpoint or converter output)
# ---------------------------------------------------------------------------


def build_world_model_from_manifest(manifest: dict[str, Any]) -> MultitaskWorldModel:
    """Constructs an (uninitialized) ``MultitaskWorldModel`` matching a
    ``tools/conversion/convert_tdmpc2_checkpoint.py`` manifest."""
    num_tasks = int(manifest["num_tasks"])
    return MultitaskWorldModel(
        num_tasks=num_tasks,
        obs_dims=[int(manifest["obs_dim"])] * num_tasks,
        action_dims=[int(d) for d in manifest["action_dims"]],
        task_dim=int(manifest["task_dim"]),
        latent_dim=int(manifest["latent_dim"]),
        enc_dim=int(manifest["enc_dim"]),
        num_enc_layers=int(manifest["num_enc_layers"]),
        mlp_dim=int(manifest["mlp_dim"]),
        simnorm_dim=int(manifest["simnorm_dim"]),
        num_bins=int(manifest["num_bins"]),
        vmin=float(manifest["vmin"]),
        vmax=float(manifest["vmax"]),
    )


def load_pwm_world_model(path: str | Path) -> tuple[MultitaskWorldModel, dict[str, Any]]:
    """Loads a frozen-to-be ``MultitaskWorldModel`` from either

    (a) an rl-garden ``TDMPC2Multitask`` checkpoint file (architecture from
        its metadata, weights from ``state["policy"]`` under ``world_model.``),
    (b) a converter output: a directory holding ``world_model.pt`` +
        ``manifest.json``, or the ``world_model.pt`` file itself.

    Returns ``(world_model, source)``; ``source`` is JSON-serializable
    provenance (format, path, tasks/obs_dims/action_dims when known).
    """
    path = Path(path)
    weights_path = path / WORLD_MODEL_FILENAME if path.is_dir() else path
    manifest_path = weights_path.parent / WORLD_MODEL_MANIFEST_FILENAME
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "metadata" in payload and "state" in payload:
        metadata = payload["metadata"]
        if metadata.get("algorithm_class") != "TDMPC2Multitask":
            raise ValueError(
                f"{weights_path} is a {metadata.get('algorithm_class')!r} checkpoint; "
                "PWM needs a TDMPC2Multitask checkpoint or a converted world model."
            )
        hp = metadata["hyperparameters"]
        world_model = MultitaskWorldModel(
            num_tasks=len(hp["tasks"]),
            obs_dims=hp["obs_dims"],
            action_dims=hp["action_dims"],
            task_dim=hp["task_dim"],
            latent_dim=hp["latent_dim"],
            enc_dim=hp["enc_dim"],
            num_enc_layers=hp["num_enc_layers"],
            mlp_dim=hp["mlp_dim"],
            simnorm_dim=hp["simnorm_dim"],
            num_bins=hp["num_bins"],
            vmin=hp["vmin"],
            vmax=hp["vmax"],
            rho=hp["rho"],
        )
        load_filtered_state_dict(world_model, payload["state"]["policy"], prefix="world_model", strict=True)
        source = {
            "format": "tdmpc2_multitask_checkpoint",
            "path": str(weights_path),
            "tasks": list(hp["tasks"]),
            "obs_dims": list(hp["obs_dims"]),
            "action_dims": list(hp["action_dims"]),
            "horizon": hp.get("horizon", "unknown"),
            "rho": hp.get("rho", "unknown"),
        }
        return world_model, source

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{weights_path} is not a TDMPC2Multitask checkpoint and has no sibling "
            f"{WORLD_MODEL_MANIFEST_FILENAME} (converter output expected)."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    world_model = build_world_model_from_manifest(manifest)
    world_model.load_state_dict(payload, strict=True)
    source = {
        "format": "converted_tdmpc2_checkpoint",
        "path": str(weights_path),
        "manifest": manifest,
        "tasks": manifest.get("tasks"),
        "obs_dims": None,
        "action_dims": list(manifest["action_dims"]),
        "horizon": "unknown",  # the converter manifest does not record them
        "rho": "unknown",
    }
    return world_model, source


def warn_on_world_model_rollout_mismatch(source: dict[str, Any], horizon: int) -> None:
    """Warns when the world model's training horizon differs from PWM's
    imagination ``horizon`` or its ``rho`` is below ``MIN_WORLD_MODEL_RHO``.
    Skipped for values the source does not record (``"unknown"``)."""
    trained_horizon, rho = source.get("horizon", "unknown"), source.get("rho", "unknown")
    if trained_horizon != "unknown" and int(trained_horizon) != horizon:
        warnings.warn(
            f"World model was trained with horizon {trained_horizon}, PWM imagines {horizon} steps.",
            stacklevel=2,
        )
    if rho != "unknown" and float(rho) < MIN_WORLD_MODEL_RHO:
        warnings.warn(
            f"World model was trained with rho {rho} < {MIN_WORLD_MODEL_RHO}; PWM expects a long-horizon model.",
            stacklevel=2,
        )


def validate_world_model_for_dataset(
    world_model: MultitaskWorldModel,
    source: dict[str, Any],
    *,
    tasks: list[str],
    obs_dims: list[int],
    action_dims: list[int],
) -> None:
    """Fails loudly when the world model was not trained on this dataset's
    task table (task count/order, padded obs/action dims)."""
    errors = []
    if world_model.num_tasks != len(tasks):
        errors.append(f"num_tasks {world_model.num_tasks} != dataset {len(tasks)}")
    if world_model.obs_dim != max(obs_dims):
        errors.append(f"padded obs_dim {world_model.obs_dim} != dataset {max(obs_dims)}")
    if world_model.action_dim != max(action_dims):
        errors.append(f"padded action_dim {world_model.action_dim} != dataset {max(action_dims)}")
    for key, expected in (("tasks", tasks), ("obs_dims", obs_dims), ("action_dims", action_dims)):
        known = source.get(key)
        if known is not None and list(known) != list(expected):
            errors.append(f"{key} {list(known)} != dataset {list(expected)}")
    if errors:
        raise ValueError("World model does not match the dataset:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _pwm_mlp(in_dim: int, units: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """PWM's MLP layout: ``Linear -> Mish -> LayerNorm`` per hidden layer,
    plain ``Linear`` output (``models/actor.py``/``models/critic.py``)."""
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden in units:
        layers += [nn.Linear(dim, hidden), nn.Mish(), nn.LayerNorm(hidden)]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class PWMActor(nn.Module):
    """Gaussian actor with a state-independent ``logstd`` parameter
    (``ActorStochasticMLP``). Returns PRE-tanh actions; the caller squashes."""

    def __init__(
        self, latent_dim: int, action_dim: int, units: tuple[int, ...], init_logstd: float, min_logstd: float
    ) -> None:
        super().__init__()
        self.mu_net = _pwm_mlp(latent_dim, units, action_dim)
        self.logstd = nn.Parameter(torch.full((action_dim,), float(init_logstd)))
        self.min_logstd = float(min_logstd)

    def forward(self, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():  # PWM clamps the parameter data, not the graph.
            self.logstd.clamp_(min=self.min_logstd)
        mu = self.mu_net(z)
        if deterministic:
            return mu
        return mu + self.logstd.exp() * torch.randn_like(mu)


class PWMCritic(nn.Module):
    """``num_critics`` independent ``V(z)`` MLPs, orthogonal init (gain
    sqrt(2), zero bias) as ``CriticMLP``. ``forward`` returns ``(K, ..., 1)``."""

    def __init__(self, latent_dim: int, units: tuple[int, ...], num_critics: int) -> None:
        super().__init__()
        self.critics = nn.ModuleList(_pwm_mlp(latent_dim, units, 1) for _ in range(num_critics))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.0))
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(z) for critic in self.critics], dim=0)


class ReturnRunningMeanStd(nn.Module):
    """Running mean/var of per-sample actor losses (PWM's
    ``utils/running_mean_std.py``, count initialized to ``epsilon``). Buffers,
    so the statistics are checkpointed with the policy. Not
    ``rl_garden.common.reward_normalizer.RunningMeanStd``: that one is
    ``torch.compile``d and starts from count 0."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)


class PWMPolicy(nn.Module):
    """Actor + critic ensemble + return normalizer over world-model latents.
    Out of the ``BasePolicy`` contract (see module docstring)."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        actor_units: tuple[int, ...],
        critic_units: tuple[int, ...],
        num_critics: int,
        init_logstd: float,
        min_logstd: float,
    ) -> None:
        super().__init__()
        self.actor = PWMActor(latent_dim, action_dim, actor_units, init_logstd, min_logstd)
        self.critic = PWMCritic(latent_dim, critic_units, num_critics)
        self.ret_rms = ReturnRunningMeanStd()


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class PWMCore:
    """Mixin (no base class) with PWM's actor/critic losses and optimizers.
    Hosts call ``_init_pwm_params(...)`` before ``_setup_model()``, set
    ``self.policy`` (``PWMPolicy``), ``self.bound_world_model`` and
    ``self.action_mask`` (``(action_dim,)``), then call
    ``_build_pwm_optimizers()``. The host's ``_update_model`` returns the
    start latents as ``{"z": (B, latent)}``."""

    def _init_pwm_params(
        self,
        *,
        horizon: int,
        gamma: float,
        lam: float,
        actor_lr: float,
        critic_lr: float,
        lr_final: float,
        lr_decay_steps: int,
        actor_grad_norm: float,
        critic_grad_norm: float,
        critic_iterations: int,
        critic_batches: int,
        num_critics: int,
        actor_units: tuple[int, ...],
        critic_units: tuple[int, ...],
        init_logstd: float,
        min_logstd: float,
        ret_rms: bool,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if critic_batches < 1 or critic_iterations < 1:
            raise ValueError("critic_batches and critic_iterations must be >= 1.")
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_final = lr_final
        self.lr_decay_steps = lr_decay_steps
        self.actor_grad_norm = actor_grad_norm
        self.critic_grad_norm = critic_grad_norm
        self.critic_iterations = critic_iterations
        self.critic_batches = critic_batches
        self.num_critics = num_critics
        self.actor_units = tuple(actor_units)
        self.critic_units = tuple(critic_units)
        self.init_logstd = init_logstd
        self.min_logstd = min_logstd
        self.use_ret_rms = ret_rms

    @property
    def world_model(self) -> WorldModel:
        return self.bound_world_model

    def _build_pwm_optimizers(self) -> None:
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.actor_lr, betas=(0.7, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.critic_lr, betas=(0.7, 0.95)
        )

    def _set_learning_rates(self) -> tuple[float, float]:
        """Linear decay from the initial lr to ``lr_final`` over
        ``lr_decay_steps`` gradient steps (PWM ``update_lrs``). ``_global_update``
        is already incremented for the current step, so the first step uses
        the initial lr, as the official code does."""
        frac = min(max(self._global_update - 1, 0) / max(self.lr_decay_steps, 1), 1.0)
        actor_lr = self.actor_lr + (self.lr_final - self.actor_lr) * frac
        critic_lr = self.critic_lr + (self.lr_final - self.critic_lr) * frac
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_lr
        return actor_lr, critic_lr

    def _policy_action(self, state: State) -> torch.Tensor:
        """Imagination policy: actor input detached (PWM ``detach=True``),
        reparameterized sample, tanh, padded dims zeroed."""
        return torch.tanh(self.policy.actor(state["z"].detach())) * self.action_mask

    def _actor_loss(self, z0: torch.Tensor) -> dict[str, torch.Tensor]:
        """Differentiable imagination from ``z0`` (``(B, latent)``). Returns
        the normalized scalar ``loss`` plus the rollout tensors the critic
        update reuses: ``zs`` ``(H+1, B, latent)``, ``rewards`` ``(H, B)``,
        ``values`` ``(H+1, B)`` (min-ensemble, graph-attached) and the
        per-sample unnormalized ``raw_loss`` ``(B,)``."""
        traj = imagine(self.world_model, self._policy_action, {"z": z0}, self.horizon, grad=True)
        zs = traj.states["z"]
        rewards = self.world_model.reward_value({"z": zs[:-1]}, traj.actions, symlog_space=True).squeeze(-1)
        values = self.policy.critic(zs).min(dim=0).values.squeeze(-1)

        discounts = self.gamma ** torch.arange(self.horizon, device=zs.device, dtype=zs.dtype)
        returns = (discounts.unsqueeze(-1) * rewards).sum(dim=0) + self.gamma**self.horizon * values[-1]
        raw_loss = -returns
        if self.use_ret_rms:
            self.policy.ret_rms.update(raw_loss.detach())
            loss = raw_loss / torch.sqrt(self.policy.ret_rms.var + 1e-5)
        else:
            loss = raw_loss / self.horizon
        return {"loss": loss.mean(), "raw_loss": raw_loss, "zs": zs, "rewards": rewards, "values": values}

    def _critic_targets(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Truncated TD(lambda) targets ``(H, B)`` for ``zs[:-1]``:
        ``G_t = r_t + gamma * ((1 - lam) V(z_{t+1}) + lam G_{t+1})``,
        ``G_{H-1} = r_{H-1} + gamma V(z_H)``, never terminating (truncated TD(lambda), unlike the official one-step TD).
        ``lambda_return`` reads ``reward[t+1]`` for row ``t``, hence the
        leading zero row."""
        reward_in = torch.cat([torch.zeros_like(rewards[:1]), rewards], dim=0)
        zeros = torch.zeros_like(values)
        return lambda_return(
            reward_in, values, values, last=zeros, term=zeros, disc=self.gamma, lam=self.lam
        )

    def _update_critic(self, zs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """``critic_iterations`` epochs over ``critic_batches`` chronological
        (unshuffled) minibatches of the flattened ``(H * B)`` rollout. When
        ``critic_batches`` does not divide ``H * B``, the remainder rows form
        one extra short minibatch (official ``drop_last=False``)."""
        latents = zs.reshape(-1, zs.shape[-1])
        flat_targets = targets.reshape(-1, 1)
        minibatch = max(latents.shape[0] // self.critic_batches, 1)
        losses, grad_norm = [], torch.zeros(())
        for _ in range(self.critic_iterations):
            for start in range(0, latents.shape[0], minibatch):
                predictions = self.policy.critic(latents[start : start + minibatch])
                loss = (predictions - flat_targets[start : start + minibatch]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.critic_grad_norm)
                self.critic_optimizer.step()
                losses.append(loss.detach())
        return {"critic_loss": float(torch.stack(losses).mean()), "critic_grad_norm": float(grad_norm)}

    def _update_actor_critic(self, batch: Any, posterior: State) -> dict[str, float]:
        del batch
        actor_lr, critic_lr = self._set_learning_rates()

        rollout = self._actor_loss(posterior["z"])
        self.actor_optimizer.zero_grad(set_to_none=True)
        rollout["loss"].backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.actor_grad_norm)
        if not torch.isfinite(actor_grad_norm):
            raise FloatingPointError("PWM actor gradient is not finite.")
        self.actor_optimizer.step()

        with torch.no_grad():
            zs = rollout["zs"][:-1].detach()
            targets = self._critic_targets(rollout["rewards"].detach(), rollout["values"].detach())
        critic_metrics = self._update_critic(zs, targets)

        return {
            "actor_loss": float(rollout["loss"].detach()),
            "actor_loss_raw": float(rollout["raw_loss"].detach().mean()),
            "actor_grad_norm": float(actor_grad_norm),
            "imagined_reward": float(rollout["rewards"].detach().mean()),
            "value_mean": float(rollout["values"].detach().mean()),
            "critic_target_mean": float(targets.mean()),
            "ret_rms_var": float(self.policy.ret_rms.var),
            "actor_logstd": float(self.policy.actor.logstd.detach().mean()),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            **critic_metrics,
        }

    def _update_targets(self) -> None:
        """No target networks (PWM)."""

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer", "critic_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon": self.horizon,
            "lam": self.lam,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "lr_final": self.lr_final,
            "lr_decay_steps": self.lr_decay_steps,
            "actor_grad_norm": self.actor_grad_norm,
            "critic_grad_norm": self.critic_grad_norm,
            "critic_iterations": self.critic_iterations,
            "critic_batches": self.critic_batches,
            "num_critics": self.num_critics,
            "actor_units": list(self.actor_units),
            "critic_units": list(self.critic_units),
            "init_logstd": self.init_logstd,
            "min_logstd": self.min_logstd,
            "ret_rms": self.use_ret_rms,
        }


class PWM(PWMCore, OfflineModelBasedAlgorithm):
    """Offline single-task PWM on a frozen multitask world model; start
    states from ``MmapMultitaskEpisodeBuffer`` filled with this task's
    episodes only (smaller than the whole dataset, no task filtering at
    sample time). Evaluation (optional ``eval_env``, normally wrapped in
    ``MultitaskPadWrapper``): ``encode(obs) -> actor mean -> tanh -> mask``."""

    _compatible_checkpoint_algorithms = ("PWM",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        world_model_checkpoint: str,
        task_idx: int,
        mmap_dir: str | Path,
        mmap_mode: MmapMode = "create",
        buffer_size: int = 1_000_000,
        batch_size: int = 512,
        horizon: int = 16,
        gamma: float = 0.99,
        lam: float = 0.95,
        actor_lr: float = 5e-4,
        critic_lr: float = 5e-4,
        lr_final: float = 1e-5,
        lr_decay_steps: int = 10_000,
        actor_grad_norm: float = 1.0,
        critic_grad_norm: float = 100.0,
        critic_iterations: int = 8,
        critic_batches: int = 4,
        num_critics: int = 3,
        actor_units: tuple[int, ...] = (400, 200, 100),
        critic_units: tuple[int, ...] = (400, 200),
        init_logstd: float = -1.0,
        min_logstd: float = -1.427,
        ret_rms: bool = True,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 0,
        num_eval_episodes: int = 10,
        num_eval_steps: int = 1_000,
        eval_env: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            buffer_size=buffer_size,
            buffer_device="cpu",  # MmapMultitaskEpisodeBuffer is always CPU-backed.
            batch_size=batch_size,
            gamma=gamma,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            num_eval_episodes=num_eval_episodes,
            eval_env=eval_env,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self.world_model_checkpoint = str(world_model_checkpoint)
        self.task_idx = int(task_idx)
        self.mmap_dir = mmap_dir
        self.mmap_mode = mmap_mode
        self._init_pwm_params(
            horizon=horizon,
            gamma=gamma,
            lam=lam,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            lr_final=lr_final,
            lr_decay_steps=lr_decay_steps,
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            critic_iterations=critic_iterations,
            critic_batches=critic_batches,
            num_critics=num_critics,
            actor_units=actor_units,
            critic_units=critic_units,
            init_logstd=init_logstd,
            min_logstd=min_logstd,
            ret_rms=ret_rms,
        )
        self._setup_model()

    def _setup_model(self) -> None:
        multitask_model, self.world_model_source = load_pwm_world_model(self.world_model_checkpoint)
        warn_on_world_model_rollout_mismatch(self.world_model_source, self.horizon)
        multitask_model.requires_grad_(False)
        self.bound_world_model = BoundTaskWorldModel(multitask_model, self.task_idx).to(self.device).eval()
        task = torch.tensor([self.task_idx], device=self.device)
        self.action_mask = multitask_model.action_mask(task)[0].to(self.device)

        expected_obs = self.env.single_observation_space["state"].shape[-1]
        expected_action = self.env.single_action_space.shape[-1]
        if (multitask_model.obs_dim, multitask_model.action_dim) != (expected_obs, expected_action):
            raise ValueError(
                f"World model padded dims (obs={multitask_model.obs_dim}, "
                f"action={multitask_model.action_dim}) do not match the env spec "
                f"(obs={expected_obs}, action={expected_action})."
            )

        self.policy = PWMPolicy(
            multitask_model.latent_dim,
            multitask_model.action_dim,
            actor_units=self.actor_units,
            critic_units=self.critic_units,
            num_critics=self.num_critics,
            init_logstd=self.init_logstd,
            min_logstd=self.min_logstd,
        ).to(self.device)
        self._build_pwm_optimizers()

        self.replay_buffer = MmapMultitaskEpisodeBuffer(
            obs_dim=multitask_model.obs_dim,
            action_dim=multitask_model.action_dim,
            buffer_size=self.buffer_size,
            horizon=self.horizon,
            mmap_dir=self.mmap_dir,
            mmap_mode=self.mmap_mode,
            storage_device="cpu",
            sample_device=self.device,
        )

    def _update_model(self, batch: Any) -> tuple[dict[str, float], State]:
        """The world model is frozen: only encode the start observations."""
        with torch.no_grad():
            z0 = self.world_model.encode(batch.obs[0])
        return {}, {"z": z0}

    def _eval_action(self, obs) -> torch.Tensor:
        obs = self._obs_to_policy_device(obs)
        state = obs["state"] if isinstance(obs, dict) else obs
        with torch.no_grad():
            z = self.world_model.encode(state.float())
            return torch.tanh(self.policy.actor(z, deterministic=True)) * self.action_mask

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "task_idx": self.task_idx,
            "world_model_checkpoint": self.world_model_checkpoint,
            "world_model_source": self.world_model_source,
        }
