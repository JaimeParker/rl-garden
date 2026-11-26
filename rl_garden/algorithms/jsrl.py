"""JSRL: Jump-Start Reinforcement Learning (Uchendu et al. 2022, arXiv:2204.02372).

A frozen "guide" policy -- loaded from another algorithm's offline-trained
checkpoint (IQL/CalQL/WSRL/AWAC) -- acts for the first ``horizon`` steps of
every episode; afterward this class's own online SAC policy (the
"exploration" policy) acts. ``horizon`` decays over training on a curriculum
driven by a moving average of eval return, mirroring upstream's
``JSRLAfterEvalCallback``/``JSRLPolicy`` (``jumpstart-rl/src/jsrl/jsrl.py``).

Not an off2on algorithm: JSRL owns no offline loss and never updates the
guide -- it consumes a *finished* checkpoint and trains a separate,
from-scratch online policy, structurally identical to how ``SUPE`` consumes
a frozen ``opal_checkpoint``. Lives in
``training/online/``, not ``training/off2on/``.

``eval/*`` reports the guide-assisted mixed rollout (``horizon`` stays
active during eval), matching upstream's own ``JSRLEvalCallback`` and the
WSRL paper's actual JSRL baseline curves -- not a pure-trainee eval.
Upstream's policy has a pure-trainee path
(``not self.training and not self.jsrl_evaluation``) but it is never
exercised during ``learn()``.

Only upstream's ``"curriculum"`` strategy is ported. Upstream's ``"random"``
strategy (resample ``horizon`` every episode end, skip curriculum eval
entirely) is a different mechanism, not needed for the WSRL-paper comparison
this port targets, and is not implemented here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import torch
from gymnasium import spaces

from rl_garden.algorithms.cql import CQLCore
from rl_garden.algorithms.observation_encoders import resolve_observation_encoders
from rl_garden.algorithms.sac import SAC
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.effective_config import (
    reject_unrecorded_import_path,
    resolve_import_path,
)
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups
from rl_garden.policies.awac_policy import AWACPolicy
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.iql_policy import IQLPolicy
from rl_garden.policies.sac_policy import WSRL_LOG_STD_MIN

GuideAlgorithm = Literal["iql", "calql", "wsrl", "awac"]

_logger = logging.getLogger(__name__)

# Keys under these prefixes are provably unused by BasePolicy.predict() for
# every guide policy class below (critic/value heads never run at inference).
# Every other key must match: it belongs to the guide's actor, its encoder or
# (AWAC) its obs normalizer.
_GUIDE_IGNORABLE_PREFIXES = ("critic.", "critic_target.", "critic_extractor.", "value.")

# Actor options the source algorithms record in their checkpoint metadata.
_GUIDE_ACTOR_OPTIONS = (
    "net_arch",
    "actor_use_layer_norm",
    "actor_use_group_norm",
    "num_groups",
    "actor_dropout_rate",
    "kernel_init",
    "backbone_type",
    "std_parameterization",
)
_GUIDE_OBSERVATION_OPTIONS = (
    "encoder_config",
    "obs_groups",
    "critic_encoder_config",
    "encoder_sharing",
)
_GUIDE_ALGORITHM_OPTIONS: dict[str, tuple[str, ...]] = {
    "iql": ("actor_distribution",),
    "awac": (),
    "calql": (
        "actor_factory",
        "actor_feature_dim",
        "policy_log_std_multiplier",
        "policy_log_std_offset",
    ),
}
_GUIDE_ALGORITHM_OPTIONS["wsrl"] = _GUIDE_ALGORITHM_OPTIONS["calql"]


def _recorded_actor_factory(hyperparameters: Mapping[str, Any], source: str):
    """The ``actor_factory`` a CQL-family checkpoint recorded as an import
    path (``None`` when it was trained with the default actor). A factory
    recorded without an import path raises ``ConfigError``."""
    spec = hyperparameters.get("actor_factory")
    if spec is None:
        return None
    reject_unrecorded_import_path("actor_factory", spec)
    _logger.info(
        "importing actor_factory %s from checkpoint metadata (guide checkpoint %s)",
        spec,
        source,
    )
    try:
        return resolve_import_path(spec)
    except (ImportError, AttributeError) as error:
        raise ImportError(
            f"Guide checkpoint {source!r} was trained with actor_factory {spec!r}, "
            f"which cannot be imported here: {error}"
        ) from error


def _infer_actor_options(
    policy_state: Mapping[str, torch.Tensor], *, fallback_std_parameterization: str, source: str
) -> dict[str, Any]:
    """The actor options a guide's weights determine, for a checkpoint whose
    metadata does not record them.

    Only the two that the weights decide: the std parameterization, and
    whether a plain MLP trunk has normalization layers (its 1-D weights).
    LayerNorm and GroupNorm have the same weights, so a norm is taken to be
    LayerNorm (the offline default) with a warning. Any other option at a
    non-default value makes the keys mismatch, which is fatal.
    """
    options: dict[str, Any] = {"std_parameterization": fallback_std_parameterization}
    if "actor.log_stds" in policy_state:
        options["std_parameterization"] = "uniform"
    elif "actor.fc_logstd.weight" in policy_state:
        options["std_parameterization"] = "exp"
    has_norm = any(
        key.startswith("actor.trunk.") and key.endswith(".weight") and value.ndim == 1
        for key, value in policy_state.items()
    )
    options["actor_use_layer_norm"] = has_norm
    if has_norm:
        _logger.warning(
            "guide checkpoint %s has actor normalization layers but does not record their "
            "kind; assuming LayerNorm. A guide trained with group norm would act differently.",
            source,
        )
    return options


def _guide_record(
    checkpoint: Mapping[str, Any],
    guide_algorithm: GuideAlgorithm,
    *,
    std_parameterization: str,
    source: str,
) -> dict[str, Any]:
    """What rebuilds the frozen guide of a source-algorithm ``checkpoint``: the
    options that shape its acting modules, from the metadata where recorded and
    from the weights otherwise, and those modules' weights. A ``JSRL``
    checkpoint embeds it."""
    if guide_algorithm not in _GUIDE_ALGORITHM_OPTIONS:
        raise ValueError(f"Unknown guide_algorithm {guide_algorithm!r}")
    hyperparameters = checkpoint["metadata"]["hyperparameters"]
    policy_state = {
        key: value
        for key, value in checkpoint["state"]["policy"].items()
        if not key.startswith(_GUIDE_IGNORABLE_PREFIXES)
    }
    names = (
        *_GUIDE_ACTOR_OPTIONS,
        *_GUIDE_OBSERVATION_OPTIONS,
        *_GUIDE_ALGORITHM_OPTIONS[guide_algorithm],
    )
    recorded = {name: hyperparameters[name] for name in names if name in hyperparameters}
    if not set(_GUIDE_ACTOR_OPTIONS) <= recorded.keys() and recorded.get("actor_factory") is None:
        recorded = {
            **_infer_actor_options(
                policy_state,
                fallback_std_parameterization=std_parameterization,
                source=source,
            ),
            **recorded,
        }
    return {
        "source": source,
        "algorithm": guide_algorithm,
        "hyperparameters": recorded,
        "policy": policy_state,
    }


def _guide_extractors(
    hyperparameters: Mapping[str, Any], observation_space: spaces.Dict
) -> dict[str, Any]:
    def encoder_config(name: str) -> Optional[EncoderConfig]:
        values = hyperparameters.get(name)
        # The guide's own weights are loaded next.
        return None if values is None else EncoderConfig(**{**values, "pretrained_weights": None})

    obs_groups = hyperparameters.get("obs_groups")
    sharing = hyperparameters.get("encoder_sharing") or "shared_critic_grad"
    encoders = resolve_observation_encoders(
        observation_space,
        encoder_config("encoder_config"),
        None if obs_groups is None else ObsGroups(**obs_groups),
        sharing,
        critic_encoder_config=encoder_config("critic_encoder_config"),
    )
    return {
        "actor_extractor": encoders.actor,
        "critic_extractor": encoders.critic,
        "encoder_sharing": sharing,
    }


def _build_guide_policy(
    record: Mapping[str, Any],
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    *,
    device: torch.device,
) -> BasePolicy:
    """The frozen guide of a ``_guide_record``: the whole policy rather than
    its actor alone, so AWAC's obs-normalizer buffers (``ObsNormalizingMixin``,
    applied inside ``extract_actor_features()``) are kept. Its critic/value
    heads, which ``predict()`` never runs, are default-shaped and untrained."""
    algorithm = record["algorithm"]
    hp = record["hyperparameters"]
    kwargs: dict[str, Any] = {
        "observation_space": observation_space,
        "action_space": action_space,
        **_guide_extractors(hp, observation_space),
        **{name: hp[name] for name in _GUIDE_ACTOR_OPTIONS if name in hp},
    }
    if algorithm == "iql":
        policy: BasePolicy = IQLPolicy(
            **kwargs, actor_distribution=hp.get("actor_distribution", "squashed")
        )
    elif algorithm == "awac":
        policy = AWACPolicy(**kwargs)
    else:
        # log_std_mode/log_std_min are hard-coded to match CQLCore's own
        # _policy_kwargs() exactly -- not user-configurable there either, so
        # not read from metadata. Built as CQLCore.policy_class (the class
        # CalQL/WSRL build), with the checkpoint's recorded actor_factory.
        policy = CQLCore.policy_class(
            **kwargs,
            log_std_mode="clamp",
            log_std_min=WSRL_LOG_STD_MIN,
            log_std_multiplier_init=hp.get("policy_log_std_multiplier"),
            log_std_offset_init=hp.get("policy_log_std_offset"),
            actor_feature_dim=hp.get("actor_feature_dim"),
            actor_factory=_recorded_actor_factory(hp, record["source"]),
        )
    missing, unexpected = policy.load_state_dict(record["policy"], strict=False)
    missing = [key for key in missing if not key.startswith(_GUIDE_IGNORABLE_PREFIXES)]
    if missing or unexpected:
        raise RuntimeError(
            f"Guide policy ({algorithm!r}) from {record['source']!r} failed to load: "
            f"{len(missing)} required key(s) missing {missing[:10]}, "
            f"{len(unexpected)} key(s) unexpected {unexpected[:10]}. This usually means an "
            "actor construction option (e.g. std_parameterization) doesn't match how the "
            "guide was actually trained -- pass the matching --guide_std_parameterization, "
            "or check --guide_algorithm and the observation keys."
        )
    policy.to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def require_guide_checkpoint(checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Guide checkpoint {checkpoint_path!r} does not exist. JSRL needs it to train from "
            "scratch, and to load a JSRL checkpoint that does not embed its guide."
        )


def _read_guide_record(
    checkpoint_path: str,
    guide_algorithm: GuideAlgorithm,
    *,
    std_parameterization: str,
) -> dict[str, Any]:
    require_guide_checkpoint(checkpoint_path)
    return _guide_record(
        load_checkpoint_file(checkpoint_path, map_location="cpu"),
        guide_algorithm,
        std_parameterization=std_parameterization,
        source=str(checkpoint_path),
    )


def _load_guide_policy(
    checkpoint_path: str,
    guide_algorithm: GuideAlgorithm,
    observation_space: spaces.Dict,
    action_space: spaces.Box,
    *,
    device: torch.device,
    std_parameterization: Literal["exp", "uniform"] = "exp",
) -> BasePolicy:
    """Reconstruct a frozen guide ``BasePolicy`` from another algorithm's
    checkpoint file. ``std_parameterization`` is used only when neither the
    checkpoint's metadata nor its weights determine it."""
    return _build_guide_policy(
        _read_guide_record(
            checkpoint_path, guide_algorithm, std_parameterization=std_parameterization
        ),
        observation_space,
        action_space,
        device=device,
    )


def embedded_guide(checkpoint: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The ``_guide_record`` in a ``JSRL`` checkpoint, ``None`` without one."""
    if checkpoint is None:
        return None
    return checkpoint["state"].get("extra", {}).get("guide")


class JSRL(SAC):
    """SAC exploration policy jump-started by a frozen guide policy.

    The guide is read from ``guide_checkpoint``. A JSRL checkpoint embeds it,
    and loading one restores it in place of any guide already held; an agent
    built without ``guide_checkpoint`` gets its guide that way and cannot act
    before."""

    _compatible_checkpoint_algorithms = ("JSRL",)

    def __init__(
        self,
        env: Any,
        guide_checkpoint: Optional[str],
        guide_algorithm: GuideAlgorithm,
        max_horizon: int,
        eval_env: Optional[Any] = None,
        n_curriculum_stages: int = 10,
        tolerance: float = 0.0,
        window_size: int = 1,
        guide_std_parameterization: Literal["exp", "uniform"] = "exp",
        **sac_kwargs: Any,
    ) -> None:
        self.guide_checkpoint = guide_checkpoint
        self.guide_algorithm = guide_algorithm
        self.max_horizon = max_horizon
        self.n_curriculum_stages = n_curriculum_stages
        self.tolerance = tolerance
        self.window_size = window_size
        self.guide_std_parameterization = guide_std_parameterization
        super().__init__(env, eval_env, **sac_kwargs)

        self.guide_policy: Optional[BasePolicy] = None
        if guide_checkpoint:
            self._set_guide(
                _read_guide_record(
                    guide_checkpoint,
                    guide_algorithm,
                    std_parameterization=guide_std_parameterization,
                )
            )

        step = max(1, max_horizon // n_curriculum_stages)
        self._horizons = sorted(set(range(0, max_horizon + 1, step)), reverse=True)
        self._horizon_step = 0
        self._mean_rewards = torch.full((window_size,), float("-inf"))
        self._moving_mean_reward = float("-inf")
        self._best_moving_mean_reward = float("-inf")
        self._tolerated_moving_mean_reward = float("-inf")

    def _set_guide(self, record: Mapping[str, Any]) -> None:
        self.guide_policy = _build_guide_policy(
            record,
            self.env.single_observation_space,
            self.env.single_action_space,
            device=self.device,
        )
        self._guide_record = dict(record)
        self.guide_checkpoint = record["source"]
        self.guide_algorithm = record["algorithm"]

    def _guide_action(self, obs, *, deterministic: bool) -> torch.Tensor:
        if self.guide_policy is None:
            raise RuntimeError(
                "JSRL has no guide policy: pass guide_checkpoint, or load a JSRL checkpoint "
                "that embeds its guide."
            )
        return self.guide_policy.predict(obs, deterministic=deterministic)

    # --- curriculum ---

    @property
    def horizon(self) -> int:
        return self._horizons[self._horizon_step]

    def _guide_mask(self, episode_step: torch.Tensor) -> torch.Tensor:
        return episode_step <= self.horizon

    def _update_jsrl_curriculum(self, mean_return: float) -> None:
        self._mean_rewards = torch.roll(self._mean_rewards, 1)
        self._mean_rewards[0] = mean_return
        self._moving_mean_reward = float(self._mean_rewards.mean().item())
        if self._mean_rewards[-1].item() == float("-inf") or self.horizon <= 0:
            return
        if self._best_moving_mean_reward == float("-inf"):
            self._best_moving_mean_reward = self._moving_mean_reward
        elif self._moving_mean_reward >= self._tolerated_moving_mean_reward:
            self._horizon_step = min(self._horizon_step + 1, len(self._horizons) - 1)
        if self._moving_mean_reward >= self._best_moving_mean_reward:
            self._tolerated_moving_mean_reward = (
                self._moving_mean_reward - self.tolerance * abs(self._moving_mean_reward)
            )
            self._best_moving_mean_reward = max(
                self._best_moving_mean_reward, self._moving_mean_reward
            )

    # --- guide/trainee blending ---

    def _blend(
        self,
        guide_actions: torch.Tensor,
        trainee_actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_shape = (mask.shape[0],) + (1,) * (trainee_actions.ndim - 1)
        return torch.where(mask.view(mask_shape), guide_actions, trainee_actions)

    def _jsrl_action(self, obs) -> torch.Tensor:
        obs_dev = self._obs_to_policy_device(obs)
        with torch.no_grad():
            guide_actions = self._guide_action(obs_dev, deterministic=False).detach()
        trainee_actions = self._policy_action(obs)
        return self._blend(guide_actions, trainee_actions, self._guide_mask(self._episode_step))

    # --- training rollout ---

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        super()._post_rollout_step(action_context, terminations, truncations, infos)
        self._episode_step += 1
        self._episode_step[(terminations | truncations).to(self.device)] = 0

    def _rollout_action(self, obs, learning_has_started: bool):
        # Mirrors OffPolicyAlgorithm._rollout_action exactly, replacing its
        # two _policy_action(obs) call sites with _jsrl_action(obs) -- the
        # only points where the trainee (rather than random exploration)
        # would otherwise act. Not calling super() mid-method since there's
        # no hook there for one consumer to plug into (parent-hook bar:
        # needs >=2 existing subclasses).
        phase = self._active_initial_training_phase()
        if phase is not None:
            actions = self._jsrl_action(obs)
            if phase.random_action_prob > 0.0:
                random_actions = self._explore_action(obs)
                mask_shape = (actions.shape[0],) + (1,) * (actions.ndim - 1)
                random_mask = (
                    torch.rand(mask_shape, device=actions.device)
                    < phase.random_action_prob
                )
                actions = torch.where(random_mask, random_actions, actions)
            return actions, actions, None
        if not learning_has_started:
            # Pre-learning_starts random warmup wins over the guide -- the
            # same collision exists in upstream's own SB3 _sample_action
            # warmup path, preserved rather than "fixed".
            actions = self._explore_action(obs)
        else:
            actions = self._jsrl_action(obs)
        return actions, actions, None

    # --- eval rollout ---

    def _eval_start_hook(self) -> None:
        super()._eval_start_hook()
        self._eval_episode_step = torch.zeros(
            self.eval_env.num_envs, dtype=torch.long, device=self.device
        )

    def _eval_step_hook(self, obs_before, critic_action, rewards, terminations, truncations, infos) -> None:
        super()._eval_step_hook(obs_before, critic_action, rewards, terminations, truncations, infos)
        self._eval_episode_step += 1
        self._eval_episode_step[(terminations | truncations).to(self.device)] = 0

    def _eval_action(self, obs) -> torch.Tensor:
        obs_dev = self._obs_to_policy_device(obs)
        with torch.no_grad():
            guide_actions = self._guide_action(obs_dev, deterministic=True)
            trainee_actions = self.policy.predict(obs_dev, deterministic=True)
        return self._blend(guide_actions, trainee_actions, self._guide_mask(self._eval_episode_step))

    def _eval_action_and_critic_action(self, obs):
        action = self._eval_action(obs)
        return action, action

    def _evaluate(self) -> dict[str, float]:
        metrics = super()._evaluate()
        if self.eval_env is None:
            return metrics
        self._update_jsrl_curriculum(metrics.get("return", float("nan")))
        metrics["jsrl_horizon"] = float(self.horizon)
        metrics["jsrl_moving_mean_reward"] = self._moving_mean_reward
        metrics["jsrl_best_moving_mean_reward"] = self._best_moving_mean_reward
        metrics["jsrl_tolerated_moving_mean_reward"] = self._tolerated_moving_mean_reward
        return metrics

    # --- checkpointing ---

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "guide_checkpoint": self.guide_checkpoint,
            "guide_algorithm": self.guide_algorithm,
            "guide_std_parameterization": self.guide_std_parameterization,
            "max_horizon": self.max_horizon,
            "n_curriculum_stages": self.n_curriculum_stages,
            "tolerance": self.tolerance,
            "window_size": self.window_size,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        guide = {} if self.guide_policy is None else {"guide": self._guide_record}
        return {
            **super()._extra_checkpoint_state(),
            **guide,
            "jsrl_horizon_step": self._horizon_step,
            "jsrl_mean_rewards": self._mean_rewards,
            "jsrl_moving_mean_reward": self._moving_mean_reward,
            "jsrl_best_moving_mean_reward": self._best_moving_mean_reward,
            "jsrl_tolerated_moving_mean_reward": self._tolerated_moving_mean_reward,
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        if "guide" in state:
            self._set_guide(state["guide"])
        elif self.guide_policy is None:
            raise ValueError(
                "The JSRL checkpoint embeds no guide policy and the agent has none; "
                "pass guide_checkpoint."
            )
        self._horizon_step = state.get("jsrl_horizon_step", 0)
        mean_rewards = state.get("jsrl_mean_rewards")
        if mean_rewards is not None:
            self._mean_rewards = mean_rewards.to(self._mean_rewards.dtype)
        self._moving_mean_reward = state.get("jsrl_moving_mean_reward", float("-inf"))
        self._best_moving_mean_reward = state.get(
            "jsrl_best_moving_mean_reward", float("-inf")
        )
        self._tolerated_moving_mean_reward = state.get(
            "jsrl_tolerated_moving_mean_reward", float("-inf")
        )
