"""JSRL: Jump-Start Reinforcement Learning (Uchendu et al. 2022, arXiv:2204.02372).

A frozen "guide" policy -- loaded from another algorithm's offline-trained
checkpoint (IQL/CalQL/WSRL/AWAC) -- acts for the first ``horizon`` steps of
every episode; afterward this class's own online SAC policy (the
"exploration" policy) acts. ``horizon`` decays over training on a curriculum
driven by a moving average of eval return, mirroring upstream's
``JSRLAfterEvalCallback``/``JSRLPolicy`` (``jumpstart-rl/src/jsrl/jsrl.py``).

Not an off2on algorithm: JSRL owns no offline loss and never updates the
guide -- it consumes a *finished* checkpoint and trains a separate,
from-scratch online policy, structurally identical to how ``DPPO`` consumes
a frozen ``bc_checkpoint`` or ``SUPE`` a frozen ``opal_checkpoint``. Lives in
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

from typing import Any, Literal, Optional

import torch
from gymnasium import spaces

from rl_garden.algorithms.sac import SAC
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.awac_policy import AWACPolicy
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.iql_policy import IQLPolicy
from rl_garden.policies.sac_policy import SACPolicy, WSRL_LOG_STD_MIN

GuideAlgorithm = Literal["iql", "calql", "wsrl", "awac"]

# Keys under these prefixes are provably unused by BasePolicy.predict() for
# every guide policy class below (critic/value heads never run at
# inference) -- a missing key there is tolerated on load. Anything else
# missing is fatal: it means the guide's actor (or, for AWAC, its obs
# normalizer) didn't actually load.
_GUIDE_IGNORABLE_PREFIXES = ("critic.", "critic_target.", "value.")


def _load_guide_policy(
    checkpoint_path: str,
    guide_algorithm: GuideAlgorithm,
    observation_space: spaces.Box,
    action_space: spaces.Box,
    *,
    device: torch.device,
    std_parameterization: Literal["exp", "uniform"] = "exp",
) -> BasePolicy:
    """Reconstruct a frozen guide ``BasePolicy`` from another algorithm's checkpoint.

    State observations only (matches this whole ported-algorithm family's
    scope). Reconstructs the *whole* policy, not just the actor, so AWAC's
    obs-normalizer buffers (``ObsNormalizingMixin``, applied inside
    ``extract_features()``) aren't silently dropped -- an actor-only load
    would run the guide on unnormalized observations. Construction kwargs
    come from the checkpoint's own ``metadata["hyperparameters"]`` where the
    source algorithm records them; critic/value submodules are built with
    best-effort/default shapes since they're provably unused by
    ``predict()`` -- only actor/features_extractor/obs-normalizer key
    mismatches are treated as fatal below.

    ``std_parameterization`` is not recorded in IQL's or AWAC's checkpoint
    metadata today, and it does change the actor's state_dict layout
    (``"exp"``: a ``fc_logstd`` Linear; ``"uniform"``: a raw ``log_stds``
    Parameter -- see ``rl_garden/networks/actor_critic.py``). This repo's
    own IQL AntMaze presets (``configs/off2on/iql_antmaze_*_paper.yaml``)
    use ``"uniform"``, not the class default ``"exp"`` -- pass the value
    matching the guide checkpoint, or the load below fails loudly with a
    clear "missing actor keys" error rather than silently constructing a
    wrong guide. CalQL/WSRL checkpoints already record their own
    ``std_parameterization`` in metadata and ignore this argument.
    """
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    hp = checkpoint["metadata"]["hyperparameters"]
    policy_state = checkpoint["state"]["policy"]
    features_extractor = FlattenExtractor(observation_space=observation_space)

    if guide_algorithm == "iql":
        policy: BasePolicy = IQLPolicy(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=hp.get("net_arch", (256, 256)),
            n_critics=hp.get("n_critics", 2),
            critic_subsample_size=hp.get("critic_subsample_size"),
            actor_distribution=hp.get("actor_distribution", "squashed"),
            std_parameterization=std_parameterization,
        ).to(device)
    elif guide_algorithm == "awac":
        policy = AWACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=hp.get("net_arch", (256, 256, 256)),
            n_critics=hp.get("n_critics", 2),
            std_parameterization=std_parameterization,
        ).to(device)
    elif guide_algorithm in ("calql", "wsrl"):
        # log_std_mode/log_std_min are hard-coded to match CQL's own
        # _setup_model() (rl_garden/algorithms/cql.py:483-485) exactly --
        # not user-configurable there either, so not read from metadata.
        policy = SACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=hp.get("net_arch", (256, 256, 256)),
            n_critics=hp.get("n_critics", 2),
            critic_subsample_size=hp.get("critic_subsample_size"),
            actor_use_layer_norm=hp.get("actor_use_layer_norm", False),
            std_parameterization=hp.get("std_parameterization", std_parameterization),
            log_std_mode="clamp",
            log_std_min=WSRL_LOG_STD_MIN,
        ).to(device)
    else:
        raise ValueError(f"Unknown guide_algorithm {guide_algorithm!r}")

    missing, _unexpected = policy.load_state_dict(policy_state, strict=False)
    fatal_missing = [k for k in missing if not k.startswith(_GUIDE_IGNORABLE_PREFIXES)]
    if fatal_missing:
        raise RuntimeError(
            f"Guide policy ({guide_algorithm!r}) failed to load "
            f"{len(fatal_missing)} required key(s) from {checkpoint_path!r}: "
            f"{fatal_missing[:10]}. This usually means an actor construction "
            "kwarg (e.g. std_parameterization) doesn't match how the guide "
            "was actually trained -- pass the matching "
            "--guide_std_parameterization."
        )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


class JSRL(SAC):
    """SAC exploration policy jump-started by a frozen guide policy."""

    _compatible_checkpoint_algorithms = ("JSRL",)

    def __init__(
        self,
        env: Any,
        guide_checkpoint: str,
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

        self.guide_policy = _load_guide_policy(
            guide_checkpoint,
            guide_algorithm,
            self.env.single_observation_space,
            self.env.single_action_space,
            device=self.device,
            std_parameterization=guide_std_parameterization,
        )

        step = max(1, max_horizon // n_curriculum_stages)
        self._horizons = sorted(set(range(0, max_horizon + 1, step)), reverse=True)
        self._horizon_step = 0
        self._mean_rewards = torch.full((window_size,), float("-inf"))
        self._moving_mean_reward = float("-inf")
        self._best_moving_mean_reward = float("-inf")
        self._tolerated_moving_mean_reward = float("-inf")

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
            guide_actions = self.guide_policy.predict(obs_dev, deterministic=False).detach()
        trainee_actions = self._policy_action(obs)
        return self._blend(guide_actions, trainee_actions, self._guide_mask(self._episode_step))

    # --- training rollout ---

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        super()._post_rollout_step(action_context, terminations, truncations, infos)
        self._episode_step += 1
        self._episode_step[terminations | truncations] = 0

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
        self._eval_episode_step[terminations | truncations] = 0

    def _eval_action(self, obs) -> torch.Tensor:
        obs_dev = self._obs_to_policy_device(obs)
        with torch.no_grad():
            guide_actions = self.guide_policy.predict(obs_dev, deterministic=True)
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
            "max_horizon": self.max_horizon,
            "n_curriculum_stages": self.n_curriculum_stages,
            "tolerance": self.tolerance,
            "window_size": self.window_size,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "jsrl_horizon_step": self._horizon_step,
            "jsrl_mean_rewards": self._mean_rewards,
            "jsrl_moving_mean_reward": self._moving_mean_reward,
            "jsrl_best_moving_mean_reward": self._best_moving_mean_reward,
            "jsrl_tolerated_moving_mean_reward": self._tolerated_moving_mean_reward,
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
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
