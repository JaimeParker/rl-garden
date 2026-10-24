"""IsaacLab environment factory.

Wraps a raw IsaacLab env (``ManagerBasedRLEnv`` or ``DirectRLEnv``, native
``isaaclab_tasks`` task or an ``RLGardenDirectRLEnv``-scaffold task) with
``_IsaacLabVecEnvAdapter`` so it exposes the same shape the rest of
``rl_garden`` targets: ``num_envs`` / ``single_observation_space`` /
``single_action_space`` / batched ``action_space``, and ``step``/``reset``
returning GPU tensors plus the standard Gymnasium ``(obs, reward,
terminated, truncated, info)`` tuple.
The raw env comes from ``v3.runtime`` (Isaac Lab 3.0) or ``v2.runtime``
(2.3.x), chosen by ``detect.generation()``; everything here is shared.

Observation extraction:
- state-only (``ObservationConfig`` requests no cameras): looks for a
  ``"state"`` key (rl-garden scaffold convention) first, falling back to
  ``"policy"`` (every native IsaacLab task's own convention -- both
  ``ManagerBasedRLEnv`` and ``DirectRLEnv`` tasks use it).
- vision requested: requires rl-garden's own cross-backend key convention
  directly (``"rgb_<cam>"``/``"depth_<cam>"``, plus ``"state"`` if the task
  returns one and it's kept) -- only ``RLGardenDirectRLEnv``-scaffold tasks
  are expected to satisfy this; native image tasks (e.g.
  ``Isaac-Cartpole-Camera-Direct``, which returns a stacked channel-first
  ``"policy"`` image plus a ``"critic"`` state) aren't supported by this backend.
  Camera set/resolution are baked into the task registration, not
  runtime-selectable through ``ObservationConfig.rgb``/``.image_size``.

Observation spaces: a task that declares ``cfg.observation_space`` as a
``gym.spaces.Dict`` keyed by the keys it returns gets that declaration
validated against its first observation (same key set; every entry a
``Box`` with the tensor's per-env shape and dtype; ``ObservationContractError``
otherwise) and used as the adapter's per-key space, so its bounds (e.g. a
``[0, 1]`` depth range) reach the encoders. Isaac Lab itself never checks
the declaration. Without a declared ``Dict`` (int placeholders, native
tasks) the spaces are inferred from the runtime tensors.

Episode statistics: Isaac Lab auto-resets inside ``step()`` and never
writes ``final_info``, so the adapter accumulates each env's return and
length itself (``EpisodeStatistics``) and sets ``info["final_info"] =
{"episode": {"return", "episode_len", ...}}`` and ``info["_final_info"] =
terminated | truncated`` on every step; consumers read only the done rows.
``info`` is a shallow copy of the env's ``extras`` (Isaac Lab reuses one
dict across steps, which is never mutated here). Two task-side
conventions: per-env metrics in ``extras["episode_metrics"]`` (a dict of
``(num_envs,)`` tensors, e.g. ``{"success": ...}``) are merged into
``final_info["episode"]`` and removed from ``info``; scalar aggregates in
``extras["log"]`` are forwarded as ``info["log"]`` only when ``log`` is a
new dict object (Isaac Lab leaves the previous one in place on steps where
no env resets, so an unchanged object is stale and dropped).
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from rl_garden.envs.episode_statistics import EpisodeStatistics
from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig
from rl_garden.envs.isaaclab.detect import generation
from rl_garden.observations.schema import ObservationContractError


def _box_from_tensor(tensor: torch.Tensor) -> gym.spaces.Box:
    if tensor.dtype == torch.uint8:
        return gym.spaces.Box(low=0, high=255, shape=tuple(tensor.shape[1:]), dtype=np.uint8)
    return gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=tuple(tensor.shape[1:]),
        dtype=np.float32,
    )


def _declared_observation_spaces(unwrapped: Any) -> dict | None:
    """Per-key spaces a task declared as a ``gym.spaces.Dict``
    ``cfg.observation_space`` (published by ``DirectRLEnv`` under the
    ``"policy"`` group), or ``None`` when it declared none (int placeholder
    -> ``Box``)."""
    groups = getattr(unwrapped, "single_observation_space", None)
    if not isinstance(groups, gym.spaces.Dict) or "policy" not in groups.spaces:
        return None
    policy = groups["policy"]
    return dict(policy.spaces) if isinstance(policy, gym.spaces.Dict) else None


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _tensor_spec(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"per-env shape={tuple(value.shape[1:])} dtype={_dtype_name(value)}"
    return type(value).__name__


def _validate_declared_spaces(declared: dict, raw_obs: dict) -> None:
    emitted = {key: _tensor_spec(value) for key, value in raw_obs.items()}
    undeclared = sorted(set(raw_obs) - set(declared))
    not_emitted = sorted(set(declared) - set(raw_obs))
    if undeclared or not_emitted:
        raise ObservationContractError(
            "IsaacLab task's declared cfg.observation_space keys differ from its observation keys: "
            f"emitted but not declared {undeclared!r}, declared but not emitted {not_emitted!r} "
            f"(declared {declared!r}, emitted {emitted!r})."
        )
    for key, space in declared.items():
        value = raw_obs[key]
        if not isinstance(space, gym.spaces.Box):
            raise ObservationContractError(
                f"IsaacLab task's declared cfg.observation_space[{key!r}] must be a gym.spaces.Box, "
                f"got {space!r} (emitted {emitted[key]})."
            )
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape[1:]) != space.shape
            or _dtype_name(value) != space.dtype.name
        ):
            raise ObservationContractError(
                f"IsaacLab task's observation {key!r} does not match its declared space: "
                f"declared {space!r}, emitted {emitted[key]}."
            )


def _check_frames_nonzero(obs: dict) -> None:
    for key, tensor in obs.items():
        if key.startswith(("rgb_", "depth_")) and tensor.max() == 0:
            raise RuntimeError(
                f"IsaacLab camera renderer produced only zero frames for key {key!r}. On Kit RTX this "
                "means the renderer was not created (e.g. driver rejected; see the Kit log). Use a "
                "driver Kit RTX accepts, or on Isaac Lab 3.0 --isaaclab.renderer newton_renderer."
            )


class _IsaacLabVecEnvAdapter(gym.Env):
    """Adapts an IsaacLab env to the rl-garden env shape (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(
        self, env: Any, seed: int, *, is_visual: bool = False, state: bool = True
    ) -> None:
        self._env = env
        self._seed = seed
        self.is_visual = is_visual
        self.state = state
        # gym.make() wraps the raw env in OrderEnforcing, which rejects
        # attribute access before the first reset() -- go through .unwrapped
        # for env-level attributes that aren't part of the standard gym.Env
        # surface (num_envs, device).
        unwrapped = env.unwrapped
        self.num_envs = unwrapped.num_envs
        self.device = unwrapped.device
        self._episode_stats = EpisodeStatistics(self.num_envs, self.device)
        # Read once, before anything else: Isaac Lab 3.0's env.close() sets
        # single_observation_space to None.
        declared = _declared_observation_spaces(unwrapped)

        obs_dict, reset_info = env.reset(seed=seed)
        # The reset-time `log` is not a training aggregate; never forward it.
        self._last_forwarded_log = reset_info.get("log")
        if declared is not None:
            _validate_declared_spaces(declared, obs_dict)
        self._last_obs = self._extract_obs(obs_dict)
        self._declared_spaces = self._declared_output_spaces(declared, obs_dict)
        _check_frames_nonzero(self._last_obs)
        # Duck-typed attribute expected by wrappers like ImageFrameStackWrapper
        # on env.unwrapped (mirrors mani_skill.envs.sapien_env.BaseEnv).
        self._init_raw_obs = self._last_obs
        self.single_observation_space = self._build_observation_space(self._last_obs)

        # IsaacLab's action_space is already batched as (num_envs, action_dim),
        # unlike Gymnasium's vector-env convention where single_action_space
        # describes one env -- slice off the leading num_envs dimension.
        action_space = unwrapped.action_space
        self.single_action_space = gym.spaces.Box(
            low=action_space.low[0],
            high=action_space.high[0],
            shape=action_space.shape[1:],
            dtype=action_space.dtype,
        )
        # off_policy.py's random-exploration phase reads the batched
        # action_space.shape directly (not single_action_space).
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _extract_obs(self, obs_dict: dict) -> dict:
        # TODO(policy-distillation): this collapses to a single obs tensor
        # even when obs_dict also has IsaacLab's native "critic" (privileged)
        # group -- expose it as an additional Dict key here (alongside
        # "state"/"policy") so PolicyDistillation(teacher_obs_keys=[...])
        # can consume it without a bespoke env wrapper. See
        # rl_garden/algorithms/policy_distillation.py.
        if not self.is_visual:
            if "state" in obs_dict:
                return {"state": obs_dict["state"]}
            if "policy" in obs_dict:
                return {"state": obs_dict["policy"]}
            raise ValueError(
                "IsaacLab backend (state-only observation) requires a 'state' or "
                f"'policy' observation key, got keys {sorted(obs_dict)!r}."
            )
        image_keys = [
            key for key in obs_dict if key.startswith(("rgb_", "depth_")) and key != "state"
        ]
        if not image_keys:
            raise ValueError(
                "IsaacLab backend (vision observation) requires "
                "'rgb_<cam>'/'depth_<cam>'-prefixed observation keys (rl-garden's own "
                "cross-backend contract -- see RLGardenDirectRLEnv's "
                f"docstring), got keys {sorted(obs_dict)!r}."
            )
        out = {key: obs_dict[key] for key in image_keys}
        if self.state and "state" in obs_dict:
            out["state"] = obs_dict["state"]
        return out

    def _declared_output_spaces(self, declared: dict | None, raw_obs: dict) -> dict:
        """Declared ``Box`` per output key of ``_extract_obs`` (empty when the
        task declared no ``Dict``); state-only mode renames the raw
        ``"state"``/``"policy"`` key to ``"state"``."""
        if declared is None:
            return {}
        if self.is_visual:
            return declared
        return {"state": declared["state" if "state" in raw_obs else "policy"]}

    def _build_observation_space(self, obs: dict):
        return gym.spaces.Dict(
            {key: self._observation_key_space(key, value) for key, value in obs.items()}
        )

    def _observation_key_space(self, key: str, tensor: torch.Tensor) -> gym.spaces.Box:
        declared = self._declared_spaces.get(key)
        if declared is None:
            return _box_from_tensor(tensor)
        shape = tuple(tensor.shape[1:])
        if shape == declared.shape:
            return declared
        if shape[1:] == declared.shape:
            # ImageFrameStackWrapper's (T, *declared.shape) stacked frames.
            return gym.spaces.Box(
                low=np.broadcast_to(declared.low, shape),
                high=np.broadcast_to(declared.high, shape),
                shape=shape,
                dtype=declared.dtype,
            )
        raise ObservationContractError(
            f"IsaacLab observation {key!r} has per-env shape {shape}, which is neither its declared "
            f"space {declared!r} nor that space with a leading frame-stack dim."
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_dict, info = self._env.reset(seed=seed if seed is not None else self._seed)
        self._last_obs = self._extract_obs(obs_dict)
        self._episode_stats.reset()
        self._last_forwarded_log = info.get("log")
        return self._last_obs, info

    def step(self, actions: torch.Tensor):
        obs_dict, reward, terminated, truncated, extras = self._env.step(actions)
        self._last_obs = self._extract_obs(obs_dict)
        info = self._step_info(extras, reward, terminated | truncated)
        return self._last_obs, reward, terminated, truncated, info

    def _step_info(self, extras: dict, reward: torch.Tensor, done: torch.Tensor) -> dict:
        """``extras`` plus ``final_info``/``_final_info`` (see module docstring)."""
        info = dict(extras)
        episode = self._episode_stats.step(reward, done)
        for key, value in info.pop("episode_metrics", {}).items():
            if key in episode:
                raise ValueError(
                    f"IsaacLab task's extras['episode_metrics'][{key!r}] collides with the "
                    f"adapter-computed episode statistic {key!r}; rename the task metric."
                )
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (self.num_envs,):
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(
                    f"IsaacLab task's extras['episode_metrics'][{key!r}] must be a "
                    f"({self.num_envs},) tensor, got {shape}."
                )
            episode[key] = value
        info["final_info"] = {"episode": episode}
        info["_final_info"] = done
        log = info.pop("log", None)
        if log is not None and log is not self._last_forwarded_log:
            info["log"] = log
            self._last_forwarded_log = log
        return info

    def update_obs_space(self, obs) -> None:
        """Duck-typed hook expected on ``env.unwrapped`` by wrappers like
        ``ImageFrameStackWrapper`` (mirrors
        ``mani_skill.envs.sapien_env.BaseEnv.update_obs_space``)."""
        self._init_raw_obs = obs
        self.single_observation_space = self._build_observation_space(obs)

    def post_update_sync(self) -> None:
        """Duck-typed hook ``OnPolicyAlgorithm.learn()`` calls once per
        rollout+update cycle, right after the update phase and before the
        next rollout's first ``step()``.

        Mitigation (not a confirmed full fix) for a render-pipeline stall
        observed on 6017-nofwd: Kit's camera-rendering pipeline can hang
        indefinitely on a render call that follows a sustained PyTorch CUDA
        compute burst (e.g. a PPO update's many backward passes). A single
        ``torch.cuda.synchronize()`` here measurably reduces how often this
        happens (state-only tasks never need it; observed several camera-
        task runs get through many more rollout+update cycles with this than
        without) but did NOT reliably eliminate the stall across repeated
        runs in testing -- it looks like a timing-sensitive race rather than
        something a single sync deterministically resolves. Syncing before
        every ``step()`` instead of once per cycle made it worse (reintroduced
        the hang reliably), so keep this placement (once per cycle) if
        investigating further.
        """
        if self.is_visual:
            torch.cuda.synchronize()

    def close(self) -> None:
        self._env.close()


def make_isaaclab_env(cfg: IsaacLabEnvConfig):
    """Build an IsaacLab env according to ``cfg``.

    Detects the installed Isaac Lab generation and lets that package's
    ``runtime.build`` create the raw env in its own launch/import order
    (``v3``: parse -> launch_simulation -> make; ``v2``: AppLauncher ->
    import -> parse -> make).
    """
    if generation() == "v3":
        from rl_garden.envs.isaaclab.v3.runtime import build
    else:
        from rl_garden.envs.isaaclab.v2.runtime import build

    env = build(cfg)
    adapter = _IsaacLabVecEnvAdapter(
        env, seed=cfg.seed, is_visual=cfg.is_visual, state=cfg.state
    )

    if cfg.frame_stack > 1:
        if not cfg.is_visual:
            raise ObservationContractError("frame_stack > 1 requires a visual observation mode")
        from rl_garden.envs.wrappers import ImageFrameStackWrapper

        wrapped = ImageFrameStackWrapper(
            adapter, frame_stack=cfg.frame_stack, autoreset_in_step=True
        )
        # ImageFrameStackWrapper doesn't forward arbitrary attributes -- copy
        # the rl-garden env-shape attributes onto the outermost object.
        # single_observation_space is already the post-frame-stack space at
        # this point (ImageFrameStackWrapper.__init__ calls update_obs_space
        # on the adapter before returning).
        wrapped.num_envs = adapter.num_envs
        wrapped.device = adapter.device
        wrapped.single_action_space = adapter.single_action_space
        wrapped.single_observation_space = adapter.single_observation_space
        wrapped.post_update_sync = adapter.post_update_sync
        return wrapped

    return adapter
