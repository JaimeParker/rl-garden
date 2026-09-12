"""Utilities for building RLBench observations/actions and loading RLBench
demo datasets into replay buffers.

RLBench (github.com/stepjam/RLBench) demos are ``List[Demo]``, each ``Demo``
a list of ``Observation`` objects -- the *same* ``Observation`` class the
live env produces, so the flattening/renaming helpers here are the single
source of truth for both ``rl_garden.envs.rlbench.env`` (the live rollout
env) and this loader, avoiding the metadata-mismatch problem
``robomimic_dataset.py`` has to guard against (its live env and its offline
HDF5 are two independently-evolving things; RLBench's aren't).

Two renaming/derivation choices are load-bearing, not stylistic:

- RLBench's own image field names are ``<camera>_rgb``/``<camera>_depth``
  (suffix). The rl-garden observation contract
  (``rl_garden/observations/schema.py``) requires exactly ``rgb_<cam>``/
  ``depth_<cam>`` (prefix) -- every image key here is renamed accordingly.
  This is RLBench's documented producer-specific mapping onto the contract
  (see ``rl_garden/buffers/_dataset_common.py::_finalize_dataset_obs_space``).
- RLBench's own README imitation-learning example derives a ground-truth
  action from a *single* stored ``Observation`` (not a pair):
  ``ground_truth_actions = [obs.joint_velocities for obs in batch]``. For
  transition ``i`` (``obs=demo[i] -> next_obs=demo[i+1]``), the action label
  comes from ``demo[i]`` itself. This only covers the confirmed default
  action mode (``JointVelocity`` arm + ``Discrete`` gripper) -- a different
  action mode would need a different derivation, not implemented here.

Demos carry no explicit reward/success/terminal field on disk. Every
successfully-collected demo is assumed to end in task success (RLBench's own
collection retries any failed attempt until it gets a clean one -- see
``TaskEnvironment._get_live_demos``), so ``reward=1.0``/``done=True`` only at
each demo's last step, ``0.0``/``False`` elsewhere -- the same
sparse-success-at-last-step convention ``ogbench_dataset.py``'s
``masks``/``d4rl_legacy_dataset.py``'s antmaze branch already use. This is an
assumption, not independently re-verified against real demo data.

A single ``path`` (not a separate ``dataset_root``/``task_name`` pair) keeps
this loader's call signature identical to every other loader
(``load_offline_dataset``/``PriorDataReplayMixin.load_offline_replay_buffer``
always call with one positional path): ``path`` is the *task's own* demo
directory (``<dataset_root>/<task_name>``, RLBench's own on-disk layout), so
``dataset_root``/``task_name`` are just ``path.parent``/``path.name``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import (
    _add_flat_transitions,
    _concat,
    _finalize_dataset_obs_space,
    _match_obs_to_buffer,
    _mc_returns,
    _to_tensor,
)
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dataset_backend_registry import (
    DatasetBackend,
    DatasetRequest,
    register_dataset_backend,
)
from rl_garden.observations.schema import ObservationContractError

RLBENCH_CAMERA_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "overhead",
    "wrist",
    "front",
)


def _require_rlbench() -> Any:
    try:
        import rlbench  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without rlbench/pyrep.
        raise ImportError(
            "Using RLBench requires the `rlbench` package (and a working `pyrep` + "
            "CoppeliaSim install -- importing `rlbench` at all pulls in `pyrep`, even "
            "for reading stored demo files with no simulator running). See "
            "docs/guides/rlbench-integration.md for install steps."
        ) from exc
    return rlbench


def build_rlbench_obs_config(*, cameras: tuple[str, ...] = (), image_size: tuple[int, int] = (128, 128)):
    """Build an ``ObservationConfig`` enabling every low-dim field, plus
    rgb+depth (never mask/point_cloud) for ``cameras``. Every camera not in
    ``cameras`` is fully disabled; an empty ``cameras`` is state-only."""
    _require_rlbench()
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)
    enabled = set(cameras)
    for camera_name in RLBENCH_CAMERA_NAMES:
        camera_config = getattr(obs_config, f"{camera_name}_camera")
        if camera_name in enabled:
            camera_config.rgb = True
            camera_config.depth = True
            camera_config.point_cloud = False
            camera_config.mask = False
            camera_config.image_size = image_size
        else:
            camera_config.set_all(False)
    return obs_config


def build_default_rlbench_action_mode():
    """The one confirmed action mode this integration supports:
    ``JointVelocity`` arm + ``Discrete`` gripper -- matches every RLBench
    README example verbatim. See module docstring on why the action-label
    derivation below is scoped to only this combination."""
    _require_rlbench()
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete

    return MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())


def flatten_rlbench_state(obs: Any) -> np.ndarray:
    """Flatten every enabled low-dim field of an ``Observation`` into one
    vector, reusing ``Observation.get_low_dim_data()`` (RLBench's own field
    list/order) rather than re-deriving it."""
    return np.asarray(obs.get_low_dim_data(), dtype=np.float32)


def flatten_rlbench_action(obs: Any) -> np.ndarray:
    """Ground-truth action for the default action mode: joint velocities
    plus a discretized (0/1) gripper command. See module docstring."""
    gripper = np.asarray([round(float(obs.gripper_open))], dtype=np.float32)
    return np.concatenate([np.asarray(obs.joint_velocities, dtype=np.float32), gripper])


def build_rlbench_observation(
    obs: Any, *, cameras: tuple[str, ...] = ()
) -> np.ndarray | dict[str, np.ndarray]:
    """Build the observation this integration exposes to rl-garden: a flat
    ``"state"`` ``Box`` array when ``cameras`` is empty, or a ``Dict``
    (``"state"`` plus ``rgb_<camera>``/``depth_<camera>`` keys per camera)
    otherwise -- matching every other backend's own state-is-flat-Box/
    vision-is-Dict convention (see e.g.
    ``rl_garden/training/offline/bc.py:_bc_kwargs``'s
    ``isinstance(obs_space, spaces.Dict)`` branch)."""
    state = flatten_rlbench_state(obs)
    if not cameras:
        return state
    result: dict[str, np.ndarray] = {"state": state}
    for camera in cameras:
        rgb = np.asarray(getattr(obs, f"{camera}_rgb"), dtype=np.uint8)
        depth = np.asarray(getattr(obs, f"{camera}_depth"), dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[..., None]
        result[f"rgb_{camera}"] = rgb
        result[f"depth_{camera}"] = depth
    return result


def _obs_space_from_observation(value: np.ndarray | dict[str, np.ndarray]) -> spaces.Box | spaces.Dict:
    if isinstance(value, dict):
        entries: dict[str, spaces.Box] = {}
        for key, v in value.items():
            if key.startswith("rgb_"):
                entries[key] = spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
            else:
                # "state" or "depth_<cam>" -- both float32.
                entries[key] = spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=np.float32)
        return spaces.Dict(entries)
    return spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)


def _stack_observations(values: list[np.ndarray | dict[str, np.ndarray]]) -> Any:
    if isinstance(values[0], dict):
        return {key: np.stack([v[key] for v in values]) for key in values[0]}
    return np.stack(values)


def _split_dataset_path(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    return str(path.parent), path.name


def infer_specs_from_rlbench(
    path: str | Path,
    *,
    cameras: tuple[str, ...] = (),
    image_size: tuple[int, int] = (128, 128),
) -> tuple[spaces.Dict, spaces.Box]:
    """Infer the strict-contract Dict obs space (plus action space) from a
    single stored demo. Pure file I/O -- never launches PyRep/CoppeliaSim
    (see module docstring)."""
    rlbench = _require_rlbench()
    dataset_root, task_name = _split_dataset_path(path)
    obs_config = build_rlbench_obs_config(cameras=cameras, image_size=image_size)

    demos = rlbench.utils.get_stored_demos(
        amount=1,
        image_paths=False,
        dataset_root=dataset_root,
        variation_number=0,
        task_name=task_name,
        obs_config=obs_config,
        random_selection=False,
    )
    if not demos or len(demos[0]) == 0:
        raise ValueError(f"No stored demo found for RLBench task at {path!r}.")
    obs = demos[0][0]
    obs_space = _finalize_dataset_obs_space(
        _obs_space_from_observation(build_rlbench_observation(obs, cameras=cameras))
    )
    action_dim = flatten_rlbench_action(obs).shape[0]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return obs_space, action_space


def _load_live_demos(task_name: str, obs_config: Any, amount: int) -> list:
    from rlbench.environment import Environment
    from rlbench.utils import name_to_task_class

    action_mode = build_default_rlbench_action_mode()
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
    env.launch()
    try:
        task_env = env.get_task(name_to_task_class(task_name))
        return task_env.get_demos(amount, live_demos=True)
    finally:
        env.shutdown()


def load_rlbench_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    path: str | Path,
    *,
    num_traj: int | None = None,
    live_demos: bool = False,
    cameras: tuple[str, ...] = (),
    image_size: tuple[int, int] = (128, 128),
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    """Load RLBench demo transitions into an existing replay buffer.

    ``live_demos=True`` drives RLBench's own motion-planning oracle in real
    time instead of reading a stored dataset -- slow, and ``path`` is only
    used for its ``task_name`` half in that case (no ``dataset_root`` needed
    since nothing is read from disk). ``success_key`` is accepted only for
    call-signature parity with every other loader; RLBench demos carry no
    on-disk success field (see module docstring's sparse-reward-at-last-step
    convention), so it is unused.
    """
    del success_key
    rlbench = _require_rlbench()
    dataset_root, task_name = _split_dataset_path(path)
    storage_device = buffer.storage_device
    gamma = float(getattr(buffer, "gamma", 0.99))

    obs_config = build_rlbench_obs_config(cameras=cameras, image_size=image_size)
    if live_demos:
        amount = num_traj if num_traj is not None else 1
        demos = _load_live_demos(task_name, obs_config, amount)
    else:
        amount = num_traj if num_traj is not None else -1
        demos = rlbench.utils.get_stored_demos(
            amount=amount,
            image_paths=False,
            dataset_root=dataset_root,
            variation_number=0,
            task_name=task_name,
            obs_config=obs_config,
            random_selection=False,
        )

    obs_parts: list[Any] = []
    next_obs_parts: list[Any] = []
    action_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    mc_parts: list[torch.Tensor] = []

    for demo in demos:
        length = len(demo) - 1
        if length <= 0:
            continue
        obs_values = [build_rlbench_observation(demo[i], cameras=cameras) for i in range(length)]
        next_obs_values = [
            build_rlbench_observation(demo[i + 1], cameras=cameras) for i in range(length)
        ]
        obs_stack = _stack_observations(obs_values)
        next_obs_stack = _stack_observations(next_obs_values)
        actions = np.stack([flatten_rlbench_action(demo[i]) for i in range(length)])

        rewards = torch.zeros(length, device=storage_device, dtype=torch.float32)
        dones = torch.zeros(length, device=storage_device, dtype=torch.float32)
        rewards[-1] = 1.0
        dones[-1] = 1.0
        if reward_scale != 1.0 or reward_bias != 0.0:
            rewards = rewards * reward_scale + reward_bias

        obs_parts.append(_to_tensor(obs_stack, storage_device))
        next_obs_parts.append(_to_tensor(next_obs_stack, storage_device))
        action_parts.append(_to_tensor(actions, storage_device).float())
        reward_parts.append(rewards)
        done_parts.append(dones)
        if hasattr(buffer, "_mc_table"):
            mc_parts.append(_mc_returns(rewards, dones, gamma))

    if not action_parts:
        raise ValueError(f"No usable RLBench demos found for task {task_name!r} at {path!r}.")

    obs_all = _concat(obs_parts)
    next_obs_all = _concat(next_obs_parts)
    obs_all, next_obs_all = _match_obs_to_buffer(buffer, obs_all, next_obs_all)
    actions_all = torch.cat(action_parts, dim=0)
    rewards_all = torch.cat(reward_parts, dim=0)
    dones_all = torch.cat(done_parts, dim=0)
    mc_returns_all = torch.cat(mc_parts, dim=0) if mc_parts else None
    successes_all = dones_all if hasattr(buffer, "_step_success") else None

    return _add_flat_transitions(
        buffer,
        obs_all,
        next_obs_all,
        actions_all,
        rewards_all,
        dones_all,
        mc_returns_all,
        successes_all,
        episode_ends=dones_all.bool(),
    )


def _resolve_camera_config(req: DatasetRequest) -> tuple[tuple[str, ...], tuple[int, int]]:
    """Shared by infer_specs/load below -- resolves the same cameras/
    image_size the live env would use.

    ``req.observation`` (the ``ObservationConfig``) decides: ``observation.rgb``
    names the cameras (empty means state-only) and ``observation.image_size``
    the resolution. Every training entrypoint that can select
    ``--dataset_backend rlbench`` mixes in ``ObservationArgs``, so
    ``req.observation`` is never ``None`` in practice; a caller that omits it
    anyway gets a clear contract error instead of an ``AttributeError`` on a
    ``backend_config`` that has no camera/image_size fields.
    """
    if req.observation is None:
        raise ObservationContractError(
            "rlbench dataset backend requires DatasetRequest.observation "
            "(an ObservationConfig) -- got None."
        )
    cameras = tuple(sorted(set(req.observation.rgb) | set(req.observation.depth)))
    image_size = req.observation.image_size or (128, 128)
    return cameras, image_size


class RLBenchDatasetBackend(DatasetBackend):
    """Unlike every other registered backend, ``observation`` genuinely
    matters here: RLBench's dataset obs shape (flat state Box vs. a Dict
    with per-camera keys) must match whatever the live env was configured
    with -- see rl_garden.envs.rlbench's own camera handling.
    """

    @classmethod
    def infer_specs(cls, req: DatasetRequest):
        cameras, image_size = _resolve_camera_config(req)
        return infer_specs_from_rlbench(req.path, cameras=cameras, image_size=image_size)

    @classmethod
    def load(cls, buffer, req: DatasetRequest) -> int:
        cameras, image_size = _resolve_camera_config(req)
        return load_rlbench_dataset_to_replay_buffer(
            buffer,
            req.path,
            num_traj=req.num_traj,
            cameras=cameras,
            image_size=image_size,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            success_key=req.success_key,
        )


register_dataset_backend("rlbench", RLBenchDatasetBackend)
