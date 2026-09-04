"""Live scripted-policy demo generation for Meta-World.

Unlike every other dataset backend, Meta-World has no on-disk demo dataset
of its own to read (see ``docs/guides/metaworld-integration.md`` -- the
recommended path for expert demos is the official Farama-hosted
``metaworld/<task>/expert-v0`` Minari datasets, loaded via the existing
``minari`` backend, no code here). This module is for generating *fresh*
demos at load time, using the expert scripted policy Meta-World ships for
every task (``metaworld.policies.ENV_POLICY_MAP``).

Meta-World's own scripted-policy test
(``3rd_party/Metaworld/tests/metaworld/envs/mujoco/sawyer_xyz/
test_scripted_policies.py``) only expects an ~80% success rate across all
50 tasks, not 100% -- so demo collection here retries failed episodes and
keeps only successful ones, the same "every collected demo is assumed to
end in task success" convention ``rlbench_dataset.py`` uses for RLBench's
live-demo path: reward=1.0/done=1.0 only at each demo's last step, 0.0/False
elsewhere.

``DatasetRequest.path`` holds the *task name* (e.g. ``"pick-place-v3"``),
not a filesystem path -- the same convention ``minari``'s ``dataset_id``/
``d4rl_legacy``'s ``env_id`` already use for this field.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import _add_flat_transitions, _concat, _mc_returns, _to_tensor
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dataset_backend_registry import (
    DatasetBackend,
    DatasetRequest,
    register_dataset_backend,
)


def _require_metaworld() -> Any:
    try:
        import metaworld  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without metaworld installed.
        raise ImportError(
            "Using the `metaworld` dataset backend requires the `metaworld` package. "
            "See docs/guides/metaworld-integration.md for install steps."
        ) from exc
    return metaworld


def infer_specs_from_metaworld(task_name: str) -> tuple[spaces.Box, spaces.Box]:
    """Read obs/action spaces directly from a throwaway single-task env --
    Meta-World's spaces are static per-task metadata, not derived from any
    on-disk data, so no episode rollout is needed."""
    _require_metaworld()
    import gymnasium as gym

    env = gym.make("Meta-World/MT1", env_name=task_name)
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()


def _rollout_scripted_demo(task_name: str, seed: int) -> list[dict[str, Any]] | None:
    """One episode using the task's expert scripted policy. Returns the list
    of per-step transitions, or ``None`` if the episode never reaches
    ``info["success"] == 1`` (caller retries with a fresh seed)."""
    _require_metaworld()
    import gymnasium as gym
    from metaworld.policies import ENV_POLICY_MAP

    env = gym.make("Meta-World/MT1", env_name=task_name, seed=seed, terminate_on_success=True)
    try:
        policy = ENV_POLICY_MAP[task_name]()
        obs, _info = env.reset(seed=seed)
        transitions: list[dict[str, Any]] = []
        succeeded = False
        done = False
        while not done:
            action = np.asarray(policy.get_action(obs), dtype=np.float32)
            next_obs, _reward, terminated, truncated, info = env.step(action)
            transitions.append({"obs": obs, "action": action, "next_obs": next_obs})
            if bool(info.get("success", 0)):
                succeeded = True
            done = terminated or truncated
            obs = next_obs
        return transitions if succeeded else None
    finally:
        env.close()


def load_metaworld_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    task_name: str,
    *,
    num_traj: int | None = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    """Generate ``num_traj`` successful scripted-policy demos for
    ``task_name`` and load them into ``buffer``.

    ``success_key`` is accepted only for call-signature parity with every
    other loader; Meta-World's success signal is always ``info["success"]``
    (see module docstring), so it is unused.
    """
    del success_key
    amount = num_traj if num_traj is not None else 1
    storage_device = buffer.storage_device
    gamma = float(getattr(buffer, "gamma", 0.99))

    obs_parts: list[torch.Tensor] = []
    next_obs_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    mc_parts: list[torch.Tensor] = []

    seed = 0
    collected = 0
    while collected < amount:
        demo = _rollout_scripted_demo(task_name, seed)
        seed += 1
        if demo is None:
            continue
        length = len(demo)
        obs_stack = np.stack([t["obs"] for t in demo]).astype(np.float32)
        next_obs_stack = np.stack([t["next_obs"] for t in demo]).astype(np.float32)
        actions = np.stack([t["action"] for t in demo])

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
        collected += 1

    obs_all = _concat(obs_parts)
    next_obs_all = _concat(next_obs_parts)
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


class MetaWorldDatasetBackend(DatasetBackend):
    @classmethod
    def infer_specs(cls, req: DatasetRequest):
        return infer_specs_from_metaworld(req.path)

    @classmethod
    def load(cls, buffer, req: DatasetRequest) -> int:
        return load_metaworld_dataset_to_replay_buffer(
            buffer,
            req.path,
            num_traj=req.num_traj,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            success_key=req.success_key,
        )


register_dataset_backend("metaworld", MetaWorldDatasetBackend)
