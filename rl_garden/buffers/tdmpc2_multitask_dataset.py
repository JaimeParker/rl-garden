"""Runtime loader for the converted TD-MPC2 multitask dataset (output of
``tools/conversion/convert_tdmpc2_multitask_dataset.py``). Zero ``tensordict``
dependency -- reads only plain ``dict``-of-``torch.Tensor`` files.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer


def infer_multitask_dataset_specs(
    dataset_dir: str | Path,
) -> tuple[list[str], list[int], list[int], list[int]]:
    """Read ``manifest.json`` only -- no tensor data touched.

    Returns ``(tasks, obs_dims, action_dims, episode_lengths)``, mirroring
    what upstream's ``make_multitask_env``/``_load_dataset`` compute before
    constructing the agent (``cfg.tasks``/``cfg.obs_shapes``/
    ``cfg.action_dims``/``cfg.episode_lengths``).
    """
    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json found in {dataset_dir}; run "
            "tools/conversion/convert_tdmpc2_multitask_dataset.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        list(manifest["tasks"]),
        list(manifest["obs_dims"]),
        list(manifest["action_dims"]),
        list(manifest["episode_lengths"]),
    )


def load_multitask_dataset(
    buffer: MmapMultitaskEpisodeBuffer,
    dataset_dir: str | Path,
    *,
    tasks: Sequence[str] = (),
    max_episodes_per_task: int = 0,
) -> int:
    """Zero-pad every task's episodes to ``buffer.obs_dim``/``buffer.action_dim``
    (matching upstream's ``MultitaskWrapper._pad_obs``/``step()`` truncation
    scheme, applied here at load time instead of per-env-step) and bulk-load
    them via ``buffer.load_episode``. Returns the total transition count
    loaded.

    ``tasks`` (empty = all) restricts loading to a subset and
    ``max_episodes_per_task`` (0 = all) keeps each task's first N episodes.
    Task ids stay the manifest indices, so task embeddings are unaffected.
    """
    dataset_dir = Path(dataset_dir)
    manifest_tasks, obs_dims, action_dims, _episode_lengths = infer_multitask_dataset_specs(dataset_dir)
    unknown = sorted(set(tasks) - set(manifest_tasks))
    if unknown:
        raise ValueError(f"Tasks {unknown} are not in {dataset_dir}/manifest.json.")
    if max_episodes_per_task < 0:
        raise ValueError(f"max_episodes_per_task must be >= 0, got {max_episodes_per_task}.")

    total = 0
    for task_idx, task in enumerate(manifest_tasks):
        if tasks and task not in tasks:
            continue
        episodes_path = dataset_dir / task / "episodes.pt"
        data = torch.load(episodes_path, weights_only=True)
        obs, action, reward = data["obs"], data["action"], data["reward"]
        if max_episodes_per_task > 0:
            obs = obs[:max_episodes_per_task]
            action = action[:max_episodes_per_task]
            reward = reward[:max_episodes_per_task]
        num_episodes, episode_length = obs.shape[0], obs.shape[1]

        obs_dim = obs_dims[task_idx]
        action_dim = action_dims[task_idx]
        if obs.shape[-1] != obs_dim or action.shape[-1] != action_dim:
            raise ValueError(
                f"Task {task!r} tensor shapes {tuple(obs.shape)}/{tuple(action.shape)} "
                f"don't match manifest obs_dim={obs_dim}/action_dim={action_dim}."
            )

        obs_padded = torch.zeros(num_episodes, episode_length, buffer.obs_dim, dtype=torch.float32)
        obs_padded[..., :obs_dim] = obs
        action_padded = torch.zeros(num_episodes, episode_length, buffer.action_dim, dtype=torch.float32)
        action_padded[..., :action_dim] = action

        for ep in range(num_episodes):
            buffer.load_episode(obs_padded[ep], action_padded[ep], reward[ep], task_idx)
            total += episode_length

    return total
