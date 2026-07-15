"""One-off converter: TD-MPC2's official mt30/mt80 offline dataset -> rl-garden's
own plain-tensor on-disk format for ``TDMPC2Multitask``.

Prerequisite (only for running this script, NOT a core rl-garden dependency):
    pip install tensordict

Why this script needs ``tensordict`` at all: the official ``<task>.pt`` files
were saved with ``torch.save(TensorDict(...))``, so ``torch.load`` on them
requires ``tensordict`` importable just to unpickle the object -- there is no
way to read the raw bytes without it. Everything downstream of that single
``torch.load`` call operates on plain tensors (via ``td.get(key)``), and the
converted output this script writes is a plain ``dict`` of ``torch.Tensor``
(via ``torch.save``) that the runtime loader
(``rl_garden.algorithms.tdmpc2.multitask.dataset``) reads with zero
``tensordict`` dependency.

**KNOWN RISK -- not yet validated against a real official file.** The input
schema assumed below (``obs``/``action``/``reward`` keyed ``TensorDict``,
shape ``(num_episodes, episode_length+1, ...)``, with ``action``/``reward``'s
row 0 a placeholder -- inferred from reading
``3rd_party/tdmpc2/tdmpc2/common/buffer.py``'s ``_prepare_batch``, which
slices ``action``/``reward`` with ``[1:]`` before use) has not been checked
against an actual downloaded mt30/mt80 file. Run this script on ONE small
task file first and sanity-check ``manifest.json``'s ``episode_lengths``
against the known task-set values before converting the full dataset.

Usage:
    python tools/conversion/convert_tdmpc2_multitask_dataset.py \\
        --src_dir /path/to/official/mt80 --dst_dir /path/to/converted \\
        --tasks walker-stand walker-walk ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def reindex_episode_tensors(
    obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift upstream's "action/reward arrive at position i" convention
    (``action[:, i]`` is the action that produced ``obs[:, i]``, with row 0 a
    placeholder -- see ``common/buffer.py::_prepare_batch``'s ``[1:]`` slice)
    into rl-garden's "action/reward depart from position i" convention
    (matching every buffer in this package: ``action[i]`` is the action taken
    FROM ``obs[i]``). Drops the final observation (position ``L``), matching
    ``EpisodeSliceBuffer``'s documented "never store the true final obs"
    convention -- a window sampled from ``MmapMultitaskEpisodeBuffer`` can
    never reach it either way.

    ``obs``: ``(num_episodes, L+1, obs_dim)``. ``action``/``reward``:
    ``(num_episodes, L+1, ...)``, row 0 ignored. Returns length-``L``
    ``(num_episodes, L, ...)`` tensors.
    """
    length = obs.shape[1] - 1
    if length < 1:
        raise ValueError(f"episode window too short to reindex: obs.shape={tuple(obs.shape)}")
    return (
        obs[:, :length].contiguous(),
        action[:, 1 : length + 1].contiguous(),
        reward[:, 1 : length + 1].contiguous(),
    )


def convert_task_file(src_path: Path, dst_dir: Path) -> tuple[int, int, int, int]:
    """Convert one official ``<task>.pt`` file to rl-garden's format.

    Returns ``(num_episodes, episode_length, obs_dim, action_dim)``.
    """
    import tensordict  # noqa: F401  (registers TensorDict as unpicklable by torch.load)

    td = torch.load(src_path, weights_only=False)
    obs = td.get("obs")
    action = td.get("action")
    reward = td.get("reward")

    my_obs, my_action, my_reward = reindex_episode_tensors(obs, action, reward)

    dst_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": my_obs, "action": my_action, "reward": my_reward}, dst_dir / "episodes.pt")

    num_episodes, episode_length, obs_dim = my_obs.shape
    action_dim = my_action.shape[-1]
    return num_episodes, episode_length, obs_dim, action_dim


def convert_dataset(src_dir: Path, dst_dir: Path, tasks: list[str]) -> dict:
    manifest: dict[str, list] = {"tasks": [], "obs_dims": [], "action_dims": [], "episode_lengths": []}
    for task in tasks:
        src_path = src_dir / f"{task}.pt"
        num_episodes, episode_length, obs_dim, action_dim = convert_task_file(src_path, dst_dir / task)
        manifest["tasks"].append(task)
        manifest["obs_dims"].append(obs_dim)
        manifest["action_dims"].append(action_dim)
        manifest["episode_lengths"].append(episode_length)
        print(
            f"[convert] {task}: {num_episodes} episodes, episode_length={episode_length}, "
            f"obs_dim={obs_dim}, action_dim={action_dim}",
            flush=True,
        )
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_dir", required=True, help="Directory containing official <task>.pt files.")
    parser.add_argument("--dst_dir", required=True, help="Output directory for the converted dataset.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names (without .pt extension).")
    args = parser.parse_args()

    manifest = convert_dataset(Path(args.src_dir), Path(args.dst_dir), args.tasks)
    print(f"[convert] wrote manifest.json with {len(manifest['tasks'])} tasks to {args.dst_dir}", flush=True)


if __name__ == "__main__":
    main()
