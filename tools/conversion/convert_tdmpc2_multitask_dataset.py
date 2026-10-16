"""One-off converter: TD-MPC2's official mt30 offline dataset -> rl-garden's
own plain-tensor on-disk format for ``TDMPC2Multitask``.

Official source: HF dataset ``nicklashansen/tdmpc2``, directory ``mt30/``
(``chunk_0.pt`` .. ``chunk_3.pt``, ~44 GB total). Each chunk is a
``torch.save(TensorDict(...))`` with MIXED tasks -- there are no per-task
files. Schema validated against the real files (2026-10-15):

    key       shape              dtype     notes
    task      (E, 501)           int32     constant per episode; index into TASK_SET['mt30']
    obs       (E, 501, 24)       float32   zero-padded to the mt30 max obs dim (24)
    action    (E, 501, 6)        float32   zero-padded to max action dim (6);
                                           row 0 is UNINITIALIZED memory (garbage, not NaN)
    reward    (E, 501)           float32   row 0 is NaN
    (no ``terminated`` key)

``E`` is 200000 for chunks 0-2 and 90000 for chunk 3 (690000 episodes,
345.69M stored steps, matching upstream ``offline_trainer.py``'s
``buffer_size=345_690_000`` / ``episode_length=501``). Row ``i`` of
``action``/``reward`` is the action/reward that PRODUCED ``obs[:, i]``
(upstream ``common/buffer.py::_prepare_batch`` slices them with ``[1:]``).

What this script does:
  1. Load every chunk (``tensordict`` if importable, otherwise a restricted
     unpickler that maps only the two tensordict classes these files
     reference -- ``TensorDict`` and ``_StringOnlyDict`` -- to plain stubs,
     so no ``tensordict`` install is needed).
  2. Group episodes by task id across all chunks, reindex to rl-garden's
     "action/reward depart from obs[i]" convention (drops action/reward
     row 0 and the final obs -> 500 transitions per episode).
  3. Infer each task's true obs/action dims from the trailing all-zero
     padding columns, crop to them, and write ``<dst>/<task>/episodes.pt``
     (plain ``dict`` of tensors) plus ``manifest.json`` with tasks in
     TASK_SET order (so manifest index == upstream task-embedding index).

Usage:
    python tools/conversion/convert_tdmpc2_multitask_dataset.py \\
        --src_dir /data/tdmpc2/mt30 --dst_dir /data/tdmpc2/mt30_rlgarden
    # validation on a subset of chunks (not all 30 tasks present):
    python tools/conversion/convert_tdmpc2_multitask_dataset.py \\
        --src_dir /data/tdmpc2/mt30 --dst_dir /tmp/check --chunks chunk_3.pt --allow_missing_tasks
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import types
from pathlib import Path

import torch

# Copied from 3rd_party/tdmpc2/tdmpc2/common/__init__.py (TASK_SET['mt30']);
# the dataset's ``task`` ids index into this list.
MT30_TASKS: tuple[str, ...] = (
    "walker-stand", "walker-walk", "walker-run", "cheetah-run", "reacher-easy",
    "reacher-hard", "acrobot-swingup", "pendulum-swingup", "cartpole-balance", "cartpole-balance-sparse",
    "cartpole-swingup", "cartpole-swingup-sparse", "cup-catch", "finger-spin", "finger-turn-easy",
    "finger-turn-hard", "fish-swim", "hopper-stand", "hopper-hop",
    "walker-walk-backwards", "walker-run-backwards", "cheetah-run-backwards", "cheetah-run-front", "cheetah-run-back",
    "cheetah-jump", "hopper-hop-backwards", "reacher-three-easy", "reacher-three-hard", "cup-spin",
    "pendulum-spin",
)
OFFICIAL_KEYS = ("task", "obs", "action", "reward")


class _TensorDictStub:
    """Receives a pickled ``TensorDict``'s ``__dict__`` state (``_tensordict``,
    ``_batch_size``, ...) via ``BUILD``."""

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class _StringOnlyDictStub(dict):
    pass


_TENSORDICT_STUBS = {
    ("tensordict.tensordict", "TensorDict"): _TensorDictStub,
    ("tensordict.utils", "_StringOnlyDict"): _StringOnlyDictStub,
}


class _TensorDictStubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split(".")[0] == "tensordict":
            try:
                return _TENSORDICT_STUBS[(module, name)]
            except KeyError:
                raise pickle.UnpicklingError(
                    f"Unsupported tensordict global {module}.{name}; install tensordict to read this file."
                ) from None
        return super().find_class(module, name)


_stub_pickle_module = types.ModuleType("tensordict_stub_pickle")
_stub_pickle_module.Unpickler = _TensorDictStubUnpickler
_stub_pickle_module.load = pickle.load


def load_chunk_without_tensordict(path: Path) -> dict[str, torch.Tensor]:
    td = torch.load(path, weights_only=False, pickle_module=_stub_pickle_module)
    if not isinstance(td, _TensorDictStub):
        raise TypeError(f"{path}: expected a pickled TensorDict, got {type(td).__name__}")
    return dict(td.__dict__["_tensordict"])


def load_official_chunk(path: Path) -> dict[str, torch.Tensor]:
    """Return the chunk's ``{key: tensor}`` for :data:`OFFICIAL_KEYS`."""
    try:
        import tensordict  # noqa: F401  (registers TensorDict for unpickling)
    except ImportError:
        data = load_chunk_without_tensordict(path)
    else:
        td = torch.load(path, weights_only=False)
        data = {key: td.get(key) for key in td.keys()}
    missing = [key for key in OFFICIAL_KEYS if key not in data]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}; found {sorted(data)}")
    return data


def reindex_episode_tensors(
    obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift upstream's "action/reward arrive at position i" convention
    (``action[:, i]`` is the action that produced ``obs[:, i]``, with row 0 a
    placeholder -- see ``common/buffer.py::_prepare_batch``'s ``[1:]`` slice)
    into rl-garden's "action/reward depart from position i" convention
    (matching every buffer in this package: ``action[i]`` is the action taken
    FROM ``obs[i]``). Drops the final observation (position ``L``), matching
    ``MmapMultitaskEpisodeBuffer.load_episode``'s documented "never store the
    true final obs" convention.

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


def split_chunk_by_task(
    data: dict[str, torch.Tensor],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Group one chunk's episodes by task id and reindex them (still padded)."""
    task = data["task"]
    if not bool((task == task[:, :1]).all()):
        raise ValueError("task id changes within an episode")
    episode_task = task[:, 0].long()
    grouped = {}
    for task_id in episode_task.unique().tolist():
        selected = (episode_task == task_id).nonzero().flatten()
        obs, action, reward = reindex_episode_tensors(
            data["obs"][selected], data["action"][selected], data["reward"][selected]
        )
        for name, tensor in (("obs", obs), ("action", action), ("reward", reward)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"task id {task_id}: non-finite values in reindexed {name}")
        grouped[int(task_id)] = (obs, action, reward)
    return grouped


def infer_unpadded_dim(padded: torch.Tensor) -> int:
    """``1 +`` the last feature column that is nonzero anywhere in ``padded``
    (``(..., D)``); upstream zero-pads obs/action to the task-set max dim."""
    nonzero_columns = (padded != 0).reshape(-1, padded.shape[-1]).any(dim=0).nonzero().flatten()
    if nonzero_columns.numel() == 0:
        raise ValueError("all feature columns are zero; cannot infer the unpadded dim")
    return int(nonzero_columns[-1]) + 1


def _domain(task: str) -> str:
    """Tasks expected to share obs/action dims (finger-turn-* adds target
    position/distance to finger-spin's 9-dim obs; reacher-three-* is a
    3-link reacher)."""
    for family in ("reacher-three", "finger-turn"):
        if task.startswith(family):
            return family
    return task.split("-")[0]


def _warn_on_inconsistent_domain_dims(manifest: dict) -> None:
    dims_by_domain: dict[str, set] = {}
    for task, obs_dim, action_dim in zip(manifest["tasks"], manifest["obs_dims"], manifest["action_dims"]):
        dims_by_domain.setdefault(_domain(task), set()).add((obs_dim, action_dim))
    for domain, dims in dims_by_domain.items():
        if len(dims) > 1:
            print(f"[convert] WARNING: domain {domain!r} has differing inferred (obs, action) dims {sorted(dims)}", flush=True)


def convert_dataset(
    src_dir: Path,
    dst_dir: Path,
    chunk_names: list[str] | None = None,
    task_names: tuple[str, ...] = MT30_TASKS,
    allow_missing_tasks: bool = False,
) -> dict:
    chunk_paths = [src_dir / name for name in chunk_names] if chunk_names else sorted(src_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunk_*.pt files in {src_dir}")

    per_task: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for chunk_path in chunk_paths:
        start = time.time()
        data = load_official_chunk(chunk_path)
        print(
            f"[convert] loaded {chunk_path.name}: "
            + ", ".join(f"{k}={tuple(data[k].shape)}/{data[k].dtype}" for k in OFFICIAL_KEYS)
            + f" ({time.time() - start:.0f}s)",
            flush=True,
        )
        for task_id, tensors in split_chunk_by_task(data).items():
            if not 0 <= task_id < len(task_names):
                raise ValueError(f"{chunk_path.name}: task id {task_id} outside task set of size {len(task_names)}")
            per_task.setdefault(task_id, []).append(tensors)
        del data

    missing = [task_names[i] for i in range(len(task_names)) if i not in per_task]
    if missing and not allow_missing_tasks:
        raise ValueError(f"tasks missing from {src_dir}: {missing} (pass --allow_missing_tasks to convert a subset)")

    manifest: dict[str, list] = {
        "tasks": [], "obs_dims": [], "action_dims": [], "episode_lengths": [], "num_episodes": [],
    }
    for task_id in sorted(per_task):
        parts = per_task.pop(task_id)
        obs = torch.cat([p[0] for p in parts])
        action = torch.cat([p[1] for p in parts])
        reward = torch.cat([p[2] for p in parts])
        del parts
        obs_dim, action_dim = infer_unpadded_dim(obs), infer_unpadded_dim(action)
        task = task_names[task_id]
        task_dir = dst_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs": obs[..., :obs_dim].contiguous(),
                "action": action[..., :action_dim].contiguous(),
                "reward": reward,
            },
            task_dir / "episodes.pt",
        )
        num_episodes, episode_length = obs.shape[0], obs.shape[1]
        manifest["tasks"].append(task)
        manifest["obs_dims"].append(obs_dim)
        manifest["action_dims"].append(action_dim)
        manifest["episode_lengths"].append(episode_length)
        manifest["num_episodes"].append(num_episodes)
        print(
            f"[convert] {task} (id {task_id}): {num_episodes} episodes, episode_length={episode_length}, "
            f"obs_dim={obs_dim}, action_dim={action_dim}",
            flush=True,
        )
    _warn_on_inconsistent_domain_dims(manifest)
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src_dir", required=True, help="Directory containing the official chunk_*.pt files.")
    parser.add_argument("--dst_dir", required=True, help="Output directory for the converted dataset.")
    parser.add_argument("--chunks", nargs="+", default=None, help="Chunk file names (default: all chunk_*.pt).")
    parser.add_argument(
        "--allow_missing_tasks", action="store_true", help="Allow converting chunks that omit some tasks."
    )
    args = parser.parse_args()

    start = time.time()
    manifest = convert_dataset(
        Path(args.src_dir), Path(args.dst_dir), args.chunks, allow_missing_tasks=args.allow_missing_tasks
    )
    total_episodes = sum(manifest["num_episodes"])
    total_transitions = sum(n * length for n, length in zip(manifest["num_episodes"], manifest["episode_lengths"]))
    print(
        f"[convert] wrote manifest.json with {len(manifest['tasks'])} tasks, {total_episodes} episodes, "
        f"{total_transitions} transitions to {args.dst_dir} in {time.time() - start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
