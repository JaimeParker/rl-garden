"""Generate WSRL-compatible offline datasets from trained policies.

The writer intentionally emits the minimal ManiSkill-style H5 shape consumed by
``load_maniskill_h5_to_replay_buffer``: one ``traj_*`` group per complete
episode, with ``obs`` length ``T + 1`` and transition fields length ``T``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np
import torch

from rl_garden.common.types import Obs

TierName = Literal["failure", "middle", "near_success", "success"]
TIERS: tuple[TierName, ...] = ("failure", "middle", "near_success", "success")
TIER_TARGETS: dict[TierName, float] = {
    "failure": 0.0,
    "middle": 0.5,
    "near_success": 0.75,
    "success": 1.0,
}


@dataclass(frozen=True)
class CheckpointScore:
    path: Path
    success_rate: float
    average_return: float
    average_length: float
    episodes: int


@dataclass(frozen=True)
class PolicySource:
    tier: TierName
    name: str
    path: Optional[Path]
    target_transitions: int
    success_rate: float
    fallback_reason: Optional[str] = None


@dataclass
class CollectionStats:
    episodes: int = 0
    transitions: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


def normalize_mix(mix: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(v) for v in mix)
    if len(values) != 4:
        raise ValueError("policy_mix must contain exactly four values.")
    if any(v < 0 for v in values):
        raise ValueError("policy_mix values must be non-negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("policy_mix must have a positive sum.")
    return tuple(v / total for v in values)  # type: ignore[return-value]


def discover_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    root = Path(checkpoint_dir)
    numbered: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint_*.pt"):
        suffix = path.stem.removeprefix("checkpoint_")
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    out = [path for _, path in sorted(numbered)]
    final = root / "final.pt"
    if final.exists():
        out.append(final)
    if not out:
        raise FileNotFoundError(f"No checkpoint_*.pt or final.pt found in {root}")
    return out


def _tier_for_rate(success_rate: float, thresholds: tuple[float, float, float]) -> TierName:
    failure_max, middle_max, success_min = thresholds
    if not 0 <= failure_max < success_min <= 1:
        raise ValueError("tier thresholds must satisfy 0 <= failure < success <= 1.")
    if success_rate <= failure_max:
        return "failure"
    if success_rate >= success_min:
        return "success"
    elif middle_max <= success_rate < success_min:
        return "near_success"
    return "middle"


def select_policy_sources(
    scores: list[CheckpointScore],
    *,
    total_transitions: int,
    policy_mix: Iterable[float] = (0.3, 0.3, 0.3, 0.1),
    thresholds: tuple[float, float, float] = (0.2, 0.6, 0.8),
    use_random_failure_fallback: bool = True,
) -> list[PolicySource]:
    """Pick one checkpoint per tier and assign normalized transition quotas."""
    if total_transitions <= 0:
        raise ValueError("total_transitions must be positive.")
    if not scores and not use_random_failure_fallback:
        raise ValueError("At least one checkpoint score is required.")

    mix = normalize_mix(policy_mix)
    buckets: dict[TierName, list[CheckpointScore]] = {tier: [] for tier in TIERS}
    for score in scores:
        buckets[_tier_for_rate(score.success_rate, thresholds)].append(score)

    sources: list[PolicySource] = []
    assigned = 0
    for i, tier in enumerate(TIERS):
        target = int(round(total_transitions * mix[i]))
        if i == len(TIERS) - 1:
            target = total_transitions - assigned
        assigned += target

        candidates = buckets[tier]
        fallback_reason = None
        if candidates:
            selected = min(
                candidates,
                key=lambda s: (abs(s.success_rate - TIER_TARGETS[tier]), str(s.path)),
            )
            path: Optional[Path] = selected.path
            success_rate = selected.success_rate
            name = selected.path.stem
        elif tier == "failure" and use_random_failure_fallback:
            path = None
            success_rate = 0.0
            name = "random"
            fallback_reason = "no checkpoint in failure tier; using random policy"
        else:
            selected = min(
                scores,
                key=lambda s: (abs(s.success_rate - TIER_TARGETS[tier]), str(s.path)),
            )
            path = selected.path
            success_rate = selected.success_rate
            name = selected.path.stem
            fallback_reason = f"no checkpoint in {tier} tier; using nearest score"

        sources.append(
            PolicySource(
                tier=tier,
                name=name,
                path=path,
                target_transitions=max(0, target),
                success_rate=float(success_rate),
                fallback_reason=fallback_reason,
            )
        )
    return sources


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Generating WSRL H5 datasets requires h5py.") from exc
    return h5py


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _stack_tree(xs: list[Any]) -> Any:
    if isinstance(xs[0], dict):
        return {key: _stack_tree([x[key] for x in xs]) for key in xs[0].keys()}
    return np.stack([_to_numpy(x) for x in xs], axis=0)


def _index_tree(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {key: _index_tree(value, index) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree[index].detach().cpu()
    return np.asarray(tree[index])


def _tree_to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_to_device(value, device) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree if tree.device == device else tree.to(device)
    return torch.as_tensor(tree, device=device)


class WSRLTrajectoryWriter:
    """Streaming H5 writer for complete WSRL offline trajectories."""

    def __init__(self, path: str | Path, metadata: Optional[dict[str, Any]] = None):
        h5py = _require_h5py()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")
        self._episode_id = 0
        self.stats = CollectionStats()
        for key, value in (metadata or {}).items():
            self._h5.attrs[key] = value

    def close(self) -> None:
        self._h5.attrs["episodes"] = self.stats.episodes
        self._h5.attrs["transitions"] = self.stats.transitions
        self._h5.attrs["successes"] = self.stats.successes
        self._h5.attrs["success_rate"] = self.stats.success_rate
        self._h5.close()

    def __enter__(self) -> "WSRLTrajectoryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write_tree(self, group: Any, key: str, data: Any) -> None:
        if isinstance(data, dict):
            sub = group.create_group(key, track_order=True)
            for subkey, value in data.items():
                self._write_tree(sub, subkey, value)
            return
        array = _to_numpy(data)
        kwargs: dict[str, Any] = {}
        if key in {"rgb", "depth", "seg"}:
            kwargs.update(compression="gzip", compression_opts=5)
        group.create_dataset(key, data=array, dtype=array.dtype, **kwargs)

    def write_episode(
        self,
        *,
        obs: list[Obs],
        actions: list[torch.Tensor],
        rewards: list[torch.Tensor],
        terminated: list[torch.Tensor | bool],
        truncated: list[torch.Tensor | bool],
        source: PolicySource,
        success: bool,
    ) -> int:
        if len(obs) != len(actions) + 1:
            raise ValueError("Episode observations must have length T + 1.")
        if not actions:
            return 0

        group = self._h5.create_group(f"traj_{self._episode_id}", track_order=True)
        self._episode_id += 1

        actions_np = np.stack([_to_numpy(a) for a in actions], axis=0).astype(np.float32)
        rewards_np = np.asarray([float(_to_numpy(r)) for r in rewards], dtype=np.float32)
        terminated_np = np.asarray([bool(_to_numpy(v)) for v in terminated], dtype=bool)
        truncated_np = np.asarray([bool(_to_numpy(v)) for v in truncated], dtype=bool)

        self._write_tree(group, "obs", _stack_tree(obs))
        group.create_dataset("actions", data=actions_np, dtype=np.float32)
        group.create_dataset("rewards", data=rewards_np, dtype=np.float32)
        group.create_dataset("terminated", data=terminated_np, dtype=bool)
        group.create_dataset("truncated", data=truncated_np, dtype=bool)

        group.attrs["source_tier"] = source.tier
        group.attrs["source_name"] = source.name
        group.attrs["source_checkpoint"] = "" if source.path is None else str(source.path)
        group.attrs["source_success_rate"] = source.success_rate
        group.attrs["final_success"] = bool(success)
        group.attrs["episode_return"] = float(rewards_np.sum())
        group.attrs["elapsed_steps"] = int(len(actions))

        self.stats.episodes += 1
        self.stats.transitions += len(actions)
        self.stats.successes += int(success)
        return len(actions)


def _extract_success(infos: dict[str, Any], env_idx: int) -> Optional[bool]:
    keys = ("success_at_end", "success_once", "success")

    def get_indexed(value: Any) -> Optional[bool]:
        if value is None:
            return None
        try:
            if isinstance(value, torch.Tensor):
                return bool(value.reshape(-1)[env_idx].item())
            array = np.asarray(value)
            if array.shape:
                return bool(array.reshape(-1)[env_idx].item())
            return bool(array.item())
        except Exception:
            return None

    final_info = infos.get("final_info")
    if isinstance(final_info, dict):
        episode = final_info.get("episode")
        if isinstance(episode, dict):
            for key in keys:
                if key in episode:
                    found = get_indexed(episode[key])
                    if found is not None:
                        return found
        for key in keys:
            if key in final_info:
                found = get_indexed(final_info[key])
                if found is not None:
                    return found

    for key in keys:
        if key in infos:
            found = get_indexed(infos[key])
            if found is not None:
                return found
    return None


def _final_next_obs(next_obs: Obs, infos: dict[str, Any], env_idx: int) -> Any:
    final_obs = infos.get("final_observation")
    if final_obs is None:
        return _index_tree(next_obs, env_idx)
    return _index_tree(final_obs, env_idx)


def _random_actions(env: Any, device: torch.device) -> torch.Tensor:
    shape = env.action_space.shape
    return 2.0 * torch.rand(shape, dtype=torch.float32, device=device) - 1.0


def _policy_actions(agent: Any, obs: Obs, *, deterministic: bool) -> torch.Tensor:
    policy_device = getattr(agent, "device", torch.device("cpu"))
    if not isinstance(policy_device, torch.device):
        policy_device = torch.device(policy_device)
    obs_on_device = _tree_to_device(obs, policy_device)
    with torch.no_grad():
        return agent.policy.predict(obs_on_device, deterministic=deterministic).detach()


def evaluate_policy_success(
    agent: Any,
    env: Any,
    *,
    episodes: int,
    deterministic: bool = True,
) -> CheckpointScore:
    """Evaluate a policy and return aggregate success/return/length metrics."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    obs, _ = env.reset()
    returns = torch.zeros(env.num_envs)
    lengths = torch.zeros(env.num_envs)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    successes = 0

    while len(completed_returns) < episodes:
        actions = _policy_actions(agent, obs, deterministic=deterministic)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        rewards_cpu = rewards.detach().cpu().float()
        done_cpu = (terminations | truncations).detach().cpu().bool()
        returns += rewards_cpu
        lengths += 1

        for env_idx, done in enumerate(done_cpu.tolist()):
            if not done or len(completed_returns) >= episodes:
                continue
            success = _extract_success(infos, env_idx)
            successes += int(bool(success))
            completed_returns.append(float(returns[env_idx].item()))
            completed_lengths.append(float(lengths[env_idx].item()))
            returns[env_idx] = 0
            lengths[env_idx] = 0
        obs = next_obs

    return CheckpointScore(
        path=Path(""),
        success_rate=successes / episodes,
        average_return=float(np.mean(completed_returns)),
        average_length=float(np.mean(completed_lengths)),
        episodes=episodes,
    )


def collect_policy_dataset(
    *,
    agent: Any | None,
    env: Any,
    writer: WSRLTrajectoryWriter,
    source: PolicySource,
    deterministic: bool = True,
    device: str | torch.device = "cpu",
) -> CollectionStats:
    """Roll out one source until its complete-episode transition quota is met."""
    target = source.target_transitions
    if target <= 0:
        return CollectionStats()

    action_device = torch.device(device)
    obs, _ = env.reset()
    episode_obs: list[list[Obs]] = [[_index_tree(obs, i)] for i in range(env.num_envs)]
    episode_actions: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    episode_rewards: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    episode_terminated: list[list[torch.Tensor | bool]] = [[] for _ in range(env.num_envs)]
    episode_truncated: list[list[torch.Tensor | bool]] = [[] for _ in range(env.num_envs)]
    stats = CollectionStats()
    print(f"[collect] starting collection for source {source.name}")

    while stats.transitions < target:
        if agent is None:
            actions = _random_actions(env, action_device)
        else:
            actions = _policy_actions(agent, obs, deterministic=deterministic)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = (terminations | truncations).detach().cpu().bool()

        for env_idx in range(env.num_envs):
            action_i = _index_tree(actions, env_idx)
            reward_i = rewards[env_idx].detach().cpu()
            term_i = terminations[env_idx].detach().cpu()
            trunc_i = truncations[env_idx].detach().cpu()
            next_obs_i = (
                _final_next_obs(next_obs, infos, env_idx)
                if bool(done[env_idx])
                else _index_tree(next_obs, env_idx)
            )

            episode_actions[env_idx].append(action_i)
            episode_rewards[env_idx].append(reward_i)
            episode_terminated[env_idx].append(term_i)
            episode_truncated[env_idx].append(trunc_i)
            episode_obs[env_idx].append(next_obs_i)

            if not bool(done[env_idx]):
                continue

            success = bool(_extract_success(infos, env_idx))
            written = writer.write_episode(
                obs=episode_obs[env_idx],
                actions=episode_actions[env_idx],
                rewards=episode_rewards[env_idx],
                terminated=episode_terminated[env_idx],
                truncated=episode_truncated[env_idx],
                source=source,
                success=success,
            )
            stats.episodes += 1
            stats.transitions += written
            stats.successes += int(success)
            episode_obs[env_idx] = [_index_tree(next_obs, env_idx)]
            episode_actions[env_idx] = []
            episode_rewards[env_idx] = []
            episode_terminated[env_idx] = []
            episode_truncated[env_idx] = []

        obs = next_obs
    print(f"[collect] finished collection for source {source.name}: "
          f"{stats.episodes} episodes, {stats.transitions} transitions, "
          f"success rate {stats.success_rate:.2%}")

    return stats
