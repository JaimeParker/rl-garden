"""Collect successful and near-success StackCube rollouts from DrQ-v2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from rl_garden.algorithms.ddpg import DDPG
from rl_garden.common import seed_everything
from rl_garden.encoders import discover_image_keys
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


STAGE_KEYS = ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static", "success")


@dataclass
class Args:
    checkpoint_path: str
    output_path: str
    num_envs: int = 64
    target_successes: int = 20
    target_near_successes: int = 20
    max_episodes: int = 20_000
    stddev: float = 0.1
    seed: int = 1001
    device: str = "auto"


@dataclass
class Episode:
    obs: list[dict[str, torch.Tensor]] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    actor_mean_actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    terminated: list[torch.Tensor] = field(default_factory=list)
    truncated: list[torch.Tensor] = field(default_factory=list)
    stages: dict[str, list[bool]] = field(
        default_factory=lambda: {key: [] for key in STAGE_KEYS}
    )

    @property
    def success(self) -> bool:
        return any(self.stages["success"])

    @property
    def near_success(self) -> bool:
        return (not self.success) and any(self.stages["is_cubeA_on_cubeB"])


def classify_episode(episode: Episode) -> str:
    if episode.success:
        return "success"
    if episode.near_success:
        return "near_success"
    return "failure"


def _index_tree(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {key: _index_tree(value, index) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree[index].detach().cpu()
    return torch.as_tensor(np.asarray(tree[index]))


def _stack_tree(values: list[Any]) -> Any:
    if isinstance(values[0], dict):
        return {key: _stack_tree([value[key] for value in values]) for key in values[0]}
    return torch.stack([torch.as_tensor(value) for value in values]).numpy()


def _info_value(infos: dict[str, Any], key: str, index: int, done: bool) -> bool:
    source: Any = infos
    if done and isinstance(infos.get("final_info"), dict):
        source = infos["final_info"]
    value = source.get(key)
    if value is None:
        raise KeyError(f"Missing required per-step info key {key!r}")
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[index].item())
    return bool(np.asarray(value).reshape(-1)[index].item())


def _final_obs(next_obs: Any, infos: dict[str, Any], index: int, done: bool) -> Any:
    if done and infos.get("final_observation") is not None:
        return _index_tree(infos["final_observation"], index)
    return _index_tree(next_obs, index)


def _write_tree(group: Any, key: str, value: Any) -> None:
    if isinstance(value, dict):
        child = group.create_group(key, track_order=True)
        for child_key, child_value in value.items():
            _write_tree(child, child_key, child_value)
        return
    array = np.asarray(value)
    kwargs = {"compression": "gzip", "compression_opts": 3} if array.ndim >= 3 else {}
    group.create_dataset(key, data=array, dtype=array.dtype, **kwargs)


class SuccessRolloutWriter:
    def __init__(self, path: str | Path, metadata: dict[str, Any]):
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "w")
        self.count = 0
        for key, value in metadata.items():
            self.handle.attrs[key] = value

    def write(self, episode: Episode, label: str) -> None:
        group = self.handle.create_group(f"traj_{self.count}", track_order=True)
        self.count += 1
        _write_tree(group, "obs", _stack_tree(episode.obs))
        group.create_dataset("actions", data=_stack_tree(episode.actions), dtype=np.float32)
        group.create_dataset(
            "actor_mean_actions",
            data=_stack_tree(episode.actor_mean_actions),
            dtype=np.float32,
        )
        group.create_dataset("rewards", data=_stack_tree(episode.rewards), dtype=np.float32)
        group.create_dataset("terminated", data=_stack_tree(episode.terminated), dtype=bool)
        group.create_dataset("truncated", data=_stack_tree(episode.truncated), dtype=bool)
        info_group = group.create_group("infos", track_order=True)
        for key, values in episode.stages.items():
            info_group.create_dataset(key, data=np.asarray(values, dtype=bool))
        group.attrs["label"] = label
        group.attrs["final_success"] = episode.success
        group.attrs["episode_return"] = float(torch.stack(episode.rewards).sum().item())
        group.attrs["elapsed_steps"] = len(episode.actions)

    def close(self, *, episodes_seen: int, successes: int, near_successes: int) -> None:
        self.handle.attrs["episodes_seen"] = episodes_seen
        self.handle.attrs["successes"] = successes
        self.handle.attrs["near_successes"] = near_successes
        self.handle.close()


def update_reservoir(
    reservoir: list[Episode], episode: Episode, seen: int, capacity: int, rng: np.random.Generator
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(episode)
        return
    replacement = int(rng.integers(0, seen))
    if replacement < capacity:
        reservoir[replacement] = episode


def _make_agent(args: Args, env: Any) -> DDPG:
    image_keys = discover_image_keys(env.single_observation_space)
    agent = DDPG(
        env=env,
        eval_env=None,
        buffer_size=max(1024, args.num_envs * 4),
        buffer_device="cuda",
        learning_starts=4_000,
        batch_size=256,
        gamma=0.8,
        tau=0.01,
        training_freq=32,
        utd=0.5,
        policy_lr=1e-4,
        q_lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        nstep=3,
        stddev_schedule="linear(1.0,0.1,500000)",
        stddev_clip=0.3,
        num_expl_steps=2_000,
        image_keys=image_keys,
        image_fusion_mode="per_key",
        image_augmentation="random_shift",
        random_shift_pad=4,
        image_augmentation_seed=args.seed + 1_000_003,
        seed=args.seed,
        device=args.device,
        logger=None,
        std_log=False,
        eval_freq=0,
        checkpoint_dir=None,
        save_final_checkpoint=False,
    )
    agent.load(args.checkpoint_path, load_replay_buffer=False, load_optimizers=False)
    agent.policy.eval()
    return agent


def main() -> None:
    args = tyro.cli(Args)
    if args.target_successes <= 0 or args.max_episodes <= 0:
        raise ValueError("target_successes and max_episodes must be positive")
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_maniskill_env(
        ManiSkillEnvConfig(
            env_id="StackCube-v1",
            num_envs=args.num_envs,
            obs_mode="rgb",
            include_state=True,
            control_mode="pd_joint_delta_pos",
            camera_width=64,
            camera_height=64,
            per_camera_rgbd=True,
            sim_backend="gpu",
            render_backend="gpu",
            render_mode="rgb_array",
            reward_mode="normalized_dense",
            ignore_terminations=True,
            record_metrics=True,
        )
    )
    agent = _make_agent(args, env)
    obs, _ = env.reset(seed=args.seed)
    episodes = [Episode(obs=[_index_tree(obs, i)]) for i in range(args.num_envs)]
    success_count = 0
    near_seen = 0
    episodes_seen = 0
    near_reservoir: list[Episode] = []
    metadata = {
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "stddev": args.stddev,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
    }
    writer = SuccessRolloutWriter(args.output_path, metadata)

    try:
        while episodes_seen < args.max_episodes and (
            success_count < args.target_successes
            or len(near_reservoir) < args.target_near_successes
        ):
            with torch.no_grad():
                policy_obs = {
                    key: value if value.device == agent.device else value.to(agent.device)
                    for key, value in obs.items()
                }
                features = agent.policy.extract_features(policy_obs)
                actor_mean = agent.policy.actor.deterministic_action(features)
                actions = agent.policy.actor(features, args.stddev).sample(clip=None)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            done = (terminated | truncated).detach().cpu().bool()

            for i in range(args.num_envs):
                episode = episodes[i]
                episode.actions.append(_index_tree(actions, i))
                episode.actor_mean_actions.append(_index_tree(actor_mean, i))
                episode.rewards.append(_index_tree(rewards, i))
                episode.terminated.append(_index_tree(terminated, i))
                episode.truncated.append(_index_tree(truncated, i))
                for key in STAGE_KEYS:
                    episode.stages[key].append(_info_value(infos, key, i, bool(done[i])))
                episode.obs.append(_final_obs(next_obs, infos, i, bool(done[i])))
                if not bool(done[i]):
                    continue

                episodes_seen += 1
                label = classify_episode(episode)
                if label == "success" and success_count < args.target_successes:
                    writer.write(episode, label)
                    success_count += 1
                    print(f"[collect] success {success_count}/{args.target_successes} "
                          f"after {episodes_seen} episodes", flush=True)
                elif label == "near_success":
                    near_seen += 1
                    update_reservoir(
                        near_reservoir,
                        episode,
                        near_seen,
                        args.target_near_successes,
                        rng,
                    )
                episodes[i] = Episode(obs=[_index_tree(next_obs, i)])
                if episodes_seen >= args.max_episodes:
                    break
            obs = next_obs

        for episode in near_reservoir:
            writer.write(episode, "near_success")
    finally:
        writer.close(
            episodes_seen=episodes_seen,
            successes=success_count,
            near_successes=len(near_reservoir),
        )
        env.close()

    summary = {
        "episodes_seen": episodes_seen,
        "successes": success_count,
        "near_successes": len(near_reservoir),
        "output_path": args.output_path,
        "complete": success_count >= args.target_successes
        and len(near_reservoir) >= args.target_near_successes,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
