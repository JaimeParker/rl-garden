"""Generate residual-offline H5 data from replayed ManiSkill demonstrations.

Only replay-successful episodes are written to the output H5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs  # noqa: F401
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode

from rl_garden.common.action_scaler import ActionScaler
from rl_garden.envs import register_custom_envs
from rl_garden.envs.wrappers import PerCameraRGBDWrapper
from rl_garden.policies.base_policies import make_base_policy


@dataclass
class Args:
    traj_path: Annotated[str, tyro.conf.Positional]
    output_path: Optional[str] = None
    count: Optional[int] = None
    capture_video: bool = False
    output_dir: Optional[str] = None
    video_fps: int = 30
    use_env_states: bool = True
    device: str = "auto"
    obs: Literal["state", "rgb"] = "state"
    include_state: bool = True
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    per_camera_rgbd: bool = True

    base_policy: Literal["act", "sac", "zero"] = "act"
    base_ckpt_path: Optional[str] = "act-peg-only"
    base_act_temporal_agg: bool = True
    base_act_temporal_agg_k: float = 0.01
    base_sac_encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    base_sac_encoder_features_dim: int = 256
    base_sac_image_fusion_mode: Optional[Literal["stack_channels", "per_key"]] = None
    base_sac_deterministic: bool = True


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Generating residual offline H5 requires h5py in the runtime env."
        ) from exc
    return h5py


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _json_path_for(path: Path) -> Path:
    return path.with_suffix(".json")


def _default_output_path(path: Path, args: Args) -> Path:
    if args.obs == "state":
        return path.with_suffix(".residual.h5")
    suffix = ".residual.rgb_state.h5" if args.include_state else ".residual.rgb.h5"
    return path.with_suffix(suffix)


def _default_output_dir(path: Path) -> Path:
    return path.parent / "residual_replay"


def _normalize_reset_kwargs(episode: dict[str, Any]) -> dict[str, Any]:
    reset_kwargs = dict(episode.get("reset_kwargs") or {})
    seed = reset_kwargs.get("seed", episode.get("episode_seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(
                f"Episode {episode.get('episode_id')} has ambiguous reset seed: {seed}."
            )
        seed = seed[0]
    reset_kwargs["seed"] = seed
    return reset_kwargs


def _set_env_state(env: Any, state: dict[str, Any]) -> None:
    target = getattr(env, "base_env", env)
    if hasattr(target, "set_state_dict"):
        target.set_state_dict(state)
        return
    target.unwrapped.set_state_dict(state)


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            return bool(value.reshape(-1)[0].item())
        array = np.asarray(value)
        if array.shape:
            return bool(array.reshape(-1)[0].item())
        return bool(array.item())
    except Exception:
        return None


def _as_scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0].item())


def _as_scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0].item())


def _extract_success(info: dict[str, Any]) -> Optional[bool]:
    keys = ("success_at_end", "success_once", "success")
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        episode = final_info.get("episode")
        if isinstance(episode, dict):
            for key in keys:
                found = _as_bool(episode.get(key))
                if found is not None:
                    return found
        for key in keys:
            found = _as_bool(final_info.get(key))
            if found is not None:
                return found
    for key in keys:
        found = _as_bool(info.get(key))
        if found is not None:
            return found
    return None


def _build_env(metadata: dict[str, Any], args: Args):
    register_custom_envs()
    env_info = metadata["env_info"]
    env_kwargs = dict(env_info.get("env_kwargs") or {})
    env_kwargs["num_envs"] = 1
    env_kwargs.setdefault("render_mode", "rgb_array")
    if args.obs == "rgb":
        env_kwargs["obs_mode"] = "rgb"
        env_kwargs.setdefault("sensor_configs", {})
        if args.camera_width is not None:
            env_kwargs["sensor_configs"]["width"] = args.camera_width
        if args.camera_height is not None:
            env_kwargs["sensor_configs"]["height"] = args.camera_height
    env = gym.make(env_info["env_id"], **env_kwargs)
    if args.obs == "rgb":
        if args.per_camera_rgbd:
            env = PerCameraRGBDWrapper(
                env,
                rgb=True,
                depth=False,
                state=args.include_state,
            )
        else:
            env = FlattenRGBDObservationWrapper(
                env,
                rgb=True,
                depth=False,
                state=args.include_state,
            )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if args.capture_video:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else _default_output_dir(Path(args.traj_path))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=str(output_dir),
            save_trajectory=False,
            save_video=True,
            trajectory_name="residual_replay",
            max_steps_per_video=env_info.get("max_episode_steps"),
            video_fps=args.video_fps,
        )
    return env


def _space(env: Any, name: str):
    single_name = f"single_{name}_space"
    if hasattr(env, single_name):
        return getattr(env, single_name)
    return getattr(env, f"{name}_space")


def _source_state_observation_space(path: Path, episodes: list[dict[str, Any]]):
    h5py = _require_h5py()
    episode_id = int(episodes[0]["episode_id"])
    with h5py.File(path, "r") as f:
        node = f[f"traj_{episode_id}"]["obs"]
        if isinstance(node, h5py.Group):
            if "state" in node:
                node = node["state"]
            else:
                node = node[next(iter(node.keys()))]
        shape = tuple(node.shape[1:])
        dtype = np.dtype(node.dtype)
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)


def _read_node(node: Any) -> Any:
    h5py = _require_h5py()
    if isinstance(node, h5py.Dataset):
        return node[()]
    if isinstance(node, h5py.Group):
        return {key: _read_node(node[key]) for key in node.keys()}
    raise TypeError(f"Unsupported H5 node type: {type(node)!r}")


def _index_tree(x: Any, idx: int) -> Any:
    if isinstance(x, dict):
        return {key: _index_tree(value, idx) for key, value in x.items()}
    return x[idx]


def _slice_tree(x: Any, start: int, end: int) -> Any:
    if isinstance(x, dict):
        return {key: _slice_tree(value, start, end) for key, value in x.items()}
    return x[start:end]


def _unbatch_tree(x: Any) -> Any:
    if isinstance(x, dict):
        return {key: _unbatch_tree(value) for key, value in x.items()}
    array = _to_numpy(x)
    if array.shape and array.shape[0] == 1:
        return array[0]
    return array


def _stack_tree(xs: list[Any]) -> Any:
    if isinstance(xs[0], dict):
        return {key: _stack_tree([x[key] for x in xs]) for key in xs[0].keys()}
    return np.stack([_to_numpy(x) for x in xs], axis=0)


def _wrapped_current_obs(env: Any) -> Any:
    wrappers = []
    current = env
    while hasattr(current, "env"):
        wrappers.append(current)
        current = current.env
    obs = current.unwrapped.get_obs()
    for wrapper in reversed(wrappers):
        if isinstance(wrapper, gym.ObservationWrapper):
            obs = wrapper.observation(obs)
    return _unbatch_tree(obs)


def _to_batched_tensor(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {key: _to_batched_tensor(value, device) for key, value in x.items()}
    tensor = torch.as_tensor(x, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor.unsqueeze(0)


def _to_numpy(x: Any) -> Any:
    if isinstance(x, dict):
        return {key: _to_numpy(value) for key, value in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _write_tree(group: Any, key: str, data: Any) -> None:
    if isinstance(data, dict):
        sub = group.create_group(key, track_order=True)
        for subkey, value in data.items():
            _write_tree(sub, subkey, value)
        return
    array = _to_numpy(data)
    kwargs: dict[str, Any] = {}
    if key.startswith("rgb") or key.startswith("depth"):
        kwargs.update(compression="gzip", compression_opts=5)
    group.create_dataset(key, data=array, dtype=array.dtype, **kwargs)


def _base_naction(
    provider, scaler: ActionScaler, obs: Any, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        output = provider.select_action(_to_batched_tensor(obs, device))
        return scaler.scale(output.actions).clamp(-1.0, 1.0).detach()


def generate(args: Args) -> dict[str, Any]:
    h5py = _require_h5py()
    device = _device(args.device)
    traj_path = Path(args.traj_path).expanduser()
    json_path = _json_path_for(traj_path)
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else _default_output_path(traj_path, args)
    )
    metadata = json.loads(json_path.read_text())
    episodes = list(metadata.get("episodes") or [])
    if args.count is not None:
        episodes = episodes[: args.count]
    if not episodes:
        raise ValueError(f"No episodes to replay from metadata: {json_path}")

    env = _build_env(metadata, args)
    action_scaler = ActionScaler.from_action_space(_space(env, "action"), device=device)
    provider = make_base_policy(
        base_policy=args.base_policy,
        observation_space=_source_state_observation_space(traj_path, episodes),
        action_space=_space(env, "action"),
        env=env,
        base_ckpt_path=args.base_ckpt_path,
        device=device,
        base_act_temporal_agg=args.base_act_temporal_agg,
        base_act_temporal_agg_k=args.base_act_temporal_agg_k,
        base_sac_encoder=args.base_sac_encoder,
        base_sac_encoder_features_dim=args.base_sac_encoder_features_dim,
        base_sac_image_fusion_mode=args.base_sac_image_fusion_mode,
        base_sac_deterministic=args.base_sac_deterministic,
    )

    kept: list[dict[str, Any]] = []
    failed_episode_ids: list[int] = []
    missing_success_episode_ids: list[int] = []
    try:
        with h5py.File(traj_path, "r") as src:
            for episode in tqdm(episodes, desc="Generating", unit="episode"):
                episode_id = int(episode["episode_id"])
                traj = src[f"traj_{episode_id}"]
                source_obs = _read_node(traj["obs"])
                env_actions = np.asarray(traj["actions"], dtype=np.float32)
                env_states = (
                    trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
                    if args.use_env_states and "env_states" in traj
                    else None
                )
                provider.reset()
                env.reset(**_normalize_reset_kwargs(episode))
                if env_states:
                    _set_env_state(env, env_states[0])
                recorded_obs = []
                if args.obs == "rgb":
                    recorded_obs.append(_wrapped_current_obs(env))

                base_actions: list[np.ndarray] = []
                next_base_actions: list[np.ndarray] = []
                rewards: list[float] = []
                terminated: list[bool] = []
                truncated: list[bool] = []
                successes: list[bool] = []
                cached_base: Optional[torch.Tensor] = None
                replay_success: Optional[bool] = None
                for step, env_action in enumerate(env_actions):
                    obs_t = _index_tree(source_obs, step)
                    if cached_base is None:
                        base = _base_naction(provider, action_scaler, obs_t, device)
                    else:
                        base = cached_base
                    _, reward, term, trunc, info = env.step(env_action)
                    if env_states and step + 1 < len(env_states):
                        _set_env_state(env, env_states[step + 1])
                    if args.obs == "rgb":
                        recorded_obs.append(_wrapped_current_obs(env))
                    next_obs_t = _index_tree(source_obs, step + 1)
                    next_base = _base_naction(
                        provider, action_scaler, next_obs_t, device
                    )
                    cached_base = next_base

                    found_success = _extract_success(info)
                    if found_success is not None:
                        replay_success = found_success
                    base_actions.append(base.squeeze(0).cpu().numpy())
                    next_base_actions.append(next_base.squeeze(0).cpu().numpy())
                    rewards.append(_as_scalar_float(reward))
                    terminated.append(_as_scalar_bool(term))
                    truncated.append(_as_scalar_bool(trunc))
                    successes.append(
                        bool(found_success) if found_success is not None else False
                    )

                if replay_success is None:
                    missing_success_episode_ids.append(episode_id)
                success = bool(replay_success) if replay_success is not None else False
                if not success:
                    failed_episode_ids.append(episode_id)
                    continue
                kept.append(
                    {
                        "source_episode_id": episode_id,
                        "episode_seed": episode.get("episode_seed"),
                        "obs": (
                            _slice_tree(source_obs, 0, len(env_actions) + 1)
                            if args.obs == "state"
                            else _stack_tree(recorded_obs)
                        ),
                        "actions": action_scaler.scale(
                            torch.as_tensor(
                                env_actions, dtype=torch.float32, device=device
                            )
                        )
                        .cpu()
                        .numpy(),
                        "env_actions": env_actions,
                        "base_actions": np.stack(base_actions).astype(np.float32),
                        "next_base_actions": np.stack(next_base_actions).astype(
                            np.float32
                        ),
                        "rewards": np.asarray(rewards, dtype=np.float32),
                        "terminated": np.asarray(terminated, dtype=bool),
                        "truncated": np.asarray(truncated, dtype=bool),
                        "success": np.asarray(successes, dtype=bool),
                    }
                )
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transitions = int(sum(item["actions"].shape[0] for item in kept))
    with h5py.File(output_path, "w") as out:
        out.attrs["dataset_type"] = "rl_garden_residual_offline"
        out.attrs["format_version"] = 1
        out.attrs["source_traj_path"] = str(traj_path)
        out.attrs["source_json_path"] = str(json_path)
        out.attrs["obs_mode"] = args.obs
        out.attrs["include_state"] = bool(args.include_state)
        out.attrs["per_camera_rgbd"] = bool(args.per_camera_rgbd)
        out.attrs["camera_width"] = (
            -1 if args.camera_width is None else args.camera_width
        )
        out.attrs["camera_height"] = (
            -1 if args.camera_height is None else args.camera_height
        )
        out.attrs["action_coordinates"] = "normalized"
        out.attrs["env_action_coordinates"] = "env"
        out.attrs["base_policy"] = args.base_policy
        out.attrs["base_ckpt_path"] = (
            ""
            if args.base_policy == "zero" or args.base_ckpt_path is None
            else str(args.base_ckpt_path)
        )
        out.attrs["episodes"] = len(kept)
        out.attrs["transitions"] = transitions
        out.attrs["replay_attempted_episodes"] = len(episodes)
        out.attrs["replay_successes"] = len(kept)
        out.attrs["replay_success_rate"] = len(kept) / len(episodes)
        out.attrs["replay_success_signal_missing"] = len(missing_success_episode_ids)
        for idx, item in enumerate(kept):
            group = out.create_group(f"traj_{idx}", track_order=True)
            _write_tree(group, "obs", item["obs"])
            for key in (
                "actions",
                "env_actions",
                "base_actions",
                "next_base_actions",
                "rewards",
                "terminated",
                "truncated",
                "success",
            ):
                _write_tree(group, key, item[key])
            group.attrs["source_episode_id"] = item["source_episode_id"]
            if item["episode_seed"] is not None:
                group.attrs["episode_seed"] = int(item["episode_seed"])
            group.attrs["elapsed_steps"] = int(item["actions"].shape[0])
            group.attrs["replay_success"] = True
            group.attrs["episode_return"] = float(item["rewards"].sum())
        summary = out.create_group("summary")
        summary.create_dataset(
            "failed_episode_ids", data=np.asarray(failed_episode_ids, dtype=np.int64)
        )
        summary.create_dataset(
            "source_episode_ids",
            data=np.asarray(
                [item["source_episode_id"] for item in kept], dtype=np.int64
            ),
        )
        summary.create_dataset(
            "missing_success_episode_ids",
            data=np.asarray(missing_success_episode_ids, dtype=np.int64),
        )

    summary = {
        "output_path": str(output_path),
        "attempted_episodes": len(episodes),
        "kept_episodes": len(kept),
        "transitions": transitions,
        "failed_episode_ids": failed_episode_ids,
        "missing_success_episode_ids": missing_success_episode_ids,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    generate(tyro.cli(Args))


if __name__ == "__main__":
    main()
