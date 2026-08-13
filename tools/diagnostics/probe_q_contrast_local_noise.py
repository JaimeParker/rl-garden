"""Probe local Q contrast around offline and actor actions.

This diagnostic samples states/actions from an offline dataset, keeps each state
fixed, perturbs an anchor action at fixed L2 radii, and reports how sharply the
checkpoint critic's min-Q changes in that local action neighborhood. For state
datasets, it also reports actor/replay-neighborhood Boltzmann concentration
metrics over deterministic/stochastic actor and replay-neighbor candidates.

Example:
    python tools/diagnostics/probe_q_contrast_local_noise.py \
      --algorithm calql \
      --dataset_path datasets/pickcube_mixed_500k.h5 \
      --checkpoint_path runs/calql_offline_pretrain__1__1779864949/checkpoints/calql_offline_pretrained.pt \
      --max_transitions 4096 \
      --output_json /tmp/q_contrast_pickcube_calql.json
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.common import seed_everything
from rl_garden.training._dataset import infer_offline_dataset_specs
from rl_garden.training.offline.calql import CalQLArgs, build_calql
from rl_garden.training.offline.iql import IQLArgs, build_iql

Algorithm = Literal["calql", "iql"]
AnchorGroup = Literal["dataset_action", "actor_det_action"]
DatasetBackend = Literal["h5", "d4rl_legacy"]
CandidateGroup = Literal["actor_stochastic", "replay_neighbor"]


@dataclass
class Args:
    algorithm: Algorithm
    dataset_path: str
    checkpoint_path: str
    dataset_backend: DatasetBackend = "h5"
    config_path: str | None = None
    output_json: str | None = None

    max_transitions: int = 4096
    batch_size: int = 256
    num_noisy_actions: int = 64
    num_actor_candidates: int = 64
    num_replay_neighbors: int = 64
    radii: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
    boltzmann_temperatures: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    anchor_groups: tuple[AnchorGroup, ...] = ("dataset_action", "actor_det_action")
    candidate_groups: tuple[CandidateGroup, ...] = ("actor_stochastic", "replay_neighbor")
    seed: int = 1
    device: str = "auto"

    # Defaults match the PickCube IQL/Cal-QL 1m runs found on 6017-nofwd.
    gamma: float = 0.8
    tau: float = 0.005
    action_low: float = -1.0
    action_high: float = 1.0
    n_critics: int = 10
    critic_subsample_size: int = 2
    actor_use_layer_norm: bool = True
    critic_use_layer_norm: bool = True
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    include_state: bool = True
    encoder: Literal["plain_conv", "resnet10", "resnet18", "vit"] = "plain_conv"
    encoder_features_dim: int = 256
    per_camera_rgbd: bool = False
    obs_mode: str = "rgb"
    camera_width: int | None = 64
    camera_height: int | None = 64
    plain_conv_pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten"
    plain_conv_weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform"
    plain_conv_last_act: bool = True
    image_augmentation: Literal["none", "random_shift"] = "none"
    image_random_shift_pad: int = 4
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["random_shift", "none"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: str | None = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False

    # IQL uses exp(advantage * temperature), so Q-space softmax uses 1 / temperature.
    # Cal-QL uses the loaded SAC alpha as its Q-space denominator.
    temperature: float = 3.0


@dataclass
class DemoBatch:
    obs: Any
    actions: torch.Tensor
    num_available_transitions: int
    num_selected_transitions: int
    num_total_episodes: int
    flat_obs: torch.Tensor | None = None


def _natural_key(name: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", name)
    return (
        name[: match.start()] if match else name,
        int(match.group(1)) if match else -1,
    )


def _trajectory_keys(handle: Any) -> list[str]:
    keys = [key for key in handle if key.startswith("traj_")]
    return sorted(keys, key=_natural_key)


def _dataset_node(group: Any, *names: str) -> Any:
    for name in names:
        if name in group:
            return group[name]
    raise KeyError(f"None of {names!r} found under H5 group {group.name!r}.")


def _read_obs_at(obs_group: Any, index: int) -> Any:
    if hasattr(obs_group, "items"):
        return {
            key: torch.as_tensor(np.asarray(value[index]))
            for key, value in obs_group.items()
        }
    return torch.as_tensor(np.asarray(obs_group[index]))


def _stack_obs(samples: list[Any]) -> Any:
    first = samples[0]
    if isinstance(first, dict):
        return {
            key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in first
        }
    return torch.stack(samples, dim=0)


def _slice_obs(obs: Any, start: int, stop: int) -> Any:
    if isinstance(obs, dict):
        return {key: value[start:stop] for key, value in obs.items()}
    return obs[start:stop]


def _to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if isinstance(tree, dict):
        return {key: _to_device(value, device) for key, value in tree.items()}
    return tree


def _flatten_obs_for_neighbors(obs: Any) -> torch.Tensor | None:
    if isinstance(obs, torch.Tensor) and obs.ndim == 2:
        return obs.float()
    if isinstance(obs, dict):
        parts: list[torch.Tensor] = []
        for value in obs.values():
            if not isinstance(value, torch.Tensor):
                return None
            if value.ndim < 2:
                return None
            parts.append(value.reshape(value.shape[0], -1).float())
        if parts:
            return torch.cat(parts, dim=1)
    return None


def load_demo_batch(
    dataset_path: str | Path,
    *,
    max_transitions: int,
    seed: int,
    dataset_backend: DatasetBackend = "h5",
) -> DemoBatch:
    if dataset_backend == "d4rl_legacy":
        return load_d4rl_legacy_demo_batch(
            str(dataset_path), max_transitions=max_transitions, seed=seed
        )
    if dataset_backend != "h5":
        raise ValueError(f"Unsupported dataset_backend: {dataset_backend!r}")

    import h5py

    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, int]] = []
    with h5py.File(dataset_path, "r") as handle:
        traj_keys = _trajectory_keys(handle)
        for traj_key in traj_keys:
            actions_node = _dataset_node(handle[traj_key], "actions", "action")
            candidates.extend((traj_key, i) for i in range(len(actions_node)))
        if not candidates:
            raise ValueError(f"No transitions found in {dataset_path}.")
        num_available = len(candidates)
        if max_transitions > 0 and len(candidates) > max_transitions:
            selected = rng.choice(len(candidates), size=max_transitions, replace=False)
            candidates = [candidates[int(i)] for i in selected]

        obs_samples: list[Any] = []
        action_samples: list[torch.Tensor] = []
        for traj_key, index in candidates:
            traj = handle[traj_key]
            obs_group = _dataset_node(traj, "obs", "observations")
            actions_node = _dataset_node(traj, "actions", "action")
            obs_samples.append(_read_obs_at(obs_group, index))
            action_samples.append(
                torch.as_tensor(np.asarray(actions_node[index])).float()
            )

    obs = _stack_obs(obs_samples)
    return DemoBatch(
        obs=obs,
        actions=torch.stack(action_samples, dim=0),
        num_available_transitions=num_available,
        num_selected_transitions=len(candidates),
        num_total_episodes=len(traj_keys),
        flat_obs=_flatten_obs_for_neighbors(obs),
    )


def load_d4rl_legacy_demo_batch(
    env_id: str,
    *,
    max_transitions: int,
    seed: int,
) -> DemoBatch:
    from rl_garden.buffers.d4rl_legacy_dataset import _make_legacy_env

    rng = np.random.default_rng(seed)
    env = _make_legacy_env(env_id)
    try:
        raw = env.get_dataset()
        observations = np.asarray(raw["observations"], dtype=np.float32)
        actions = np.asarray(raw["actions"], dtype=np.float32)
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "Legacy D4RL observations/actions length mismatch: "
                f"{observations.shape[0]} vs {actions.shape[0]}."
            )
        num_available = int(actions.shape[0])
        indices = np.arange(num_available)
        if max_transitions > 0 and num_available > max_transitions:
            indices = rng.choice(num_available, size=max_transitions, replace=False)
        selected_obs = torch.as_tensor(observations[indices]).float()
        selected_actions = torch.as_tensor(actions[indices]).float()
        timeouts = np.asarray(raw.get("timeouts", np.zeros(num_available)), dtype=bool)
        terminals = np.asarray(raw.get("terminals", np.zeros(num_available)), dtype=bool)
        num_episodes = int(np.count_nonzero(timeouts | terminals))
    finally:
        env.close()

    return DemoBatch(
        obs=selected_obs,
        actions=selected_actions,
        num_available_transitions=num_available,
        num_selected_transitions=int(selected_actions.shape[0]),
        num_total_episodes=max(num_episodes, 1),
        flat_obs=selected_obs,
    )


def fixed_radius_noise(
    actions: torch.Tensor,
    *,
    radius: float,
    n: int,
    low: torch.Tensor,
    high: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    noise = torch.randn(
        actions.shape[0],
        n,
        actions.shape[-1],
        device=actions.device,
        dtype=actions.dtype,
        generator=generator,
    )
    noise = radius * noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    raw = actions[:, None, :] + noise
    noisy = raw.clamp(low.view(1, 1, -1), high.view(1, 1, -1))
    delta = noisy - actions[:, None, :]
    effective_radius = delta.norm(dim=-1)
    clip_fraction = (raw != noisy).any(dim=-1).float()
    return noisy, effective_radius, clip_fraction


def _min_q(policy: Any, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return policy.q_values_all(features, actions, target=False).min(dim=0).values.squeeze(-1)


def _actor_input(policy: Any, features: torch.Tensor) -> torch.Tensor:
    if hasattr(policy, "_transform_features_for_actor"):
        return policy._transform_features_for_actor(features)
    return features


def deterministic_actor_action(policy: Any, features: torch.Tensor) -> torch.Tensor:
    return policy.actor.deterministic_action(_actor_input(policy, features))


def sample_actor_actions(
    policy: Any,
    features: torch.Tensor,
    *,
    n: int,
) -> torch.Tensor:
    if hasattr(policy, "_sample_actor_actions"):
        actions, _log_prob = policy._sample_actor_actions(features, n)
        return actions
    actor_input = _actor_input(policy, features)
    parts: list[torch.Tensor] = []
    for _ in range(n):
        action, _log_prob = policy.actor.action_log_prob(actor_input)
        parts.append(action)
    return torch.stack(parts, dim=1)


def action_grad_norm(
    policy: Any,
    features: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    probe_actions = actions.detach().clone().requires_grad_(True)
    q = _min_q(policy, features.detach(), probe_actions).sum()
    grad = torch.autograd.grad(
        q,
        probe_actions,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    return grad.norm(dim=-1).detach()


def local_q_contrast_samples(
    policy: Any,
    features: torch.Tensor,
    anchor_actions: torch.Tensor,
    *,
    radius: float,
    num_noisy_actions: int,
    denominator: float,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    noisy, effective_radius, clip_fraction = fixed_radius_noise(
        anchor_actions,
        radius=radius,
        n=num_noisy_actions,
        low=action_low,
        high=action_high,
        generator=generator,
    )
    batch = anchor_actions.shape[0]
    flat_features = (
        features[:, None, :]
        .expand(batch, num_noisy_actions, features.shape[-1])
        .reshape(batch * num_noisy_actions, features.shape[-1])
    )
    flat_noisy = noisy.reshape(batch * num_noisy_actions, anchor_actions.shape[-1])
    q_anchor = _min_q(policy, features, anchor_actions)
    q_noisy = _min_q(policy, flat_features, flat_noisy).reshape(
        batch, num_noisy_actions
    )
    abs_dq_each = (q_noisy - q_anchor[:, None]).abs()
    q_drop_each = (q_anchor[:, None] - q_noisy).clamp_min(0.0)
    logits = torch.cat([q_anchor[:, None], q_noisy], dim=1) / max(denominator, 1e-12)
    weights = torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)
    ess = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
    grad_norm = action_grad_norm(policy, features, anchor_actions)
    return {
        "q_anchor": q_anchor.detach(),
        "q_noisy_mean": q_noisy.mean(dim=1).detach(),
        "q_drop": q_drop_each.mean(dim=1).detach(),
        "abs_dq": abs_dq_each.mean(dim=1).detach(),
        "local_lipschitz": (
            abs_dq_each / effective_radius.clamp_min(1e-12)
        ).mean(dim=1).detach(),
        "dq_over_denominator": (
            abs_dq_each / max(denominator, 1e-12)
        ).mean(dim=1).detach(),
        "local_ess": ess.detach(),
        "local_entropy": entropy.detach(),
        "max_weight": weights.max(dim=1).values.detach(),
        "anchor_top1": (q_anchor >= q_noisy.max(dim=1).values).float().detach(),
        "action_grad_norm": grad_norm,
        "effective_radius": effective_radius.mean(dim=1).detach(),
        "clip_fraction": clip_fraction.mean(dim=1).detach(),
    }


def replay_neighbor_actions(
    flat_obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    start: int,
    stop: int,
    n: int,
) -> torch.Tensor:
    if flat_obs is None:
        raise ValueError("replay_neighbor candidates require flat_obs.")
    query = flat_obs[start:stop].float()
    reference = flat_obs.float()
    distances = torch.cdist(query, reference)
    k = min(max(int(n), 1), reference.shape[0])
    indices = torch.topk(distances, k=k, dim=1, largest=False).indices
    return actions[indices]


def candidate_q_contrast_samples(
    policy: Any,
    features: torch.Tensor,
    actor_det_actions: torch.Tensor,
    candidate_actions: torch.Tensor,
    *,
    denominator: float,
    temperature_multipliers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    batch, num_candidates, action_dim = candidate_actions.shape
    flat_features = (
        features[:, None, :]
        .expand(batch, num_candidates, features.shape[-1])
        .reshape(batch * num_candidates, features.shape[-1])
    )
    flat_candidates = candidate_actions.reshape(batch * num_candidates, action_dim)
    q_candidates = _min_q(policy, flat_features, flat_candidates).reshape(
        batch, num_candidates
    )
    q_actor = _min_q(policy, features, actor_det_actions)
    best_index = q_candidates.argmax(dim=1)
    best_actions = candidate_actions[
        torch.arange(batch, device=features.device), best_index
    ]
    best_q = q_candidates[torch.arange(batch, device=features.device), best_index]
    actor_to_best_dist = (actor_det_actions - best_actions).norm(dim=-1)
    out = {
        "q_actor": q_actor.detach(),
        "q_candidate_mean": q_candidates.mean(dim=1).detach(),
        "q_candidate_max": best_q.detach(),
        "q_best_minus_actor": (best_q - q_actor).detach(),
        "actor_to_best_dist": actor_to_best_dist.detach(),
        "actor_top1": (q_actor >= best_q).float().detach(),
    }
    for multiplier in temperature_multipliers:
        temperature = max(float(denominator) * float(multiplier), 1e-12)
        weights = torch.softmax(q_candidates / temperature, dim=1)
        suffix = str(multiplier).replace(".", "p")
        out[f"ess_x{suffix}"] = (
            1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)
        ).detach()
        out[f"max_weight_x{suffix}"] = weights.max(dim=1).values.detach()
    return out


def summarize_samples(samples: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, parts in samples.items():
        values = torch.cat(parts, dim=0).float()
        out[f"{key}_mean"] = float(values.mean().item())
        if key in {
            "q_drop",
            "abs_dq",
            "local_lipschitz",
            "dq_over_denominator",
            "action_grad_norm",
            "q_best_minus_actor",
            "actor_to_best_dist",
        }:
            out[f"{key}_p50"] = float(torch.quantile(values, 0.5).item())
            out[f"{key}_p90"] = float(torch.quantile(values, 0.9).item())
        if key == "action_grad_norm":
            out[f"{key}_p99"] = float(torch.quantile(values, 0.99).item())
        if key == "local_ess" or key.startswith("ess_x"):
            out[f"{key}_p10"] = float(torch.quantile(values, 0.1).item())
        if key == "max_weight" or key.startswith("max_weight_x"):
            out[f"{key}_p90"] = float(torch.quantile(values, 0.9).item())
    return out


def _apply_mapping_to_args(args: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, Mapping):
            _apply_mapping_to_args(current, value)
        else:
            setattr(args, key, value)


def _config_args(path: Path) -> Mapping[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    values = config.get("inputs", config.get("args", config))
    if not isinstance(values, Mapping):
        raise TypeError(f"Config {path} does not contain an args mapping.")
    return values


def _default_config_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).parent.parent / "config.json"


def _config_path_for_args(args: Args) -> Path:
    if args.config_path is not None:
        return Path(args.config_path)
    return _default_config_path(args.checkpoint_path)


def _args_for_algorithm(args: Args) -> Any:
    algo_args = CalQLArgs() if args.algorithm == "calql" else IQLArgs()
    config_path = _config_path_for_args(args)
    has_config = config_path.exists()
    if has_config:
        _apply_mapping_to_args(algo_args, _config_args(config_path))
        algo_args.offline_dataset = args.dataset_path
        if hasattr(algo_args, "device"):
            algo_args.device = args.device
    else:
        for key, value in asdict(args).items():
            if key == "dataset_path":
                algo_args.offline_dataset = value
            elif key in {"checkpoint_path", "output_json", "config_path"}:
                continue
            elif hasattr(algo_args, key):
                setattr(algo_args, key, value)
    algo_args.dataset_backend = args.dataset_backend
    algo_args.buffer_device = "cpu"
    algo_args.batch_size = max(1, min(args.batch_size, 256))
    algo_args.save_replay_buffer = False
    algo_args.std_log = False
    algo_args.eval_freq = 0
    return algo_args


def _resolved_algorithm_summary(args: Args, algo_args: Any) -> dict[str, Any]:
    keys = (
        "offline_dataset",
        "dataset_backend",
        "obs_mode",
        "gamma",
        "n_critics",
        "critic_subsample_size",
        "actor_use_layer_norm",
        "critic_use_layer_norm",
        "hidden_dim",
        "actor_hidden_layers",
        "critic_hidden_layers",
        "kernel_init",
        "reward_scale",
        "reward_bias",
        "device",
        "batch_size",
        "buffer_device",
    )
    config_path = _config_path_for_args(args)
    summary = {
        key: getattr(algo_args, key)
        for key in keys
        if hasattr(algo_args, key)
    }
    summary["config_path_used"] = str(config_path) if config_path.exists() else None
    return summary


def _build_agent_from_algorithm_args(
    algorithm: Algorithm,
    algo_args: Any,
    checkpoint_path: str,
) -> Any:
    obs_space, action_space = infer_offline_dataset_specs(algo_args)
    env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=1)
    if algorithm == "calql":
        agent = build_calql(algo_args, env_spec, logger=None, eval_env=None)
    else:
        agent = build_iql(algo_args, env_spec, logger=None, eval_env=None)
    agent.load(checkpoint_path, load_replay_buffer=False, load_optimizers=False)
    agent.policy.eval()
    return agent


def _build_agent(args: Args) -> Any:
    algo_args = _args_for_algorithm(args)
    return _build_agent_from_algorithm_args(args.algorithm, algo_args, args.checkpoint_path)


def _denominator(agent: Any, args: Args) -> tuple[str, float]:
    if args.algorithm == "calql" and hasattr(agent, "_current_alpha"):
        return "alpha", float(agent._current_alpha().detach().cpu().item())
    return "inverse_iql_temperature", 1.0 / float(args.temperature)


def _validate_args(args: Args) -> None:
    if args.max_transitions == 0 or args.max_transitions < -1:
        raise ValueError("--max-transitions must be positive or -1 for all transitions.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_noisy_actions <= 0:
        raise ValueError("--num-noisy-actions must be positive.")
    if args.num_actor_candidates <= 0:
        raise ValueError("--num-actor-candidates must be positive.")
    if args.num_replay_neighbors <= 0:
        raise ValueError("--num-replay-neighbors must be positive.")
    if not args.radii or any(radius <= 0 for radius in args.radii):
        raise ValueError("--radii must contain only positive values.")
    if not args.boltzmann_temperatures or any(
        temperature <= 0 for temperature in args.boltzmann_temperatures
    ):
        raise ValueError("--boltzmann-temperatures must contain only positive values.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")


def probe_agent(agent: Any, batch: DemoBatch, args: Args) -> dict[str, Any]:
    device = torch.device(agent.device)
    obs = _to_device(batch.obs, device)
    actions = batch.actions.to(device)
    flat_obs = batch.flat_obs.to(device) if batch.flat_obs is not None else None
    action_low = torch.full(
        (actions.shape[-1],), args.action_low, dtype=actions.dtype, device=device
    )
    action_high = torch.full(
        (actions.shape[-1],), args.action_high, dtype=actions.dtype, device=device
    )
    denom_name, denom = _denominator(agent, args)
    generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
    local_noise_results: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []

    for anchor_group in args.anchor_groups:
        acc_by_radius: dict[float, dict[str, list[torch.Tensor]]] = {
            radius: {} for radius in args.radii
        }
        for start in range(0, actions.shape[0], args.batch_size):
            stop = min(start + args.batch_size, actions.shape[0])
            obs_chunk = _slice_obs(obs, start, stop)
            dataset_actions = actions[start:stop]
            with torch.no_grad():
                features = agent.policy.extract_features(obs_chunk, stop_gradient=True)
                if anchor_group == "dataset_action":
                    anchor_actions = dataset_actions
                else:
                    anchor_actions = deterministic_actor_action(agent.policy, features)
            for radius in args.radii:
                acc = acc_by_radius[radius]
                samples = local_q_contrast_samples(
                    agent.policy,
                    features.detach(),
                    anchor_actions.detach(),
                    radius=radius,
                    num_noisy_actions=args.num_noisy_actions,
                    denominator=denom,
                    action_low=action_low,
                    action_high=action_high,
                    generator=generator,
                )
                for key, value in samples.items():
                    acc.setdefault(key, []).append(value.cpu())
        for radius, acc in acc_by_radius.items():
            local_noise_results.append(
                {
                    "anchor_group": anchor_group,
                    "radius": radius,
                    "num_states": batch.num_selected_transitions,
                    "num_noisy_actions": args.num_noisy_actions,
                    "action_dim": int(actions.shape[-1]),
                    "denominator_name": denom_name,
                    "denominator_value": denom,
                    **summarize_samples(acc),
                }
            )
    for candidate_group in args.candidate_groups:
        if candidate_group == "replay_neighbor" and flat_obs is None:
            continue
        acc: dict[str, list[torch.Tensor]] = {}
        num_candidates = (
            args.num_actor_candidates
            if candidate_group == "actor_stochastic"
            else min(args.num_replay_neighbors, actions.shape[0])
        )
        for start in range(0, actions.shape[0], args.batch_size):
            stop = min(start + args.batch_size, actions.shape[0])
            obs_chunk = _slice_obs(obs, start, stop)
            with torch.no_grad():
                features = agent.policy.extract_features(obs_chunk, stop_gradient=True)
                actor_det = deterministic_actor_action(agent.policy, features)
                if candidate_group == "actor_stochastic":
                    candidate_actions = sample_actor_actions(
                        agent.policy, features, n=args.num_actor_candidates
                    )
                else:
                    assert flat_obs is not None
                    candidate_actions = replay_neighbor_actions(
                        flat_obs,
                        actions,
                        start=start,
                        stop=stop,
                        n=args.num_replay_neighbors,
                    )
            samples = candidate_q_contrast_samples(
                agent.policy,
                features.detach(),
                actor_det.detach(),
                candidate_actions.detach(),
                denominator=denom,
                temperature_multipliers=args.boltzmann_temperatures,
            )
            for key, value in samples.items():
                acc.setdefault(key, []).append(value.cpu())
        candidate_results.append(
            {
                "candidate_group": candidate_group,
                "num_states": batch.num_selected_transitions,
                "num_candidate_actions": int(num_candidates),
                "action_dim": int(actions.shape[-1]),
                "denominator_name": denom_name,
                "denominator_value": denom,
                "temperature_multipliers": list(args.boltzmann_temperatures),
                **summarize_samples(acc),
            }
        )
    return {
        "checkpoint_path": args.checkpoint_path,
        "algorithm": args.algorithm,
        "dataset_path": args.dataset_path,
        "dataset_backend": args.dataset_backend,
        "num_available_transitions": batch.num_available_transitions,
        "num_selected_transitions": batch.num_selected_transitions,
        "num_total_episodes": batch.num_total_episodes,
        "results": local_noise_results,
        "candidate_results": candidate_results,
    }


def main() -> None:
    args = tyro.cli(Args)
    _validate_args(args)
    seed_everything(args.seed)
    batch = load_demo_batch(
        args.dataset_path,
        max_transitions=args.max_transitions,
        seed=args.seed,
        dataset_backend=args.dataset_backend,
    )
    algo_args = _args_for_algorithm(args)
    agent = _build_agent_from_algorithm_args(
        args.algorithm, algo_args, args.checkpoint_path
    )
    payload = {
        "args": asdict(args),
        "resolved_algorithm_args": _resolved_algorithm_summary(args, algo_args),
        **probe_agent(agent, batch, args),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
