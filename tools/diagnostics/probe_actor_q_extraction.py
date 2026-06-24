"""Probe actor imitation and critic action ranking on fixed demo batches.

This script is for offline diagnosis of vision SAC/BC checkpoints.  It samples
transitions from a ManiSkill H5 demo dataset, then reports whether the actor
assigns high likelihood to demo actions and whether a SAC critic ranks demo
actions above policy/random actions.

Example:
    python tools/diagnostics/probe_actor_q_extraction.py \
      --dataset_path datasets/stackcube_state_oracle_rgb_1m.h5 \
      --checkpoint_paths runs/<sac-run>/checkpoints/final.pt \
      --bc_actor_checkpoint_paths runs/<bc-run>/checkpoints/bc_offline_pretrained.pt \
      --max_transitions 1024 --output_json /tmp/actor_q_probe.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import tyro

from rl_garden.algorithms import SAC
from rl_garden.common import seed_everything
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.cli_args import (
    image_encoder_factory_from_args,
    image_keys_from_env,
    vit_sac_kwargs_from_args,
)
from rl_garden.encoders.plain_conv import PlainConv
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


Scope = Literal["all", "success", "both"]


@dataclass
class Args:
    dataset_path: str
    checkpoint_paths: tuple[str, ...] = ()
    bc_actor_checkpoint_paths: tuple[str, ...] = ()
    output_json: Optional[str] = None

    env_id: str = "StackCube-v1"
    obs_mode: str = "rgb"
    include_state: bool = True
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    render_mode: str = "rgb_array"
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    per_camera_rgbd: bool = True
    num_envs: int = 1

    encoder: Literal["plain_conv", "resnet10", "resnet18", "vit"] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "per_key"
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["random_shift", "none"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    plain_conv_weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform"
    plain_conv_last_act: bool = True
    plain_conv_pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten"
    image_keys: Optional[str] = None
    image_augmentation: Literal["none", "random_shift"] = "none"
    image_random_shift_pad: int = 4

    seed: int = 1
    device: str = "auto"
    max_transitions: int = 1024
    scopes: Scope = "both"
    batch_size: int = 256

    gamma: float = 0.8
    tau: float = 0.01
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    ent_coef: float | str = 0.003
    target_entropy: float | str = "auto"
    alpha_tuning: Literal["legacy_exp", "log_alpha", "lagrange_softplus"] = "legacy_exp"
    critic_impl: Literal["vmap", "legacy"] = "vmap"
    n_critics: int = 10
    critic_subsample_size: Optional[int] = 2
    actor_use_layer_norm: bool = True
    critic_use_layer_norm: bool = False
    hidden_dim: int = 256
    actor_hidden_layers: int = 3
    critic_hidden_layers: int = 3
    actor_log_std_min: float = -5.0
    actor_log_std_mode: Literal["clamp", "tanh"] = "clamp"


@dataclass
class DemoBatch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    scope: str
    num_available_transitions: int
    num_selected_transitions: int
    num_success_episodes: int
    num_total_episodes: int


def _natural_key(name: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", name)
    return (
        name[: match.start()] if match else name,
        int(match.group(1)) if match else -1,
    )


def _trajectory_keys(handle: Any) -> list[str]:
    keys = [key for key in handle.keys() if key.startswith("traj_")]
    return sorted(keys, key=_natural_key)


def _dataset_node(group: Any, *names: str) -> Any:
    for name in names:
        if name in group:
            return group[name]
    raise KeyError(f"None of {names!r} found under H5 group {group.name!r}.")


def _episode_success(traj: Any) -> Optional[bool]:
    for key in (
        "final_success",
        "success",
        "success_at_end",
        "success_once",
        "is_success",
    ):
        if key in traj.attrs:
            return bool(np.asarray(traj.attrs[key]).reshape(-1)[-1])
    for group_name in ("infos", "info", "episode"):
        if group_name not in traj:
            continue
        group = traj[group_name]
        for key in ("success_at_end", "success_once", "success", "is_success"):
            if key in group:
                value = np.asarray(group[key])
                return bool(value.reshape(-1)[-1])
    return None


def _read_obs_at(obs_group: Any, index: int) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(np.asarray(value[index]))
        for key, value in obs_group.items()
    }


def _stack_obs(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = samples[0].keys()
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys
    }


def _to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if isinstance(tree, dict):
        return {key: _to_device(value, device) for key, value in tree.items()}
    return tree


def load_demo_batch(
    dataset_path: str | Path,
    *,
    scope: Literal["all", "success"],
    max_transitions: int,
    seed: int,
) -> DemoBatch:
    import h5py

    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, int]] = []
    success_episodes = 0
    total_episodes = 0
    saw_success_metadata = False
    with h5py.File(dataset_path, "r") as handle:
        for traj_key in _trajectory_keys(handle):
            traj = handle[traj_key]
            actions_node = _dataset_node(traj, "actions", "action")
            success = _episode_success(traj)
            if success is not None:
                saw_success_metadata = True
            include = scope == "all" or success is True
            total_episodes += 1
            success_episodes += int(success is True)
            if include:
                candidates.extend((traj_key, i) for i in range(len(actions_node)))

        if scope == "success" and not saw_success_metadata:
            raise ValueError(
                f"No episode success metadata found in {dataset_path}; "
                "cannot build a success-only probe batch."
            )
        if not candidates:
            raise ValueError(f"No transitions selected for scope={scope!r}.")
        num_available = len(candidates)
        if max_transitions > 0 and len(candidates) > max_transitions:
            selected = rng.choice(len(candidates), size=max_transitions, replace=False)
            candidates = [candidates[int(i)] for i in selected]

        obs_samples: list[dict[str, torch.Tensor]] = []
        action_samples: list[torch.Tensor] = []
        for traj_key, index in candidates:
            traj = handle[traj_key]
            obs_group = _dataset_node(traj, "obs", "observations")
            actions_node = _dataset_node(traj, "actions", "action")
            obs_samples.append(_read_obs_at(obs_group, index))
            action_samples.append(
                torch.as_tensor(np.asarray(actions_node[index])).float()
            )

    return DemoBatch(
        obs=_stack_obs(obs_samples),
        actions=torch.stack(action_samples, dim=0),
        scope=scope,
        num_available_transitions=num_available,
        num_selected_transitions=len(candidates),
        num_success_episodes=success_episodes,
        num_total_episodes=total_episodes,
    )


def relative_dormant_ratio(
    activations: torch.Tensor, threshold: float = 0.1
) -> torch.Tensor:
    scores = activations.detach().abs().mean(dim=0)
    denom = scores.mean().clamp_min(1e-12)
    return (scores / denom < threshold).float().mean()


def relative_channel_dormant_ratio(
    feature_map: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Sokar-style dormant ratio over CNN channels, averaging batch and space."""
    scores = feature_map.detach().abs().mean(dim=(0, 2, 3))
    denom = scores.mean().clamp_min(1e-12)
    return (scores / denom < threshold).float().mean()


def _mean_sample_norm(value: torch.Tensor) -> torch.Tensor:
    return value.detach().flatten(1).norm(dim=-1).mean()


class PlainConvActivationProbe:
    """Capture PlainConv bottleneck activations without modifying model code."""

    def __init__(self, features_extractor: nn.Module) -> None:
        self.modules = {
            name: module
            for name, module in features_extractor.named_modules()
            if isinstance(module, PlainConv)
        }
        self.activations: dict[str, dict[str, torch.Tensor]] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> PlainConvActivationProbe:
        for name, module in self.modules.items():
            branch = self.activations.setdefault(name, {})
            self._handles.extend(
                (
                    module.cnn.register_forward_hook(self._capture(branch, "pre_pool")),
                    module.pool.register_forward_hook(self._capture(branch, "pooled")),
                    module.fc.register_forward_hook(self._capture(branch, "encoded")),
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _capture(target: dict[str, torch.Tensor], key: str):
        def hook(_module, _inputs, output):
            target[key] = output

        return hook

    def metrics(self, q_sum: torch.Tensor) -> dict[str, dict[str, Any]]:
        tensors_by_id: dict[int, torch.Tensor] = {}
        for branch in self.activations.values():
            for value in branch.values():
                tensors_by_id.setdefault(id(value), value)
        unique_tensors = list(tensors_by_id.values())
        unique_grads = torch.autograd.grad(
            q_sum,
            unique_tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        grads_by_id = {
            id(value): grad for value, grad in zip(unique_tensors, unique_grads)
        }

        out: dict[str, dict[str, Any]] = {}
        for name, module in self.modules.items():
            branch = self.activations[name]
            pre_pool = branch["pre_pool"]
            pooled = branch["pooled"]
            encoded = branch["encoded"]
            linear = next(
                layer for layer in module.fc.modules() if isinstance(layer, nn.Linear)
            )
            flatten_dim = int(np.prod(pre_pool.shape[1:]))
            out[name] = {
                "pooling": module.pooling,
                "pre_pool_shape": list(pre_pool.shape[1:]),
                "flatten_input_dim": flatten_dim,
                "bottleneck_input_dim": linear.in_features,
                "bottleneck_output_dim": linear.out_features,
                "bottleneck_parameter_count": sum(
                    parameter.numel() for parameter in module.fc.parameters()
                ),
                "bottleneck_compression_ratio": flatten_dim / linear.in_features,
                "pre_pool_feature_norm": _scalar(_mean_sample_norm(pre_pool)),
                "pre_pool_channel_dormant_ratio": _scalar(
                    relative_channel_dormant_ratio(pre_pool)
                ),
                "pooled_feature_norm": _scalar(_mean_sample_norm(pooled)),
                "pooled_dormant_ratio": _scalar(
                    relative_dormant_ratio(pooled.flatten(1))
                ),
                "encoded_feature_norm": _scalar(_mean_sample_norm(encoded)),
                "encoded_dormant_ratio": _scalar(
                    relative_dormant_ratio(encoded.flatten(1))
                ),
                "q_grad_pre_pool_norm": _scalar(
                    _mean_sample_norm(grads_by_id[id(pre_pool)])
                ),
                "q_grad_pooled_norm": _scalar(
                    _mean_sample_norm(grads_by_id[id(pooled)])
                ),
                "q_grad_encoded_norm": _scalar(
                    _mean_sample_norm(grads_by_id[id(encoded)])
                ),
            }
        return out


def _scalar(value: Optional[torch.Tensor | float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_sum_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).sum(dim=-1).mean()


def _min_q(policy: Any, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return policy.q_values_all(features, actions, target=False).min(dim=0).values


def _q_and_grad(
    policy: Any,
    features: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probe_actions = actions.detach().clone().requires_grad_(True)
    q = _min_q(policy, features.detach(), probe_actions)
    grad = torch.autograd.grad(
        q.sum(), probe_actions, retain_graph=False, create_graph=False
    )[0]
    return q.detach(), grad.norm(dim=-1).mean().detach()


def _empty_accumulator() -> dict[str, float]:
    return {
        "n": 0.0,
        "log_prob_demo_sum": 0.0,
        "mse_pi_det_demo_sum": 0.0,
        "mse_pi_sample_demo_sum": 0.0,
        "feature_norm_sum": 0.0,
        "feature_dormant_sum": 0.0,
        "q_demo_sum": 0.0,
        "q_pi_det_sum": 0.0,
        "q_pi_sample_sum": 0.0,
        "q_random_sum": 0.0,
        "grad_demo_sum": 0.0,
        "grad_pi_det_sum": 0.0,
        "critic_hidden_dormant_sum": 0.0,
    }


def _accumulate_plain_conv_metrics(
    accumulator: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    n: float,
) -> None:
    for branch_name, branch_metrics in metrics.items():
        branch_acc = accumulator.setdefault(branch_name, {"n": 0.0})
        branch_acc["n"] += n
        for key, value in branch_metrics.items():
            if isinstance(value, (int, float)):
                branch_acc[key] = branch_acc.get(key, 0.0) + float(value) * n
            else:
                branch_acc.setdefault(key, value)


def _finalize_plain_conv_metrics(
    accumulator: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for branch_name, branch_acc in accumulator.items():
        n = max(float(branch_acc["n"]), 1.0)
        out[branch_name] = {
            key: value / n if isinstance(value, (int, float)) and key != "n" else value
            for key, value in branch_acc.items()
            if key != "n"
        }
    return out


def _finalize(acc: dict[str, float], *, include_q: bool) -> dict[str, Any]:
    n = max(acc["n"], 1.0)
    out: dict[str, Optional[float]] = {
        "n": acc["n"],
        "log_prob_demo_mean": acc["log_prob_demo_sum"] / n,
        "nll_demo_mean": -acc["log_prob_demo_sum"] / n,
        "mse_pi_det_demo": acc["mse_pi_det_demo_sum"] / n,
        "mse_pi_sample_demo": acc["mse_pi_sample_demo_sum"] / n,
        "feature_norm": acc["feature_norm_sum"] / n,
        "feature_dormant_ratio": acc["feature_dormant_sum"] / n,
    }
    if not include_q:
        out.update(
            {
                "q_demo": None,
                "q_pi_det": None,
                "q_pi_sample": None,
                "q_random": None,
                "q_demo_minus_random": None,
                "q_pi_det_minus_demo": None,
                "q_pi_sample_minus_demo": None,
                "grad_norm_demo": None,
                "grad_norm_pi_det": None,
                "critic_hidden_dormant_ratio": None,
            }
        )
        return out

    q_demo = acc["q_demo_sum"] / n
    q_pi_det = acc["q_pi_det_sum"] / n
    q_pi_sample = acc["q_pi_sample_sum"] / n
    q_random = acc["q_random_sum"] / n
    out.update(
        {
            "q_demo": q_demo,
            "q_pi_det": q_pi_det,
            "q_pi_sample": q_pi_sample,
            "q_random": q_random,
            "q_demo_minus_random": q_demo - q_random,
            "q_pi_det_minus_demo": q_pi_det - q_demo,
            "q_pi_sample_minus_demo": q_pi_sample - q_demo,
            "grad_norm_demo": acc["grad_demo_sum"] / n,
            "grad_norm_pi_det": acc["grad_pi_det_sum"] / n,
            "critic_hidden_dormant_ratio": acc["critic_hidden_dormant_sum"] / n,
        }
    )
    return out


def _probe_checkpoint_impl(
    agent: SAC,
    batch: DemoBatch,
    *,
    include_q: bool,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    policy = agent.policy
    policy.eval()
    device = agent.device
    obs = _to_device(batch.obs, device)
    actions = batch.actions.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    acc = _empty_accumulator()
    plain_conv_acc: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        action_low = torch.as_tensor(
            agent.env.single_action_space.low, device=device, dtype=actions.dtype
        )
        action_high = torch.as_tensor(
            agent.env.single_action_space.high, device=device, dtype=actions.dtype
        )

    for start in range(0, actions.shape[0], batch_size):
        stop = min(start + batch_size, actions.shape[0])
        obs_chunk = {key: value[start:stop] for key, value in obs.items()}
        demo_actions = actions[start:stop]
        n = float(demo_actions.shape[0])

        with torch.enable_grad():
            with PlainConvActivationProbe(policy.features_extractor) as conv_probe:
                features_live = policy.extract_features(
                    obs_chunk,
                    stop_gradient=False,
                )
            if conv_probe.activations and include_q:
                q_for_encoder = _min_q(policy, features_live, demo_actions)
                _accumulate_plain_conv_metrics(
                    plain_conv_acc,
                    conv_probe.metrics(q_for_encoder.sum()),
                    n,
                )
            features = features_live.detach()
            actor_features = policy._transform_features_for_actor(features)
            log_prob_demo = policy.actor.evaluate_action_log_prob(
                actor_features, demo_actions
            )
            pi_det = policy.actor.deterministic_action(actor_features)
            pi_sample, _ = policy.actor.action_log_prob(actor_features)

            acc["n"] += n
            acc["log_prob_demo_sum"] += float(log_prob_demo.sum().detach().cpu().item())
            acc["mse_pi_det_demo_sum"] += float(
                _mean_sum_mse(pi_det, demo_actions).detach().cpu().item() * n
            )
            acc["mse_pi_sample_demo_sum"] += float(
                _mean_sum_mse(pi_sample, demo_actions).detach().cpu().item() * n
            )
            acc["feature_norm_sum"] += float(
                features.norm(dim=-1).mean().detach().cpu().item() * n
            )
            acc["feature_dormant_sum"] += float(
                relative_dormant_ratio(features).detach().cpu().item() * n
            )

            if not include_q:
                continue

            random_actions = action_low + (action_high - action_low) * torch.rand(
                demo_actions.shape,
                device=device,
                dtype=demo_actions.dtype,
                generator=generator,
            )
            q_demo, grad_demo = _q_and_grad(policy, features, demo_actions)
            q_pi_det, grad_pi_det = _q_and_grad(policy, features, pi_det)
            q_pi_sample = _min_q(policy, features, pi_sample).detach()
            q_random = _min_q(policy, features, random_actions).detach()

            acc["q_demo_sum"] += float(q_demo.mean().cpu().item() * n)
            acc["q_pi_det_sum"] += float(q_pi_det.mean().cpu().item() * n)
            acc["q_pi_sample_sum"] += float(q_pi_sample.mean().cpu().item() * n)
            acc["q_random_sum"] += float(q_random.mean().cpu().item() * n)
            acc["grad_demo_sum"] += float(grad_demo.cpu().item() * n)
            acc["grad_pi_det_sum"] += float(grad_pi_det.cpu().item() * n)
            if hasattr(policy.critic, "trunk_features_first"):
                trunk = policy.critic.trunk_features_first(features, demo_actions)
                acc["critic_hidden_dormant_sum"] += float(
                    relative_dormant_ratio(trunk).detach().cpu().item() * n
                )

    out = _finalize(acc, include_q=include_q)
    out["plain_conv_branches"] = _finalize_plain_conv_metrics(plain_conv_acc)
    return out


def probe_checkpoint(
    agent: SAC,
    batch: DemoBatch,
    *,
    include_q: bool,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run the offline probe without perturbing process-level training RNG."""
    device = torch.device(agent.device)
    devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [
            torch.cuda.current_device() if device.index is None else device.index
        ]
    with torch.random.fork_rng(devices=devices):
        return _probe_checkpoint_impl(
            agent,
            batch,
            include_q=include_q,
            seed=seed,
            batch_size=batch_size,
        )


def _make_env(args: Args):
    cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=args.render_mode,
        reconfiguration_freq=1,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        per_camera_rgbd=args.per_camera_rgbd,
    )
    return make_maniskill_env(cfg)


def _make_agent(args: Args, env) -> SAC:
    image_keys = image_keys_from_env(env, args)
    hidden = [args.hidden_dim] * args.actor_hidden_layers
    critic_hidden = [args.hidden_dim] * args.critic_hidden_layers
    return SAC(
        env=env,
        eval_env=None,
        buffer_size=1,
        buffer_device="cpu",
        learning_starts=0,
        batch_size=1,
        training_freq=1,
        gamma=args.gamma,
        tau=args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        alpha_tuning=args.alpha_tuning,
        critic_impl=args.critic_impl,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_mode=args.actor_log_std_mode,
        net_arch={"pi": hidden, "qf": critic_hidden},
        seed=args.seed,
        device=args.device,
        logger=None,
        std_log=False,
        eval_freq=0,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_final_checkpoint=False,
        image_keys=image_keys,
        image_encoder_factory=image_encoder_factory_from_args(args),
        image_fusion_mode=args.image_fusion_mode,
        image_augmentation=args.image_augmentation,
        random_shift_pad=args.image_random_shift_pad,
        image_augmentation_seed=args.seed + 1_000_003,
        **vit_sac_kwargs_from_args(args, image_keys),
    )


def _checkpoint_algorithm(path: str, device: torch.device) -> str:
    checkpoint = load_checkpoint_file(path, map_location=device)
    return str(checkpoint.get("metadata", {}).get("algorithm_class", "unknown"))


def _scope_names(scopes: Scope) -> tuple[Literal["all", "success"], ...]:
    if scopes == "both":
        return ("all", "success")
    return (scopes,)


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)
    env = _make_env(args)
    results: list[dict[str, Any]] = []
    try:
        batches = {
            scope: load_demo_batch(
                args.dataset_path,
                scope=scope,
                max_transitions=args.max_transitions,
                seed=args.seed + i,
            )
            for i, scope in enumerate(_scope_names(args.scopes))
        }

        for checkpoint_path in args.checkpoint_paths:
            agent = _make_agent(args, env)
            agent.load(checkpoint_path, load_replay_buffer=False, load_optimizers=False)
            algorithm = _checkpoint_algorithm(checkpoint_path, agent.device)
            for scope, batch in batches.items():
                metrics = probe_checkpoint(
                    agent,
                    batch,
                    include_q=True,
                    seed=args.seed + 10_000,
                    batch_size=args.batch_size,
                )
                results.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_kind": "sac",
                        "algorithm_class": algorithm,
                        "scope": scope,
                        "num_available_transitions": batch.num_available_transitions,
                        "num_selected_transitions": batch.num_selected_transitions,
                        "num_success_episodes": batch.num_success_episodes,
                        "num_total_episodes": batch.num_total_episodes,
                        **metrics,
                    }
                )

        for checkpoint_path in args.bc_actor_checkpoint_paths:
            agent = _make_agent(args, env)
            agent.load_actor_checkpoint(checkpoint_path, strict=True)
            algorithm = _checkpoint_algorithm(checkpoint_path, agent.device)
            for scope, batch in batches.items():
                metrics = probe_checkpoint(
                    agent,
                    batch,
                    include_q=False,
                    seed=args.seed + 20_000,
                    batch_size=args.batch_size,
                )
                results.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_kind": "bc_actor",
                        "algorithm_class": algorithm,
                        "scope": scope,
                        "num_available_transitions": batch.num_available_transitions,
                        "num_selected_transitions": batch.num_selected_transitions,
                        "num_success_episodes": batch.num_success_episodes,
                        "num_total_episodes": batch.num_total_episodes,
                        **metrics,
                    }
                )

        payload = {"args": asdict(args), "results": results}
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.output_json is not None:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
