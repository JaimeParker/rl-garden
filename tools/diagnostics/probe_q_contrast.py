"""Probe Q-function action contrast from an offline checkpoint and dataset.

Example:
    python tools/diagnostics/probe_q_contrast.py \
      --checkpoint-path runs/<run>/checkpoints/iql_offline_pretrained.pt \
      --offline-dataset-path datasets/pickcube_mixed_500k.h5 \
      --output-json /tmp/q_contrast.json
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import torch
import tyro
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.training.offline._dataset import (
    infer_offline_dataset_specs,
    load_offline_dataset,
)


@dataclass(frozen=True)
class QContrastConfig:
    num_uniform_actions: int = 20
    percentiles: tuple[float, ...] = (10.0, 50.0, 90.0, 99.0)


def summary_stats(
    values: torch.Tensor,
    *,
    prefix: str,
    percentiles: Iterable[float] = (50.0, 90.0, 99.0),
) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"Cannot summarize empty tensor for {prefix!r}.")
    if not torch.isfinite(flat).all():
        raise ValueError(f"Non-finite values found while summarizing {prefix!r}.")

    result = {
        f"{prefix}/mean": float(flat.mean().item()),
        f"{prefix}/std": float(flat.std(unbiased=False).item()),
        f"{prefix}/min": float(flat.min().item()),
        f"{prefix}/max": float(flat.max().item()),
    }
    for percentile in percentiles:
        quantile = torch.quantile(flat, float(percentile) / 100.0)
        result[f"{prefix}/p{int(percentile):02d}"] = float(quantile.item())
    return result


def uniform_actions(
    *,
    batch_size: int,
    num_actions: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}.")
    low = action_low.float()
    high = action_high.float()
    sample = torch.rand(
        batch_size,
        num_actions,
        *low.shape,
        device=low.device,
        generator=generator,
    )
    return low + (high - low) * sample


def compute_action_grad_norm(
    policy: Any,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    ensemble_reduce: str = "min",
) -> torch.Tensor:
    if ensemble_reduce not in {"min", "mean"}:
        raise ValueError(f"Unknown ensemble_reduce={ensemble_reduce!r}.")

    probe_actions = actions.detach().clone().requires_grad_(True)
    features = policy.extract_features(obs, stop_gradient=False)
    q_all = policy.q_values_all(features, probe_actions, target=False)
    q_value = q_all.min(dim=0).values if ensemble_reduce == "min" else q_all.mean(dim=0)
    gradient = torch.autograd.grad(q_value.sum(), probe_actions)[0]
    return gradient.norm(dim=-1)


def compute_q_contrast_metrics(
    policy: Any,
    obs: torch.Tensor,
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    config: QContrastConfig = QContrastConfig(),
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    if not isinstance(obs, torch.Tensor):
        raise TypeError("Q contrast diagnostics require Tensor/Box observations.")
    if obs.ndim < 2:
        raise ValueError(f"Expected batched observations, got shape={tuple(obs.shape)}.")

    batch_size = obs.shape[0]
    actions_uniform = uniform_actions(
        batch_size=batch_size,
        num_actions=config.num_uniform_actions,
        action_low=action_low.to(obs.device),
        action_high=action_high.to(obs.device),
        generator=generator,
    )
    with torch.no_grad():
        policy_actions = policy.predict(obs, deterministic=True)
        features = policy.extract_features(obs, stop_gradient=False)
        q_policy_all = policy.q_values_all(features, policy_actions, target=False)
        flat_uniform = actions_uniform.reshape(batch_size * config.num_uniform_actions, -1)
        obs_repeated = obs.repeat_interleave(config.num_uniform_actions, dim=0)
        uniform_features = policy.extract_features(obs_repeated, stop_gradient=False)
        q_uniform_all = policy.q_values_all(
            uniform_features, flat_uniform, target=False
        ).reshape(-1, batch_size, config.num_uniform_actions)

    reductions = (
        ("min", q_uniform_all.min(dim=0).values, q_policy_all.min(dim=0).values),
        ("mean_ensemble", q_uniform_all.mean(dim=0), q_policy_all.mean(dim=0)),
    )
    metrics: dict[str, float] = {}
    for name, q_uniform, q_policy in reductions:
        values = {
            "var_uniform": q_uniform.var(dim=1, unbiased=False),
            "range_uniform": q_uniform.max(dim=1).values
            - q_uniform.min(dim=1).values,
            "policy_uniform_gap": q_policy - q_uniform.mean(dim=1),
            "q_policy": q_policy,
            "q_uniform_mean": q_uniform.mean(dim=1),
        }
        for metric_name, metric_values in values.items():
            metrics.update(
                summary_stats(
                    metric_values,
                    prefix=f"q_contrast/{name}/{metric_name}",
                    percentiles=config.percentiles,
                )
            )

    flat_uniform_actions = actions_uniform.reshape(
        batch_size * config.num_uniform_actions, -1
    )
    obs_repeated = obs.repeat_interleave(config.num_uniform_actions, dim=0)
    for reduction, output_name in (("min", "min"), ("mean", "mean_ensemble")):
        grad_policy = compute_action_grad_norm(
            policy, obs, policy_actions, ensemble_reduce=reduction
        )
        grad_uniform = compute_action_grad_norm(
            policy, obs_repeated, flat_uniform_actions, ensemble_reduce=reduction
        ).reshape(batch_size, config.num_uniform_actions)
        metrics.update(
            summary_stats(
                grad_policy,
                prefix=f"q_contrast/{output_name}/grad_norm_policy",
                percentiles=config.percentiles,
            )
        )
        metrics.update(
            summary_stats(
                grad_uniform.mean(dim=1),
                prefix=f"q_contrast/{output_name}/grad_norm_uniform",
                percentiles=config.percentiles,
            )
        )
    return metrics


@dataclass
class QContrastArgs:
    checkpoint_path: str | None = None
    dataset_source: Literal["maniskill_h5", "minari"] = "maniskill_h5"
    offline_dataset_path: str | None = None
    offline_num_traj: int | None = None
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: str | None = None
    action_low: float = -1.0
    action_high: float = 1.0
    spec_num_envs: int = 1
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    device: str = "auto"
    probe_size: int = 256
    num_uniform_actions: int = 20
    seed: int = 1
    output_json: str | None = None


_ALGORITHM_FROM_CLASS = {
    "CQL": "cql",
    "OfflineCQL": "cql",
    "CalQL": "calql",
    "OfflineCalQL": "calql",
    "WSRL": "wsrl",
    "IQL": "iql",
}


def _algorithm_components(algorithm: str) -> tuple[type, Callable]:
    if algorithm == "iql":
        from rl_garden.training.offline.iql import IQLArgs, build_iql

        return IQLArgs, build_iql
    if algorithm == "cql":
        from rl_garden.training.offline.cql import CQLArgs, build_cql

        return CQLArgs, build_cql
    if algorithm == "calql":
        from rl_garden.training.offline.calql import CalQLArgs, build_calql

        return CalQLArgs, build_calql
    if algorithm == "wsrl":
        from rl_garden.training.offline.wsrl import WSRLOfflineArgs, build_wsrl

        return WSRLOfflineArgs, build_wsrl
    raise ValueError(f"Unsupported Q contrast algorithm: {algorithm!r}")


def _checkpoint_algorithm(checkpoint: dict[str, Any]) -> str:
    algorithm_class = checkpoint.get("metadata", {}).get("algorithm_class")
    try:
        return _ALGORITHM_FROM_CLASS[algorithm_class]
    except KeyError as exc:
        raise SystemExit(
            "Q contrast supports IQL/CQL/CalQL/WSRL checkpoints; "
            f"got algorithm_class={algorithm_class!r}."
        ) from exc


def _agent_args(
    diagnostic_args: QContrastArgs,
    checkpoint: dict[str, Any],
    args_cls: type,
) -> Any:
    args = args_cls()
    hparams = checkpoint.get("metadata", {}).get("hyperparameters", {})
    for key, value in hparams.items():
        if hasattr(args, key) and key not in {"device", "buffer_device"}:
            setattr(args, key, value)

    for key in (
        "dataset_source",
        "offline_dataset_path",
        "offline_num_traj",
        "reward_scale",
        "reward_bias",
        "success_key",
        "action_low",
        "action_high",
        "spec_num_envs",
        "buffer_size",
        "buffer_device",
        "seed",
    ):
        setattr(args, key, getattr(diagnostic_args, key))
    if hasattr(args, "device"):
        args.device = diagnostic_args.device
    if hasattr(args, "use_compile"):
        args.use_compile = False
    args.log_type = "none"
    args.std_log = False
    args.load_replay_buffer = False
    args.save_replay_buffer = False
    args.save_final_checkpoint = False
    args.checkpoint_freq = 0
    return args


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def main() -> None:
    args = tyro.cli(QContrastArgs)
    if args.checkpoint_path is None:
        raise SystemExit("--checkpoint-path is required.")
    if args.offline_dataset_path is None:
        raise SystemExit("--offline-dataset-path is required.")
    if args.probe_size <= 0:
        raise SystemExit("--probe-size must be positive.")
    if args.num_uniform_actions <= 0:
        raise SystemExit("--num-uniform-actions must be positive.")
    if args.buffer_device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available; falling back to CPU buffer.", stacklevel=2)
        args.buffer_device = "cpu"

    seed_everything(args.seed)
    enable_fast_math()
    checkpoint = load_checkpoint_file(args.checkpoint_path, map_location="cpu")
    algorithm = _checkpoint_algorithm(checkpoint)
    args_cls, build_agent = _algorithm_components(algorithm)
    agent_args = _agent_args(args, checkpoint, args_cls)

    obs_space, action_space = infer_offline_dataset_specs(agent_args)
    if not isinstance(obs_space, spaces.Box):
        raise SystemExit("Q contrast currently supports flat Box/state datasets only.")
    env_spec = OfflineEnvSpec(
        obs_space, action_space, num_envs=agent_args.spec_num_envs
    )
    logger = Logger.create(
        log_type="none",
        log_dir="runs",
        run_name=f"q_contrast__{algorithm}",
        config=None,
    )
    try:
        agent = build_agent(agent_args, env_spec, logger)
        loaded = load_offline_dataset(agent.replay_buffer, agent_args)
        if loaded < args.probe_size:
            raise SystemExit(
                f"Dataset loaded {loaded} transitions, smaller than "
                f"probe_size={args.probe_size}."
            )

        agent.load(
            args.checkpoint_path,
            load_replay_buffer=False,
            load_optimizers=False,
        )
        agent.policy.eval()
        obs = agent.replay_buffer.sample(args.probe_size).obs.to(agent.device)
        metrics = compute_q_contrast_metrics(
            agent.policy,
            obs,
            action_low=torch.as_tensor(action_space.low, device=agent.device),
            action_high=torch.as_tensor(action_space.high, device=agent.device),
            config=QContrastConfig(num_uniform_actions=args.num_uniform_actions),
            generator=_make_generator(agent.device, args.seed),
        )
    finally:
        logger.close()

    payload = {
        "algorithm": algorithm,
        "checkpoint_path": args.checkpoint_path,
        "dataset_source": args.dataset_source,
        "offline_dataset_path": args.offline_dataset_path,
        "loaded_transitions": int(loaded),
        "probe_size": args.probe_size,
        "num_uniform_actions": args.num_uniform_actions,
        "seed": args.seed,
        "device": str(agent.device),
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
