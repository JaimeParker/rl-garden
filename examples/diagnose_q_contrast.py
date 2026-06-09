"""Offline Q contrast diagnostics for trained IQL/CQL/CalQL checkpoints.

This entrypoint deliberately does not create a ManiSkill environment. It loads
a flat offline H5 dataset, samples states from the replay buffer, and measures
how sharply the learned Q function varies over actions around those states.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from gymnasium import spaces

from pretrain_offline import build_offline_agent
from rl_garden.algorithms import OfflineEnvSpec, infer_specs_from_h5
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.cli_args import OfflinePretrainArgs
from rl_garden.common.q_contrast import QContrastConfig, compute_q_contrast_metrics


@dataclass
class QContrastArgs(OfflinePretrainArgs):
    checkpoint_path: str | None = None
    probe_size: int = 256
    num_uniform_actions: int = 20
    output_json: str | None = None
    log_type: str = "none"
    std_log: bool = True
    load_replay_buffer: bool = False


_ALGORITHM_FROM_CLASS = {
    "CQL": "cql",
    "OfflineCQL": "cql",
    "CalQL": "calql",
    "OfflineCalQL": "calql",
    "WSRL": "wsrl",
    "IQL": "iql",
}


def _checkpoint_algorithm(checkpoint: dict[str, Any]) -> str:
    algorithm_class = checkpoint.get("metadata", {}).get("algorithm_class")
    try:
        return _ALGORITHM_FROM_CLASS[algorithm_class]
    except KeyError as exc:
        raise SystemExit(
            "Q contrast diagnostics support IQL/CQL/CalQL/WSRL checkpoints; "
            f"got algorithm_class={algorithm_class!r}."
        ) from exc


def _apply_checkpoint_hparams(args: QContrastArgs, checkpoint: dict[str, Any]) -> None:
    """Use checkpoint hyperparameters for model-shape-affecting arguments."""
    hparams = checkpoint.get("metadata", {}).get("hyperparameters", {})
    for key, value in hparams.items():
        if not hasattr(args, key):
            continue
        if key in {"device", "buffer_device"}:
            continue
        setattr(args, key, value)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def main() -> None:
    args = tyro.cli(QContrastArgs)
    if args.checkpoint_path is None:
        raise SystemExit("--checkpoint_path is required.")
    if args.offline_dataset_path is None:
        raise SystemExit("--offline_dataset_path is required.")
    if args.probe_size <= 0:
        raise SystemExit("--probe_size must be positive.")
    if args.num_uniform_actions <= 0:
        raise SystemExit("--num_uniform_actions must be positive.")

    seed_everything(args.seed)
    enable_fast_math()

    checkpoint = load_checkpoint_file(args.checkpoint_path, map_location="cpu")
    algorithm = _checkpoint_algorithm(checkpoint)
    args.algorithm = algorithm
    args.agent = None
    _apply_checkpoint_hparams(args, checkpoint)
    args.load_checkpoint = args.checkpoint_path
    args.load_replay_buffer = False
    args.log_type = "none"
    args.save_final_checkpoint = False
    args.checkpoint_freq = 0

    obs_space, action_space = infer_specs_from_h5(
        args.offline_dataset_path,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    if not isinstance(obs_space, spaces.Box):
        raise SystemExit(
            "Q contrast diagnostics currently support flat Box/state H5 datasets only."
        )
    env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=args.spec_num_envs)
    logger = Logger.create(
        log_type="none",
        log_dir=args.log_dir,
        run_name=f"q_contrast__{algorithm}__{int(time.time())}",
        config=None,
    )
    agent = build_offline_agent(args, env_spec, logger, algorithm)
    loaded = load_maniskill_h5_to_replay_buffer(
        agent.replay_buffer,
        args.offline_dataset_path,
        num_traj=args.offline_num_traj,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        success_key=args.success_key,
    )
    if loaded < args.probe_size:
        raise SystemExit(
            f"Dataset loaded {loaded} transitions, smaller than probe_size={args.probe_size}."
        )

    agent.load(args.checkpoint_path, load_replay_buffer=False, load_optimizers=False)
    agent.policy.eval()

    batch = agent.replay_buffer.sample(args.probe_size)
    obs = batch.obs.to(agent.device)
    action_low = torch.as_tensor(action_space.low, device=agent.device)
    action_high = torch.as_tensor(action_space.high, device=agent.device)
    generator = _make_generator(agent.device, args.seed)

    metrics = compute_q_contrast_metrics(
        agent.policy,
        obs,
        action_low=action_low,
        action_high=action_high,
        config=QContrastConfig(num_uniform_actions=args.num_uniform_actions),
        generator=generator,
    )
    payload: dict[str, Any] = {
        "algorithm": algorithm,
        "checkpoint_path": str(args.checkpoint_path),
        "offline_dataset_path": str(args.offline_dataset_path),
        "loaded_transitions": int(loaded),
        "probe_size": int(args.probe_size),
        "num_uniform_actions": int(args.num_uniform_actions),
        "seed": int(args.seed),
        "device": str(agent.device),
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    logger.close()


if __name__ == "__main__":
    main()
