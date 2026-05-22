"""Evaluate offline RL checkpoints (CQL / CalQL / WSRL) with video recording.

Usage::

    CUDA_VISIBLE_DEVICES=3 python tests/eval_offline.py --checkout-path runs/pickcube_calql_offline_700k/checkpoints/calql_offline_pretrained.pt
    
The script auto-detects the algorithm class from checkpoint metadata, builds a
minimal offline agent, injects a real ManiSkill eval env with video recording
(via ``RecordEpisode``), loads the checkpoint weights, and runs evaluation.

Only state-based agents are supported (``obs_mode="state"``). Video output is
written to ``<checkpoint_dir>/eval_videos/`` by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import tyro

from rl_garden.algorithms import CalQL, CQL, OfflineEnvSpec, WSRL
from rl_garden.common import seed_everything
from rl_garden.common.checkpoint import _canonical_algorithm_class, load_checkpoint_file
from rl_garden.common.utils import get_device
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class EvalOfflineArgs:
    checkout_path: str  # required -- path to a .pt checkpoint
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_episodes: int = 50
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    seed: int = 1
    device: str = "auto"
    buffer_device: str = "cpu"
    output_dir: Optional[str] = None
    video_fps: int = 30
    capture_video: bool = True


# Map canonical algorithm class names to their constructors.
_ALGORITHM_REGISTRY: dict[str, type] = {
    "CQL": CQL,
    "CalQL": CalQL,
    "WSRL": WSRL,
}


def _build_agent(
    algorithm_class: str,
    env_spec: OfflineEnvSpec,
    device: torch.device,
    seed: int,
    buffer_device: str,
):
    """Construct a minimal offline agent for evaluation.

    Only the policy/encoder is needed for inference; replay buffers, optimizers,
    and training knobs are left at their defaults and never exercised.
    """
    agent_cls = _ALGORITHM_REGISTRY[algorithm_class]
    return agent_cls(
        env=env_spec,
        device=device,
        seed=seed,
        std_log=False,
        log_freq=0,
        buffer_device=buffer_device,
    )


def main() -> None:
    args = tyro.cli(EvalOfflineArgs)
    seed_everything(args.seed)
    device = get_device(args.device)

    checkpoint_path = Path(args.checkout_path)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    if args.buffer_device == "cuda" and not torch.cuda.is_available():
        print("[eval_offline] CUDA not available; falling back to CPU.")
        args.buffer_device = "cpu"

    # 1. Read checkpoint metadata to auto-detect algorithm class
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    algorithm_class_raw = metadata.get("algorithm_class", None)
    if algorithm_class_raw is None:
        raise SystemExit("Checkpoint metadata missing 'algorithm_class'.")

    algorithm_class = _canonical_algorithm_class(algorithm_class_raw)
    if algorithm_class not in _ALGORITHM_REGISTRY:
        supported = ", ".join(sorted(_ALGORITHM_REGISTRY))
        raise SystemExit(
            f"Unsupported algorithm class '{algorithm_class}' "
            f"(raw: '{algorithm_class_raw}'). Supported: {supported}"
        )

    print(f"[eval_offline] checkpoint = {checkpoint_path}")
    print(
        f"[eval_offline] algorithm = {algorithm_class} "
        f"(raw = {algorithm_class_raw})"
    )

    # 2. Build eval env with video recording
    output_dir = args.output_dir or str(checkpoint_path.parent / "eval_videos")
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reconfiguration_freq=1,
        render_mode="rgb_array",
        record_dir=output_dir,
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_episodes,
    )
    eval_env = make_maniskill_env(eval_cfg)
    print(
        f"[eval_offline] env = {args.env_id}  "
        f"num_envs={args.num_envs}  video_dir={output_dir}"
    )

    # 3. Build minimal agent & inject eval env
    env_spec = OfflineEnvSpec(
        observation_space=eval_env.single_observation_space,
        action_space=eval_env.single_action_space,
        num_envs=args.num_envs,
    )
    agent = _build_agent(algorithm_class, env_spec, device, args.seed, args.buffer_device)
    # _setup_model() allocates policy, replay buffer, and optimizers.
    agent._setup_model()

    # Inject eval env after construction (offline algorithm constructors
    # accept it but WSRL derives from the rollout path; setting it
    # post-init is the single-slot approach that works for all three).
    agent.eval_env = eval_env
    agent.num_eval_steps = args.num_eval_episodes

    # 4. Load checkpoint weights (no replay buffer, no optimizers)
    agent.load(
        checkpoint_path,
        strict=False,
        load_replay_buffer=False,
        load_optimizers=False,
    )
    print("[eval_offline] checkpoint loaded successfully.")

    # 5. Run evaluation
    print(f"[eval_offline] evaluating for {args.num_eval_episodes} episodes ...")
    metrics = agent._evaluate()

    # 6. Report
    print("\n=== Evaluation Results ===")
    # Surface the most important metrics first.
    headline_keys = ("success_at_end", "success_once", "return")
    for key in headline_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    for key, value in sorted(metrics.items()):
        if key not in headline_keys:
            print(f"  {key}: {value:.4f}")
    print(f"\nVideo saved to: {output_dir}")

    eval_env.close()


if __name__ == "__main__":
    main()
