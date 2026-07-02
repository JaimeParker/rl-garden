"""Probe DrQ-v2 critic action ranking around StackCube release decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import tyro

from rl_garden.algorithms.ddpg import DDPG
from rl_garden.common import seed_everything
from rl_garden.encoders import discover_image_keys
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    dataset_path: str
    checkpoint_path: str
    output_json: str
    pre_event_steps: int = 15
    local_action_samples: int = 256
    local_action_std: float = 0.1
    min_success_decisions: int = 5
    seed: int = 1001
    device: str = "auto"


def decision_index(grasped: np.ndarray, on_cube: np.ndarray) -> tuple[int, str]:
    grasped = np.asarray(grasped, dtype=bool)
    on_cube = np.asarray(on_cube, dtype=bool)
    for index in range(1, len(grasped)):
        if grasped[index - 1] and not grasped[index] and (
            on_cube[index - 1] or on_cube[index]
        ):
            return index, "release"
    candidates = np.flatnonzero(on_cube)
    if len(candidates):
        return int(candidates[0]), "on_cube"
    raise ValueError("Episode has neither an on-cube release nor an on-cube state")


def event_window(on_cube: np.ndarray, pre_event_steps: int) -> tuple[int, range]:
    """Return the first on-cube index and its inclusive pre-event window."""
    if pre_event_steps < 0:
        raise ValueError("pre_event_steps must be non-negative")
    candidates = np.flatnonzero(np.asarray(on_cube, dtype=bool))
    if not len(candidates):
        raise ValueError("Episode has no on-cube state")
    event = int(candidates[0])
    return event, range(max(0, event - pre_event_steps), event + 1)


def gradient_alignment(gradient: torch.Tensor, delta: torch.Tensor) -> float:
    """Cosine between the local Q gradient and behavior-action direction."""
    denominator = gradient.norm() * delta.norm()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(torch.dot(gradient, delta).div(denominator).item())


def _read_obs(group: Any, index: int) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in group.items():
        if hasattr(value, "keys"):
            nested = _read_obs(value, index)
            result[key] = nested  # type: ignore[assignment]
        else:
            result[key] = torch.as_tensor(np.asarray(value[index]))
    return result


def _to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, dict):
        return {key: _to_device(value, device) for key, value in tree.items()}
    return tree.to(device)


def _make_agent(args: Args, env: Any) -> DDPG:
    agent = DDPG(
        env=env,
        eval_env=None,
        buffer_size=1024,
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
        image_keys=discover_image_keys(env.single_observation_space),
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


def _q_all(agent: DDPG, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return agent.policy.q_values_all(features, actions, target=False).squeeze(-1)


def _probe_step(
    agent: DDPG,
    obs: dict[str, torch.Tensor],
    behavior_action: torch.Tensor,
    recorded_actor_mean: torch.Tensor,
    *,
    local_action_samples: int,
    local_action_std: float,
    generator: torch.Generator,
) -> dict[str, float]:
    if local_action_samples < 1:
        raise ValueError("local_action_samples must be positive")
    if local_action_std <= 0:
        raise ValueError("local_action_std must be positive")

    obs = _to_device(obs, agent.device)
    behavior_action = behavior_action.to(agent.device).float()
    recorded_actor_mean = recorded_actor_mean.to(agent.device).float()
    with torch.no_grad():
        features = agent.policy.extract_features(obs)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        actor_mean = agent.policy.actor.deterministic_action(features).squeeze(0)
        pair_actions = torch.stack((behavior_action, actor_mean, recorded_actor_mean))
        pair_q = _q_all(agent, features.expand(3, -1), pair_actions)
        pair_min = pair_q.min(dim=0).values

        noise = torch.randn(
            local_action_samples,
            actor_mean.numel(),
            generator=generator,
            device=agent.device,
        )
        candidates = (actor_mean.unsqueeze(0) + local_action_std * noise).clamp(-1, 1)
        candidates = torch.cat(
            (actor_mean.unsqueeze(0), behavior_action.unsqueeze(0), candidates), dim=0
        )
        candidate_q_all = _q_all(
            agent, features.expand(candidates.shape[0], -1), candidates
        )
        candidate_q = candidate_q_all.min(dim=0).values
        best_index = int(candidate_q.argmax().item())
        actor_rank_fraction = float((candidate_q > candidate_q[0]).float().mean().item())

    action_for_grad = actor_mean.detach().clone().requires_grad_(True)
    q_for_grad_all = _q_all(agent, features.detach(), action_for_grad.unsqueeze(0))
    q_for_grad = q_for_grad_all.min(dim=0).values.sum()
    gradient = torch.autograd.grad(q_for_grad, action_for_grad)[0].detach()
    behavior_delta = behavior_action - actor_mean

    per_critic_advantage = pair_q[:, 0] - pair_q[:, 1]
    return {
        "q_behavior": float(pair_min[0].item()),
        "q_actor": float(pair_min[1].item()),
        "q_recorded_actor": float(pair_min[2].item()),
        "q_behavior_minus_actor": float((pair_min[0] - pair_min[1]).item()),
        "critic_behavior_preference_fraction": float(
            (per_critic_advantage > 0).float().mean().item()
        ),
        "critic_disagreement_behavior": float(
            (pair_q[0, 0] - pair_q[1, 0]).abs().item()
        ),
        "critic_disagreement_actor": float(
            (pair_q[0, 1] - pair_q[1, 1]).abs().item()
        ),
        "action_delta_norm": float(behavior_delta.norm().item()),
        "q_action_grad_norm": float(gradient.norm().item()),
        "q_gradient_behavior_cosine": gradient_alignment(gradient, behavior_delta),
        "local_best_q_minus_actor": float((candidate_q[best_index] - candidate_q[0]).item()),
        "local_best_action_distance": float(
            (candidates[best_index] - actor_mean).norm().item()
        ),
        "actor_rank_fraction": actor_rank_fraction,
    }


def _mean(records: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [float(record[key]) for record in records if key in record]
    return float(np.mean(values)) if values else None


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)
    env = make_maniskill_env(
        ManiSkillEnvConfig(
            env_id="StackCube-v1",
            num_envs=1,
            obs_mode="rgb",
            include_state=True,
            control_mode="pd_joint_delta_pos",
            camera_width=64,
            camera_height=64,
            per_camera_rgbd=True,
            sim_backend="gpu",
            render_backend="gpu",
            reward_mode="normalized_dense",
        )
    )
    agent = _make_agent(args, env)
    generator = torch.Generator(device=agent.device)
    generator.manual_seed(args.seed + 2_000_003)

    import h5py

    records: list[dict[str, Any]] = []
    missing_decisions = 0
    with h5py.File(args.dataset_path, "r") as handle:
        for key in sorted((key for key in handle if key.startswith("traj_")), key=lambda x: int(x[5:])):
            traj = handle[key]
            on_cube = np.asarray(traj["infos/is_cubeA_on_cubeB"])
            try:
                event_index, indices = event_window(on_cube, args.pre_event_steps)
            except ValueError:
                missing_decisions += 1
                continue
            release_index, decision_type = decision_index(
                np.asarray(traj["infos/is_cubeA_grasped"]), on_cube
            )
            for index in indices:
                obs = _read_obs(traj["obs"], index)
                obs = {obs_key: value.unsqueeze(0) for obs_key, value in obs.items()}
                result = _probe_step(
                    agent,
                    obs,
                    torch.as_tensor(np.asarray(traj["actions"][index])),
                    torch.as_tensor(np.asarray(traj["actor_mean_actions"][index])),
                    local_action_samples=args.local_action_samples,
                    local_action_std=args.local_action_std,
                    generator=generator,
                )
                result.update(
                    trajectory=key,
                    label=str(traj.attrs["label"]),
                    step_index=index,
                    event_index=event_index,
                    relative_step=index - event_index,
                    release_index=release_index,
                    decision_type=decision_type,
                )
                records.append(result)
    env.close()

    success_records = [record for record in records if record["label"] == "success"]
    near_records = [record for record in records if record["label"] == "near_success"]
    metric_keys = (
        "q_behavior_minus_actor",
        "critic_behavior_preference_fraction",
        "critic_disagreement_behavior",
        "critic_disagreement_actor",
        "action_delta_norm",
        "q_action_grad_norm",
        "q_gradient_behavior_cosine",
        "local_best_q_minus_actor",
        "local_best_action_distance",
        "actor_rank_fraction",
    )
    relative_steps = sorted({int(record["relative_step"]) for record in records})
    output = {
        "checkpoint": args.checkpoint_path,
        "dataset": args.dataset_path,
        "missing_decisions": missing_decisions,
        "groups": {
            label: {
                "count": len(group),
                **{key: _mean(group, key) for key in metric_keys},
            }
            for label, group in (("success", success_records), ("near_success", near_records))
        },
        "by_relative_step": {
            label: {
                str(relative_step): {
                    "count": len(group),
                    **{metric: _mean(group, metric) for metric in metric_keys},
                }
                for relative_step in relative_steps
                if (
                    group := [
                        record
                        for record in records
                        if record["label"] == label
                        and record["relative_step"] == relative_step
                    ]
                )
            }
            for label in ("success", "near_success")
        },
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["groups"], indent=2), flush=True)
    successful_trajectories = {record["trajectory"] for record in success_records}
    if len(successful_trajectories) < args.min_success_decisions:
        raise SystemExit(
            f"Only {len(successful_trajectories)} successful trajectories were usable; "
            f"need at least {args.min_success_decisions}"
        )


if __name__ == "__main__":
    main()
