"""Probe camera usage for a trained SAC RGBD checkpoint.

The script runs deterministic evaluation while applying counterfactual camera
ablations to the observations seen by the policy. It is intended to answer a
narrow question: does a trained dual-camera policy actually depend on each
camera branch?

Example:
    python examples/probe_sac_rgbd_camera_ablation.py \
      --checkpoint_path runs/<run>/checkpoints/final.pt \
      --encoder resnet10 \
      --pretrained_weights resnet10_pretrained_converted \
      --image_fusion_mode per_key
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro

from rl_garden.algorithms import SAC
from rl_garden.common import seed_everything
from rl_garden.common.cli_args import (
    image_encoder_factory_from_args,
    image_keys_from_env,
    vit_sac_kwargs_from_args,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    checkpoint_path: str

    env_id: str = "StackCube-v1"
    seed: int = 1
    num_eval_envs: int = 16
    num_eval_steps: int = 50
    device: str = "auto"

    obs_mode: str = "rgb"
    include_state: bool = True
    control_mode: str = "pd_joint_delta_pos"
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    render_mode: str = "rgb_array"
    per_camera_rgbd: bool = True

    encoder: Literal["plain_conv", "resnet10", "resnet18", "vit"] = "resnet10"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "per_key"
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["none", "random_shift"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False

    buffer_size: int = 200_000
    buffer_device: str = "cuda"
    batch_size: int = 512
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.25
    gamma: float = 0.8
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 1e-4
    alpha_tuning: Literal["legacy_exp", "log_alpha", "lagrange_softplus"] = "legacy_exp"
    ent_coef: float | str = "auto"
    target_entropy: float | str = "auto"
    alpha_lr: Optional[float] = None
    critic_impl: Literal["vmap", "legacy"] = "vmap"

    output_path: Optional[str] = None


def _clone_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in obs.items()}


def _apply_ablation(
    obs: dict[str, torch.Tensor],
    *,
    condition: str,
    camera_key: Optional[str],
) -> dict[str, torch.Tensor]:
    if condition == "both":
        return obs
    assert camera_key is not None
    out = _clone_obs(obs)
    if condition == "zero":
        out[camera_key] = torch.zeros_like(out[camera_key])
    elif condition == "shuffle":
        batch = out[camera_key].shape[0]
        if batch > 1:
            out[camera_key] = out[camera_key][torch.roll(torch.arange(batch, device=out[camera_key].device), 1)]
    else:
        raise ValueError(f"Unknown ablation condition: {condition}")
    return out


def _mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().detach().cpu().item())


def _build_agent(args: Args) -> tuple[SAC, tuple[str, ...]]:
    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        render_mode=args.render_mode,
        per_camera_rgbd=args.per_camera_rgbd,
        reconfiguration_freq=1,
    )
    env = make_maniskill_env(env_cfg)
    image_keys = image_keys_from_env(env, args)
    agent = SAC(
        env=env,
        eval_env=None,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_tuning=args.alpha_tuning,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        alpha_lr=args.alpha_lr,
        critic_impl=args.critic_impl,
        seed=args.seed,
        device=args.device,
        logger=None,
        std_log=False,
        eval_freq=0,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=None,
        checkpoint_freq=0,
        save_final_checkpoint=False,
        image_keys=image_keys,
        image_encoder_factory=image_encoder_factory_from_args(args),
        image_fusion_mode=args.image_fusion_mode,
        **vit_sac_kwargs_from_args(args, image_keys),
    )
    agent.load(args.checkpoint_path, load_replay_buffer=False, load_optimizers=False)
    agent.policy.eval()
    return agent, image_keys


def _rollout(
    agent: SAC,
    *,
    condition: str,
    camera_key: Optional[str],
    num_steps: int,
) -> dict[str, float]:
    obs, _ = agent.env.reset()
    returns = []
    successes_at_end = []
    successes_once = []
    reward_sum = 0.0
    action_abs_sum = 0.0
    n_action_steps = 0
    for _ in range(num_steps):
        policy_obs = _apply_ablation(obs, condition=condition, camera_key=camera_key)
        with torch.no_grad():
            action = agent.policy.predict(
                agent._obs_to_policy_device(policy_obs),
                deterministic=True,
            )
        action_abs_sum += float(action.abs().mean().detach().cpu().item())
        n_action_steps += 1
        obs, reward, _, _, infos = agent.env.step(action)
        reward_sum += float(reward.float().mean().detach().cpu().item())
        if "final_info" in infos:
            episode = infos["final_info"]["episode"]
            if "return" in episode:
                returns.append(episode["return"].float().mean())
            if "success_at_end" in episode:
                successes_at_end.append(episode["success_at_end"].float().mean())
            if "success_once" in episode:
                successes_once.append(episode["success_once"].float().mean())

    def _stack_mean(values: list[torch.Tensor]) -> Optional[float]:
        if not values:
            return None
        return float(torch.stack(values).mean().detach().cpu().item())

    return {
        "mean_step_reward": reward_sum / max(num_steps, 1),
        "episode_return": _stack_mean(returns),
        "success_at_end": _stack_mean(successes_at_end),
        "success_once": _stack_mean(successes_once),
        "mean_abs_action": action_abs_sum / max(n_action_steps, 1),
    }


def _sensitivity(
    agent: SAC,
    image_keys: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    obs, _ = agent.env.reset()
    obs_device = agent._obs_to_policy_device(obs)
    with torch.no_grad():
        base_features = agent.policy.extract_features(obs_device)
        base_action = agent.policy.predict(obs_device, deterministic=True)
    out = {}
    for key in image_keys:
        out[key] = {}
        for condition in ("zero", "shuffle"):
            ablated = agent._obs_to_policy_device(
                _apply_ablation(obs, condition=condition, camera_key=key)
            )
            with torch.no_grad():
                features = agent.policy.extract_features(ablated)
                action = agent.policy.predict(ablated, deterministic=True)
            out[key][f"{condition}_feature_delta"] = _mean_abs_delta(features, base_features)
            out[key][f"{condition}_action_delta"] = _mean_abs_delta(action, base_action)
    return out


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)
    agent, image_keys = _build_agent(args)
    conditions: list[tuple[str, Optional[str]]] = [("both", None)]
    for key in image_keys:
        conditions.append(("zero", key))
        conditions.append(("shuffle", key))

    rollouts = {}
    for condition, camera_key in conditions:
        name = condition if camera_key is None else f"{condition}:{camera_key}"
        rollouts[name] = _rollout(
            agent,
            condition=condition,
            camera_key=camera_key,
            num_steps=args.num_eval_steps,
        )

    result = {
        "args": asdict(args),
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "image_keys": image_keys,
        "sensitivity": _sensitivity(agent, image_keys),
        "rollouts": rollouts,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.output_path is not None:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(text)
    agent.env.close()


if __name__ == "__main__":
    main()
