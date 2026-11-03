"""Train an ACT policy from pickled transition lists, with no simulator.

Usage::

    python -m rl_garden.models.act.train_trajectories \\
        --demo-paths "demos/*.pkl" --total-iters 50000 --exp-name act-my-task

Each pkl holds a list of transition dicts (see ``trajectory_dataset``). Observations with
``rgb_<camera>`` keys train a visual ACT policy over those cameras (``--no-use-cameras`` trains on
``state`` alone). Checkpoints go to ``<log-dir>/<run-name>/checkpoints/`` (``<iter>.pt`` every
``--save-freq`` iterations, ``final.pt`` at the end) and load with
``ACTBaseActionProvider.from_checkpoint(ckpt_path=...)``: they hold ``agent`` and ``ema_agent``
weights, ``norm_stats`` and the full ``act_config``.

The model is ``ACTPolicyModel`` (the provider's own, so checkpoint keys match the loader), the
loss and optimiser follow the vendored ACT scripts (L1 + ``kl_weight`` * KL, AdamW with a
separate backbone learning rate, learning rate x0.1 after two thirds of the iterations, an EMA
copy of the weights); unlike them the L1 term ignores padded chunk entries, as in the original
ACT, so no control-mode-specific padding is needed. There is no evaluation loop.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_garden.common import Logger, seed_everything
from rl_garden.models.act.config import ACT_CONFIG_CHECKPOINT_KEY, ACTConfig
from rl_garden.models.act.provider import ACTPolicyModel
from rl_garden.models.act.trajectory_dataset import (
    ACTBatch,
    ACTTrajectoryDataset,
    load_transition_pkls,
)

# What ACTBaseActionProvider normalises images with at inference.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Args:
    demo_paths: tuple[str, ...] = ()
    """glob pattern(s) of pickled transition lists"""
    exp_name: Optional[str] = None
    """run name; defaults to act_trajectories__<seed>__<unix time>"""
    log_dir: str = "runs"
    """runs go to <log_dir>/<run name>/ (checkpoints under checkpoints/)"""
    log_type: str = "tensorboard"
    """tensorboard, wandb or none"""
    wandb_project: str = "rl-garden"
    wandb_entity: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    """train on CUDA when available"""

    total_iters: int = 100_000
    batch_size: int = 256
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    kl_weight: float = 10.0
    log_freq: int = 1000
    save_freq: Optional[int] = None
    """also save <iter>.pt every this many iterations"""

    use_cameras: bool = True
    """train on the rgb_<camera> keys when the observations have them"""
    split_on_observation_gaps: bool = True
    """also end an episode where a transition's next_obs differs from the next transition's obs"""
    backbone_weights: str = "imagenet"
    """visual backbone init: imagenet (torchvision IMAGENET1K_V1), none, or a torchvision state-dict path"""

    # ACT architecture (stored in the checkpoint's act_config)
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    dilation: bool = False
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: Optional[int] = None
    """attention heads; default 8 for a visual policy, 4 for a state-only one"""
    num_queries: int = 30
    pre_norm: bool = False
    image_size: int = 224
    """camera images are resized to image_size x image_size, in training and at inference"""


def act_config_from_args(args: Args, *, visual: bool) -> ACTConfig:
    nheads = args.nheads if args.nheads is not None else (8 if visual else 4)
    if args.hidden_dim % nheads != 0:
        raise ValueError(f"hidden_dim={args.hidden_dim} is not divisible by nheads={nheads}")
    return ACTConfig(
        position_embedding=args.position_embedding,
        backbone=args.backbone,
        lr_backbone=args.lr_backbone,
        dilation=args.dilation,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        nheads=nheads,
        num_queries=args.num_queries,
        pre_norm=args.pre_norm,
        image_size=args.image_size,
    )


def load_backbone_weights(policy: ACTPolicyModel, weights: str, *, backbone: str) -> None:
    """Initialise the visual backbone from ``weights``: ``"imagenet"``, ``"none"`` or the path of
    a torchvision ResNet state dict (its ``fc`` head is dropped)."""

    if weights == "none":
        return
    if weights == "imagenet":
        import torchvision

        weight_enum = torchvision.models.get_model_weights(backbone)
        state_dict = weight_enum["IMAGENET1K_V1"].get_state_dict(progress=True)
    else:
        state_dict = torch.load(weights, map_location="cpu")
    body = policy.model.backbones[0][0].body
    trunk_state = {key: value for key, value in state_dict.items() if not key.startswith("fc.")}
    missing, unexpected = body.load_state_dict(trunk_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"backbone weights {weights!r} do not fit {backbone}: missing={missing}, unexpected={unexpected}"
        )


def prepare_act_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """``(B, N, 3, H, W)`` uint8 camera images -> the normalised float input of the ACT backbone,
    as ``ACTBaseActionProvider`` prepares them at inference."""

    batch, num_cameras, channels, height, width = images.shape
    rgb = images.float() / 255.0
    if height != image_size or width != image_size:
        rgb = F.interpolate(
            rgb.reshape(batch * num_cameras, channels, height, width),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, num_cameras, channels, image_size, image_size)
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(1, 1, 3, 1, 1)
    return (rgb - mean) / std


def act_loss(
    policy: ACTPolicyModel, batch: ACTBatch, *, image_size: int, kl_weight: float
) -> dict[str, torch.Tensor]:
    """ACT's CVAE loss: L1 over the non-padded chunk entries plus ``kl_weight`` * KL."""

    if batch.images is None:
        obs: Union[torch.Tensor, dict[str, torch.Tensor]] = batch.state
    else:
        obs = {"state": batch.state, "rgb": prepare_act_images(batch.images, image_size)}
    predicted_chunk, (mu, logvar) = policy.model(obs, batch.action_chunk)
    per_entry_l1 = F.l1_loss(predicted_chunk, batch.action_chunk, reduction="none")
    l1 = (per_entry_l1 * ~batch.is_pad.unsqueeze(-1)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return {"l1": l1, "kl": kl, "loss": l1 + kl_weight * kl}


class ACTWeightEma:
    """EMA copy of a model's weights with the decay schedule of diffusers' ``EMAModel`` as the
    vendored ACT scripts construct it: ``min(max_decay, (1 + t) / (10 + t))`` at update
    ``t + 1``, so the first update copies the weights. Parameters without gradients are copied."""

    def __init__(self, model: nn.Module, *, max_decay: float = 0.9999) -> None:
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.max_decay = max_decay
        self.num_updates = 0

    def decay(self) -> float:
        step = self.num_updates - 1
        if step <= 0:
            return 0.0
        return min(self.max_decay, (1 + step) / (10 + step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        weight = 1.0 - self.decay()
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            if param.requires_grad:
                ema_param.lerp_(param, weight)
            else:
                ema_param.copy_(param)


def save_act_checkpoint(
    path: Path,
    *,
    policy: ACTPolicyModel,
    ema: ACTWeightEma,
    dataset: ACTTrajectoryDataset,
    config: ACTConfig,
    iteration: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": policy.state_dict(),
            "ema_agent": ema.model.state_dict(),
            "norm_stats": dataset.norm_stats,
            ACT_CONFIG_CHECKPOINT_KEY: dataclasses.asdict(config),
            "camera_keys": list(dataset.camera_keys),
            "iteration": int(iteration),
        },
        path,
    )
    return path


def _optimizer(policy: ACTPolicyModel, args: Args) -> torch.optim.Optimizer:
    named = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
    groups = [{"params": [param for name, param in named if "backbone" not in name]}]
    backbone_params = [param for name, param in named if "backbone" in name]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def train_act_from_trajectories(args: Args) -> Path:
    """Train ACT on ``args.demo_paths``; returns the path of the final checkpoint."""

    if not args.demo_paths:
        raise ValueError("--demo-paths needs at least one glob pattern")
    seed_everything(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    run_name = args.exp_name or f"act_trajectories__{args.seed}__{int(time.time())}"
    checkpoint_dir = Path(args.log_dir) / run_name / "checkpoints"

    dataset = ACTTrajectoryDataset(
        load_transition_pkls(args.demo_paths),
        num_queries=args.num_queries,
        device=device,
        use_cameras=args.use_cameras,
        split_on_observation_gaps=args.split_on_observation_gaps,
    )
    visual = bool(dataset.camera_keys)
    config = act_config_from_args(args, visual=visual)
    print(
        f"[act] {len(dataset)} steps in {dataset.num_episodes} episodes, state_dim={dataset.state_dim}, "
        f"action_dim={dataset.action_dim}, cameras={list(dataset.camera_keys) or 'none'}, device={device}"
    )

    policy = ACTPolicyModel(
        state_dim=dataset.state_dim, action_dim=dataset.action_dim, visual=visual, config=config
    )
    if visual:
        load_backbone_weights(policy, args.backbone_weights, backbone=args.backbone)
    policy.to(device).train()
    ema = ACTWeightEma(policy)
    optimizer = _optimizer(policy, args)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(2 / 3 * args.total_iters))
    )

    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=dataclasses.asdict(args),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    update_seconds = 0.0
    try:
        for iteration in range(args.total_iters):
            tick = time.perf_counter()
            losses = act_loss(
                policy,
                dataset.sample(args.batch_size),
                image_size=config.image_size,
                kl_weight=args.kl_weight,
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            lr_scheduler.step()
            ema.update(policy)
            update_seconds += time.perf_counter() - tick

            if iteration % args.log_freq == 0:
                total_loss = losses["loss"].item()
                print(f"[act] iteration {iteration}, loss {total_loss:.4f}")
                logger.add_scalar("losses/total_loss", total_loss, iteration)
                logger.add_scalar("losses/l1", losses["l1"].item(), iteration)
                logger.add_scalar("losses/kl", losses["kl"].item(), iteration)
                logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                logger.add_scalar("time/update", update_seconds, iteration)
            if args.save_freq and iteration > 0 and iteration % args.save_freq == 0:
                save_act_checkpoint(
                    checkpoint_dir / f"{iteration}.pt",
                    policy=policy,
                    ema=ema,
                    dataset=dataset,
                    config=config,
                    iteration=iteration,
                )
    finally:
        logger.close()
    final_path = save_act_checkpoint(
        checkpoint_dir / "final.pt",
        policy=policy,
        ema=ema,
        dataset=dataset,
        config=config,
        iteration=args.total_iters,
    )
    print(f"[act] saved {os.fspath(final_path)}")
    return final_path


def main() -> None:
    import tyro

    train_act_from_trajectories(tyro.cli(Args))


if __name__ == "__main__":
    main()
