"""Train ACT on converted RoboTwin RGB demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from rl_garden.models.act.act.utils import IterationBasedBatchSampler, worker_init_fn
from rl_garden.models.act.config import ACTConfig
from rl_garden.models.act.provider import ACTPolicyModel
from rl_garden.models.act.robotwin_dataset import (
    ROBOTWIN_ACT_CAMERA_NAMES,
    RoboTwinACTDataset,
    normalize_robotwin_task_name,
    robotwin_json_path,
)


class EMA:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = [param.detach().clone() for param in parameters if param.requires_grad]

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [param for param in parameters if param.requires_grad]
        for shadow, param in zip(self.shadow, params):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [param for param in parameters if param.requires_grad]
        for shadow, param in zip(self.shadow, params):
            param.copy_(shadow)


class ACTTrainer(nn.Module):
    def __init__(self, *, state_dim: int, action_dim: int, config: ACTConfig, kl_weight: float) -> None:
        super().__init__()
        self.policy = ACTPolicyModel(
            state_dim=state_dim,
            action_dim=action_dim,
            visual=True,
            config=config,
        )
        self.kl_weight = float(kl_weight)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )

    def compute_loss(self, obs: dict[str, torch.Tensor], action_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        obs = dict(obs)
        obs["rgb"] = (obs["rgb"].float() / 255.0 - self.rgb_mean) / self.rgb_std
        pred, (mu, logvar) = self.policy.model(obs, action_seq)
        total_kld = _kl_divergence(mu, logvar)
        l1 = F.l1_loss(action_seq, pred)
        return {"l1": l1, "kl": total_kld, "loss": l1 + total_kld * self.kl_weight}


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu.ndim == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.ndim == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0, keepdim=True)[0]


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--env-id", default="open_laptop")
    parser.add_argument("--control-mode", default="delta_ee", choices=["delta_ee"])
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=_str_to_bool, default=True)
    parser.add_argument("--torch-deterministic", type=_str_to_bool, default=True)
    parser.add_argument("--total-iters", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--num-dataload-workers", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--position-embedding", default="sine")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--masks", type=_str_to_bool, default=False)
    parser.add_argument("--dilation", type=_str_to_bool, default=False)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--pre-norm", type=_str_to_bool, default=False)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--camera-names", nargs="+", default=list(ROBOTWIN_ACT_CAMERA_NAMES))
    parser.add_argument("--track", type=_str_to_bool, default=False)
    parser.add_argument("--wandb-project-name", default="rl-garden")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def _assert_dataset_control_mode(args: argparse.Namespace) -> None:
    json_path = robotwin_json_path(args.demo_path)
    if not json_path.exists():
        return
    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    control_mode = metadata.get("env_info", {}).get("env_kwargs", {}).get("control_mode")
    if control_mode is not None and control_mode != args.control_mode:
        raise ValueError(
            f"Dataset control_mode={control_mode!r} does not match "
            f"--control-mode {args.control_mode!r}."
        )


def _device(args: argparse.Namespace) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def _save_checkpoint(
    path: Path,
    *,
    agent: ACTTrainer,
    ema_agent: ACTTrainer,
    dataset: RoboTwinACTDataset,
    args: argparse.Namespace,
    act_config: ACTConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "norm_stats": dataset.norm_stats,
            "agent": agent.policy.state_dict(),
            "ema_agent": ema_agent.policy.state_dict(),
            "config": {
                **vars(args),
                "act_config": asdict(act_config),
                "state_dim": dataset.state_dim,
                "action_dim": dataset.action_dim,
                "camera_names": list(dataset.camera_names),
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    args.env_id = normalize_robotwin_task_name(args.env_id)
    _assert_dataset_control_mode(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = _device(args)
    run_name = args.exp_name or f"{args.env_id}_act_{args.seed}_{int(time.time())}"
    run_dir = Path(args.output_dir).expanduser() / run_name
    checkpoint_dir = run_dir / "checkpoints"
    writer = SummaryWriter(str(run_dir))

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group="ACT",
            tags=["act", "robotwin", args.env_id],
        )

    dataset = RoboTwinACTDataset(
        args.demo_path,
        num_queries=args.num_queries,
        num_traj=args.num_demos,
        camera_names=args.camera_names,
        image_size=(args.image_height, args.image_width),
        control_mode=args.control_mode,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    loader = DataLoader(
        dataset,
        batch_sampler=iter_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )

    act_config = ACTConfig(
        position_embedding=args.position_embedding,
        backbone=args.backbone,
        lr_backbone=args.lr_backbone,
        masks=args.masks,
        dilation=args.dilation,
        include_depth=False,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        nheads=args.nheads,
        num_queries=args.num_queries,
        pre_norm=args.pre_norm,
        image_size=(args.image_height, args.image_width),
    )
    agent = ACTTrainer(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        config=act_config,
        kl_weight=args.kl_weight,
    ).to(device)
    ema_agent = ACTTrainer(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        config=act_config,
        kl_weight=args.kl_weight,
    ).to(device)
    ema_agent.load_state_dict(agent.state_dict())
    ema = EMA(agent.parameters(), decay=args.ema_decay)

    param_groups = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, max(1, int((2 / 3) * args.total_iters))
    )

    agent.train()
    print(
        f"training ACT run={run_name} demos={dataset.num_traj} "
        f"samples={len(dataset)} state_dim={dataset.state_dim} action_dim={dataset.action_dim}",
        flush=True,
    )
    for cur_iter, batch in enumerate(loader):
        obs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch["observations"].items()
        }
        actions = batch["actions"].to(device, non_blocking=True)
        loss = agent.compute_loss(obs, actions)
        optimizer.zero_grad(set_to_none=True)
        loss["loss"].backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())

        if cur_iter % args.log_freq == 0:
            print(
                f"iter={cur_iter} loss={loss['loss'].item():.6g} "
                f"l1={loss['l1'].item():.6g} kl={loss['kl'].item():.6g}",
                flush=True,
            )
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("charts/backbone_learning_rate", optimizer.param_groups[1]["lr"], cur_iter)
            for key, value in loss.items():
                writer.add_scalar(f"losses/{key}", float(value.item()), cur_iter)

        if args.save_freq and cur_iter > 0 and cur_iter % args.save_freq == 0:
            ema.copy_to(ema_agent.parameters())
            _save_checkpoint(
                checkpoint_dir / f"checkpoint_{cur_iter}.pt",
                agent=agent,
                ema_agent=ema_agent,
                dataset=dataset,
                args=args,
                act_config=act_config,
            )

    ema.copy_to(ema_agent.parameters())
    final_path = checkpoint_dir / "final.pt"
    _save_checkpoint(
        final_path,
        agent=agent,
        ema_agent=ema_agent,
        dataset=dataset,
        args=args,
        act_config=act_config,
    )
    writer.close()
    if args.track:
        import wandb

        wandb.finish()
    print(f"wrote checkpoint: {final_path}", flush=True)
    return final_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
