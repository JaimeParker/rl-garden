"""Shared training loop for binary classifiers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from rl_garden.common.utils import get_device, seed_everything
from rl_garden.models.reward.classifiers.shared.metrics import compute_metrics


@dataclass(frozen=True)
class BaseTrainConfig:
    output_dir: Path
    image_size: int = 128
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0
    seed: int = 42
    device: str | torch.device = "auto"
    normalize: bool = True
    resnet_pretrained: bool = True


class BaseBinaryClassifierTrainer:
    """Shared training loop with dataset/model hooks."""

    def __init__(self, config: BaseTrainConfig) -> None:
        self.config = config
        self.device = get_device(config.device)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(str(self.output_dir / "runs"))

        seed_everything(config.seed)

    def build_dataset(self) -> Dataset:
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def _build_loaders(self, dataset: Dataset) -> tuple[DataLoader, DataLoader]:
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer | None,
    ) -> tuple[float, dict[str, float]]:
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = 0.0
        preds: list[float] = []
        labels: list[float] = []

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            preds.extend(outputs.detach().cpu().numpy().flatten())
            labels.extend(targets.detach().cpu().numpy().flatten())

        metrics = compute_metrics(np.array(preds), np.array(labels))
        return total_loss / max(1, len(loader)), metrics

    def train(self, *, config_payload: dict[str, object]) -> None:
        dataset = self.build_dataset()
        train_loader, val_loader = self._build_loaders(dataset)
        model = self.build_model()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        best_f1 = 0.0
        best_model_path = self.output_dir / "best_model.pt"

        for epoch in trange(self.config.num_epochs, desc="Training"):
            train_loss, _ = self._run_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_metrics = self._run_epoch(model, val_loader, criterion, None)

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            for key, val in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", val, epoch)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), best_model_path)

            scheduler.step(val_metrics["f1"])
            if optimizer.param_groups[0]["lr"] < 1e-6:
                break

        self.writer.close()

        config_payload = dict(config_payload)
        config_payload.update(
            {
                "best_f1": best_f1,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.num_epochs,
            }
        )
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_payload, f, indent=2)
