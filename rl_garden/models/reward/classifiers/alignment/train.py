#!/usr/bin/env python3
"""Train the alignment reward classifier (ResNet)."""
from __future__ import annotations

import argparse
from pathlib import Path

from rl_garden.models.reward.classifiers.alignment.loop import (
    AlignmentClassifierConfig,
    AlignmentClassifierTrainer,
)


def parse_args() -> AlignmentClassifierConfig:
    parser = argparse.ArgumentParser(description="Train alignment reward classifier")

    parser.add_argument(
        "--label_files",
        nargs="+",
        default=[
            "data/epi0-19_trimmed/dual_camera_rgb_labels.hdf5",
            "data/epi20-36_trimmed/dual_camera_rgb_labels.hdf5",
        ],
    )
    parser.add_argument("--output_dir", default="data/checkpoints_alignment")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_pretrained", action="store_true")

    args = parser.parse_args()
    return AlignmentClassifierConfig(
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        resnet_pretrained=not args.no_pretrained,
        label_files=[Path(p) for p in args.label_files],
    )


def main() -> None:
    config = parse_args()
    trainer = AlignmentClassifierTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
