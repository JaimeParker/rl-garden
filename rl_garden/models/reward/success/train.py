"""Reward-classifier training entrypoint, mirroring HIL-SERL's own
``3rd_party/hil-serl/examples/train_reward_classifier.py`` (balanced
success/failure batch sampling, ``sigmoid_binary_cross_entropy``-equivalent
loss, random-shift augmentation, fixed number of epochs, no train/val split
-- see ``docs/hil_serl_roadmap.md`` item 2). Standalone argparse CLI, not
routed through ``BaseAlgorithmRegistry``: this is supervised classifier
training, not an RL algorithm.

Usage::

    python -m rl_garden.models.reward.success.train \\
        --success_paths classifier_data/*success*.pkl \\
        --failure_paths classifier_data/*failure*.pkl \\
        --output_dir classifier_ckpt/
"""
from __future__ import annotations

import argparse
import os
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.utils.data import DataLoader

from rl_garden.encoders.augment import RandomShiftsAug
from rl_garden.models.reward.success.data import SuccessClassifierDataset, collate_obs_label
from rl_garden.models.reward.success.model import SuccessClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the online success reward classifier.")
    parser.add_argument("--success_paths", nargs="+", required=True, help="Glob pattern(s) for success pkl files.")
    parser.add_argument("--failure_paths", nargs="+", required=True, help="Glob pattern(s) for failure pkl files.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pretrained_weights",
        default="resnet10-imagenet",
        help="Passed to SuccessClassifier's frozen ResNet encoder; see "
        "rl_garden/encoders/resnet.py:pretrained_dir(). The encoder is "
        "always frozen, so 'none' (no pretrained weights loaded) only makes "
        "sense for smoke-testing this script, not real training.",
    )
    return parser.parse_args()


def _infer_observation_space(sample_obs: dict) -> spaces.Dict:
    return spaces.Dict(
        {k: spaces.Box(0, 255, tuple(v.shape), dtype=np.uint8) for k, v in sample_obs.items()}
    )


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    args = parse_args()

    success_ds = SuccessClassifierDataset(args.success_paths, [])
    failure_ds = SuccessClassifierDataset([], args.failure_paths)
    print(f"success examples: {len(success_ds)}", flush=True)
    print(f"failure examples: {len(failure_ds)}", flush=True)

    half_batch = args.batch_size // 2
    success_loader = DataLoader(
        success_ds, batch_size=half_batch, shuffle=True, drop_last=True, collate_fn=collate_obs_label
    )
    failure_loader = DataLoader(
        failure_ds, batch_size=half_batch, shuffle=True, drop_last=True, collate_fn=collate_obs_label
    )
    success_iter = _cycle(success_loader)
    failure_iter = _cycle(failure_loader)

    sample_obs, _ = success_ds[0]
    image_keys = list(sample_obs.keys())
    observation_space = _infer_observation_space(sample_obs)

    pretrained_weights = None if args.pretrained_weights.lower() == "none" else args.pretrained_weights
    model = SuccessClassifier(observation_space, image_keys, pretrained_weights=pretrained_weights).to(
        args.device
    )
    augment = RandomShiftsAug(padding=4).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.num_epochs):
        pos_obs, pos_labels = next(success_iter)
        neg_obs, neg_labels = next(failure_iter)
        obs = {
            k: augment(torch.cat([pos_obs[k], neg_obs[k]]).float().to(args.device)) for k in image_keys
        }
        labels = torch.cat([pos_labels, neg_labels]).to(args.device)

        logits = model(obs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = ((torch.sigmoid(logits) >= 0.5) == (labels >= 0.5)).float().mean()
        print(
            f"epoch {epoch + 1}/{args.num_epochs} loss={loss.item():.4f} accuracy={accuracy.item():.4f}",
            flush=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "success_classifier.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
