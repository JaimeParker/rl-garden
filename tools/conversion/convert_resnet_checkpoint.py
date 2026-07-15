#!/usr/bin/env python3
"""Convert torchvision-style ResNet checkpoints to rl-garden ResNetEncoder keys.

Example:
    python scripts/convert_resnet_checkpoint.py \
        --input pretrained_models/resnet10_pretrained.pt \
        --output pretrained_models/resnet10_pretrained_converted.pt \
        --arch resnet10
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.resnet import ResNetEncoder

_ARCH_TO_STAGE_SIZES: dict[str, Sequence[int]] = {
    "resnet10": (1, 1, 1, 1),
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}


def _unwrap_state_dict(state: object) -> dict[str, torch.Tensor]:
    if isinstance(state, dict) and "state_dict" in state and isinstance(
        state["state_dict"], dict
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state_dict or {'state_dict': ...}")
    return state


def _strip_prefix(key: str, prefixes: Iterable[str]) -> str:
    out = key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p) :]
                changed = True
    return out


def _parse_layer_block(key: str) -> tuple[int, int, str] | None:
    if not key.startswith("layer"):
        return None
    parts = key.split(".")
    if len(parts) < 3:
        return None
    layer_token = parts[0]
    if len(layer_token) != 6 or not layer_token[5].isdigit():
        return None
    stage_idx = int(layer_token[5]) - 1
    if stage_idx < 0:
        return None
    block_token = parts[1]
    if not block_token.isdigit():
        return None
    block_idx = int(block_token)
    suffix = ".".join(parts[2:])
    return stage_idx, block_idx, suffix


def _map_key_torchvision_to_rlg(
    key: str,
    stage_sizes: Sequence[int],
) -> str | None:
    if key == "conv1.weight":
        return "stem_conv.weight"
    if key.startswith("bn1."):
        bn_suffix = key.split(".", 1)[1]
        if bn_suffix == "weight":
            return "stem_norm.weight"
        if bn_suffix == "bias":
            return "stem_norm.bias"
        return None

    parsed = _parse_layer_block(key)
    if parsed is None:
        return None
    stage_idx, block_idx, suffix = parsed
    if stage_idx >= len(stage_sizes):
        return None
    if block_idx >= stage_sizes[stage_idx]:
        return None
    flat_idx = sum(stage_sizes[:stage_idx]) + block_idx
    block_prefix = f"blocks.{flat_idx}."

    if suffix == "conv1.weight":
        return block_prefix + "conv1.weight"
    if suffix == "conv2.weight":
        return block_prefix + "conv2.weight"
    if suffix.startswith("bn1."):
        end = suffix.split(".", 1)[1]
        if end == "weight":
            return block_prefix + "norm1.weight"
        if end == "bias":
            return block_prefix + "norm1.bias"
        return None
    if suffix.startswith("bn2."):
        end = suffix.split(".", 1)[1]
        if end == "weight":
            return block_prefix + "norm2.weight"
        if end == "bias":
            return block_prefix + "norm2.bias"
        return None
    if suffix == "downsample.0.weight":
        return block_prefix + "proj.weight"
    if suffix.startswith("downsample.1."):
        end = suffix.split(".", 2)[2]
        if end == "weight":
            return block_prefix + "proj_norm.weight"
        if end == "bias":
            return block_prefix + "proj_norm.bias"
        return None
    return None


def convert_state_dict(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    stage_sizes: Sequence[int],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    converted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for raw_key, tensor in source_state.items():
        key = _strip_prefix(raw_key, prefixes=("module.", "encoder.", "backbone."))
        mapped = _map_key_torchvision_to_rlg(key, stage_sizes=stage_sizes)
        if mapped is None:
            skipped.append(f"{raw_key}:unsupported")
            continue
        target_tensor = target_state.get(mapped)
        if target_tensor is None:
            skipped.append(f"{raw_key}:target-missing:{mapped}")
            continue
        if tuple(tensor.shape) != tuple(target_tensor.shape):
            skipped.append(
                f"{raw_key}:shape-mismatch:{tuple(tensor.shape)}!={tuple(target_tensor.shape)}"
            )
            continue
        converted[mapped] = tensor
    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="source checkpoint path")
    parser.add_argument("--output", required=True, type=Path, help="output checkpoint path")
    parser.add_argument(
        "--arch",
        default="resnet10",
        choices=sorted(_ARCH_TO_STAGE_SIZES.keys()),
        help="target ResNetEncoder architecture",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_sizes = _ARCH_TO_STAGE_SIZES[args.arch]

    source = torch.load(args.input, map_location="cpu")
    source_state = _unwrap_state_dict(source)

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)
    target_encoder = ResNetEncoder(obs_space, stage_sizes=stage_sizes)
    target_state = target_encoder.state_dict()

    converted, skipped = convert_state_dict(
        source_state=source_state,
        target_state=target_state,
        stage_sizes=stage_sizes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)

    print(
        f"mapped={len(converted)} skipped={len(skipped)} "
        f"source={args.input} output={args.output}"
    )
    if skipped:
        print("skipped_samples=" + "; ".join(skipped[:10]))


if __name__ == "__main__":
    main()
