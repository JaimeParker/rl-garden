"""Configuration and checkpoint helpers for ACT base policies."""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_ACT_CHECKPOINT = "act-peg-only"


@dataclass
class ACTConfig:
    """Minimal ACT architecture config needed for inference."""

    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 0.0
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 4
    num_queries: int = 30
    pre_norm: bool = False
    image_size: int = 224


@dataclass(frozen=True)
class ACTCheckpointSpec:
    state_dim: int
    action_dim: int
    visual: bool


def act_pretrained_dir() -> Path:
    """Directory searched for named ACT checkpoints.

    Resolution order follows the ResNet encoder convention:

    1. ``$RL_GARDEN_PRETRAINED_DIR`` if set.
    2. ``pretrained/act/`` at the repo root.
    """

    env = os.environ.get("RL_GARDEN_PRETRAINED_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # rl_garden/models/act/config.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[3] / "pretrained" / "act"


def resolve_act_checkpoint_path(ckpt_path: str | os.PathLike[str] | None) -> Path:
    """Resolve an ACT checkpoint path or pretrained checkpoint name."""

    raw = DEFAULT_ACT_CHECKPOINT if ckpt_path is None else str(ckpt_path)
    direct = Path(raw).expanduser()
    if direct.exists():
        return direct.resolve()

    pretrained_dir = act_pretrained_dir()
    candidates: list[Path] = []
    if direct.suffix:
        candidates.append(pretrained_dir / direct.name)
    else:
        candidates.append(pretrained_dir / f"{raw}.pt")
        candidates.append(pretrained_dir / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(p) for p in [direct, *candidates])
    raise FileNotFoundError(
        f"ACT checkpoint {raw!r} was not found. Tried: {tried}. "
        "Pass --ckpt-path /path/to/file.pt or place <name>.pt in "
        f"{pretrained_dir}."
    )


def select_act_state_dict(
    checkpoint: dict[str, Any],
    *,
    state_dict_key: str = "ema_agent",
) -> dict[str, torch.Tensor]:
    """Select the ACT model state dict from a training checkpoint."""

    if state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif state_dict_key == "ema_agent" and "agent" in checkpoint:
        state_dict = checkpoint["agent"]
    elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        keys = ", ".join(sorted(str(k) for k in checkpoint.keys()))
        raise KeyError(
            f"ACT checkpoint does not contain {state_dict_key!r} or fallback "
            f"'agent'. Available keys: {keys}"
        )
    if not isinstance(state_dict, dict):
        raise TypeError(f"ACT state dict under {state_dict_key!r} is not a dict.")
    return state_dict


def _count_indexed_layers(keys: set[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    return len({int(match.group(1)) for key in keys if (match := pattern.match(key))})


def infer_act_config(state_dict: dict[str, torch.Tensor]) -> tuple[ACTConfig, ACTCheckpointSpec]:
    """Infer ACT architecture fields stored implicitly in a checkpoint."""

    keys = set(state_dict.keys())
    visual = any(key.startswith("model.backbones.") for key in keys)

    action_head = state_dict["model.action_head.weight"]
    input_state = state_dict["model.input_proj_robot_state.weight"]
    query_embed = state_dict["model.query_embed.weight"]

    hidden_dim = int(action_head.shape[1])
    action_dim = int(action_head.shape[0])
    state_dim = int(input_state.shape[1])
    num_queries = int(query_embed.shape[0])

    enc_layers = _count_indexed_layers(keys, "model.transformer.encoder.layers") or 2
    dec_layers = _count_indexed_layers(keys, "model.transformer.decoder.layers") or 4
    ff_key = "model.transformer.encoder.layers.0.linear1.weight"
    dim_feedforward = int(state_dict[ff_key].shape[0]) if ff_key in state_dict else 512

    nheads = 8 if visual else 4
    if hidden_dim % nheads != 0:
        for candidate in (8, 4, 2, 1):
            if hidden_dim % candidate == 0:
                nheads = candidate
                break

    include_depth = False
    conv1_key = "model.backbones.0.0.body.conv1.weight"
    if conv1_key in state_dict:
        include_depth = int(state_dict[conv1_key].shape[1]) > 3

    config = ACTConfig(
        hidden_dim=hidden_dim,
        nheads=nheads,
        num_queries=num_queries,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        include_depth=include_depth,
    )
    spec = ACTCheckpointSpec(
        state_dim=state_dim,
        action_dim=action_dim,
        visual=visual,
    )
    return config, spec


# Checkpoint key holding ``dataclasses.asdict(ACTConfig)``, written by trainers that know the
# full architecture (``rl_garden.models.act.train_trajectories``).
ACT_CONFIG_CHECKPOINT_KEY = "act_config"
# ACTConfig fields ``infer_act_config`` reads off the weight shapes.
_SHAPE_DERIVED_FIELDS = (
    "hidden_dim",
    "num_queries",
    "enc_layers",
    "dec_layers",
    "dim_feedforward",
    "include_depth",
)


def resolve_act_config(
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> tuple[ACTConfig, ACTCheckpointSpec]:
    """ACT architecture of a checkpoint.

    Without a stored ``act_config`` this is ``infer_act_config(state_dict)``. With one, the
    stored config also supplies the fields weight shapes cannot reveal (``nheads``,
    ``image_size``, ``backbone``, ``pre_norm``, ...); the shape-derived fields must agree with
    the weights.
    """

    inferred, spec = infer_act_config(state_dict)
    stored = checkpoint.get(ACT_CONFIG_CHECKPOINT_KEY)
    if stored is None:
        return inferred, spec
    known_fields = {field.name for field in dataclasses.fields(ACTConfig)}
    unknown = sorted(set(stored) - known_fields)
    if unknown:
        raise ValueError(f"ACT checkpoint {ACT_CONFIG_CHECKPOINT_KEY!r} has unknown fields {unknown}.")
    config = ACTConfig(**stored)
    mismatched = {
        name: (getattr(config, name), getattr(inferred, name))
        for name in _SHAPE_DERIVED_FIELDS
        if getattr(config, name) != getattr(inferred, name)
    }
    if mismatched:
        raise ValueError(
            f"ACT checkpoint {ACT_CONFIG_CHECKPOINT_KEY!r} disagrees with its weights "
            f"(stored, from weights): {mismatched}."
        )
    return config, spec
