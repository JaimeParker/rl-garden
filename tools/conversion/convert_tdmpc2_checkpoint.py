"""Convert an upstream/HF TD-MPC2 multitask agent checkpoint into an
rl-garden ``MultitaskWorldModel`` state_dict for PWM.

Input: ``torch.save({"model": WorldModel.state_dict()})`` from upstream
``tdmpc2/tdmpc2.py`` (also the HF ``imgeorgiev/pwm`` multitask models,
which were trained with a TD-MPC2 fork). ``tensordict`` is imported if
available (only needed to unpickle checkpoints that store TensorDicts).

Key rules:
  - dropped: ``_pi.*`` (policy prior), every key containing ``_Qs``
    (``_Qs``/``_target_Qs``/``_detach_Qs`` and their ``*_params`` forms),
    ``_termination.*``, ``log_std_min``/``log_std_dif``, ``_action_masks``.
  - ``_encoder.state.<rest>`` (upstream) or ``_encoder.<rest>`` (PWM fork)
    -> ``encoder.<rest>``; ``_task_emb.weight`` -> ``task_emb.weight``;
    ``_dynamics.*``/``_reward.*`` unchanged.
  - ``_action_masks`` is a registered buffer in rl-garden's
    ``MultitaskWorldModel``; it is rebuilt from per-task ``action_dims``
    taken (in priority order) from ``--action_dims``, ``--dataset_dir``'s
    ``manifest.json``, or the source's own ``_action_masks`` row sums.

Architecture (latent/enc/mlp/task dims, encoder depth, num_tasks, num_bins,
obs/action dims) is inferred from the weight shapes; ``simnorm_dim``/
``vmin``/``vmax`` cannot be and default to TD-MPC2's 8/-10/10.

Output in ``--dst_dir``: ``world_model.pt`` (state_dict) + ``manifest.json``.
The result is validated with ``load_state_dict(strict=True)`` into a
freshly built ``MultitaskWorldModel``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

_DROP_PREFIXES = ("_pi.", "_termination.")
_DROP_EXACT = ("log_std_min", "log_std_dif", "_action_masks")


def _is_dropped(key: str) -> bool:
    return key.startswith(_DROP_PREFIXES) or "_Qs" in key or key in _DROP_EXACT


def _rename(key: str) -> str:
    if key.startswith("_encoder.state."):
        return "encoder." + key[len("_encoder.state."):]
    if key.startswith("_encoder."):
        return "encoder." + key[len("_encoder."):]
    if key == "_task_emb.weight":
        return "task_emb.weight"
    if key.startswith(("_dynamics.", "_reward.")):
        return key
    raise KeyError(f"Unrecognized TD-MPC2 world-model key {key!r}.")


def _linear_weights(state_dict: dict[str, torch.Tensor], prefix: str) -> list[torch.Tensor]:
    """2-D ``<prefix><i>.weight`` tensors sorted by layer index (LayerNorm
    weights are 1-D ``<i>.ln.weight`` and excluded)."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.weight$")
    found = sorted(
        (int(match.group(1)), value)
        for key, value in state_dict.items()
        if (match := pattern.match(key)) and value.ndim == 2
    )
    if not found:
        raise KeyError(f"No linear layers under {prefix!r}.")
    return [value for _, value in found]


def infer_architecture(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Shape-derived ``MultitaskWorldModel`` hyperparameters from a renamed
    state_dict."""
    if "task_emb.weight" not in state_dict:
        raise KeyError("No _task_emb.weight: not a multitask TD-MPC2 checkpoint.")
    num_tasks, task_dim = state_dict["task_emb.weight"].shape
    encoder = _linear_weights(state_dict, "encoder.")
    enc_dim = encoder[0].shape[0]
    obs_dim = encoder[0].shape[1] - task_dim
    latent_dim = encoder[-1].shape[0]
    dynamics = _linear_weights(state_dict, "_dynamics.")
    mlp_dim = dynamics[0].shape[0]
    action_dim = dynamics[0].shape[1] - latent_dim - task_dim
    num_bins = _linear_weights(state_dict, "_reward.")[-1].shape[0]
    return {
        "num_tasks": int(num_tasks),
        "task_dim": int(task_dim),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "latent_dim": int(latent_dim),
        "enc_dim": int(enc_dim),
        # upstream encoder hidden layers = max(num_enc_layers - 1, 1)
        "num_enc_layers": len(encoder),
        "mlp_dim": int(mlp_dim),
        "num_bins": int(num_bins),
    }


def convert_state_dict(
    source: dict[str, torch.Tensor],
    *,
    action_dims: Optional[Sequence[int]] = None,
    tasks: Optional[Sequence[str]] = None,
    simnorm_dim: int = 8,
    vmin: float = -10.0,
    vmax: float = 10.0,
    source_name: str = "",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[str]]:
    """Returns ``(state_dict, manifest, dropped_keys)``; validates the result
    with a strict load into a matching ``MultitaskWorldModel``."""
    dropped = sorted(key for key in source if _is_dropped(key))
    converted = {_rename(key): value.detach().clone() for key, value in source.items() if not _is_dropped(key)}
    arch = infer_architecture(converted)

    if action_dims is None and "_action_masks" in source:
        action_dims = [int(row.sum().item()) for row in source["_action_masks"]]
    if action_dims is None:
        raise ValueError(
            "Per-task action dims are unknown: the checkpoint has no _action_masks buffer; "
            "pass --action_dims or --dataset_dir."
        )
    action_dims = [int(d) for d in action_dims]
    if len(action_dims) != arch["num_tasks"] or max(action_dims) != arch["action_dim"]:
        raise ValueError(
            f"action_dims {action_dims} inconsistent with checkpoint "
            f"(num_tasks={arch['num_tasks']}, padded action_dim={arch['action_dim']})."
        )
    if tasks is not None and len(tasks) != arch["num_tasks"]:
        raise ValueError(f"{len(tasks)} task names for a {arch['num_tasks']}-task checkpoint.")

    masks = torch.zeros(arch["num_tasks"], arch["action_dim"])
    for i, dim in enumerate(action_dims):
        masks[i, :dim] = 1.0
    converted["_action_masks"] = masks

    manifest: dict[str, Any] = {
        **arch,
        "action_dims": action_dims,
        "tasks": list(tasks) if tasks is not None else None,
        "simnorm_dim": int(simnorm_dim),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "source": source_name,
    }

    from rl_garden.algorithms.pwm import build_world_model_from_manifest

    build_world_model_from_manifest(manifest).load_state_dict(converted, strict=True)
    return converted, manifest, dropped


def load_upstream_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        import tensordict  # noqa: F401  (registers TensorDict types for unpickling)
    except ImportError:
        pass
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return {key: torch.as_tensor(value) for key, value in dict(state_dict).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Upstream/HF TD-MPC2 multitask agent .pt file.")
    parser.add_argument("--dst_dir", required=True, help="Output directory (world_model.pt + manifest.json).")
    parser.add_argument("--dataset_dir", default=None, help="Converted dataset dir; its manifest gives tasks/action_dims.")
    parser.add_argument("--action_dims", type=int, nargs="+", default=None, help="Per-task action dims.")
    parser.add_argument("--simnorm_dim", type=int, default=8)
    parser.add_argument("--vmin", type=float, default=-10.0)
    parser.add_argument("--vmax", type=float, default=10.0)
    args = parser.parse_args()

    tasks = None
    action_dims = args.action_dims
    if args.dataset_dir is not None:
        dataset_manifest = json.loads((Path(args.dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
        tasks = dataset_manifest["tasks"]
        action_dims = action_dims or dataset_manifest["action_dims"]

    source = load_upstream_checkpoint(Path(args.src))
    state_dict, manifest, dropped = convert_state_dict(
        source,
        action_dims=action_dims,
        tasks=tasks,
        simnorm_dim=args.simnorm_dim,
        vmin=args.vmin,
        vmax=args.vmax,
        source_name=str(args.src),
    )
    if args.dataset_dir is not None and manifest["obs_dim"] != max(dataset_manifest["obs_dims"]):
        raise ValueError(
            f"Checkpoint padded obs_dim {manifest['obs_dim']} != dataset {max(dataset_manifest['obs_dims'])}."
        )

    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, dst / "world_model.pt")
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[convert] dropped {len(dropped)} keys: {dropped}", flush=True)
    print(f"[convert] kept {len(state_dict)} keys; manifest: {json.dumps(manifest)}", flush=True)


if __name__ == "__main__":
    main()
