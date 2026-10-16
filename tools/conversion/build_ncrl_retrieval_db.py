"""Build NCRL's experience-rehearsal retrieval DB (official ``3rd_party/ncrl/src/prepare_dataset_kv.py``).

For every episode of a converted sequence dataset
(``tools/conversion/convert_ncrl_dataset.py`` output, one directory per
embodiment) the first frame is embedded with the encoder of an
``RSSMPretrain`` checkpoint (``RSSM.encode``, float32, no autocast). The
online algorithm (``rl_garden.algorithms.ncrl.NCRL``) embeds its
``env.reset()`` frame with the same encoder and retrieves the nearest
episodes by L2 distance.

Output (``torch.save``)::

    embedding       float16 (N, D)
    embodiment      list[str] (N,)
    episode_id      int64 (N,)   per-embodiment index in episodes.pt
    action_dim      int64 (N,)   real (unpadded) action dim
    episode_length  int64 (N,)
    metadata        {format, checkpoint, dataset_dir, obs_key,
                     encoder_fingerprint, padded_action_dim, embedding_dim}

``encoder_fingerprint`` is checked when NCRL loads the DB, so a DB built from
another checkpoint is rejected. Paths are stored resolved for reference only.

Usage:
    python tools/conversion/build_ncrl_retrieval_db.py \\
        --dataset_dir /data/ncrl/seq_v1/dmcontrol \\
        --checkpoint runs/rssm_pretrain/checkpoints/final.pt \\
        --out /data/ncrl/seq_v1/dmcontrol_retrieval_db.pt --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_garden.algorithms.ncrl import (  # noqa: E402
    RETRIEVAL_DB_FORMAT,
    encoder_fingerprint,
    load_rssm_pretrain_checkpoint,
    retrieval_embedding,
    rssm_from_pretrain_checkpoint,
)
from rl_garden.buffers.sequence_dataset import (  # noqa: E402
    OBS_FILE,
    read_part_manifest,
    sequence_dataset_part_dirs,
)


def build_retrieval_db(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    checkpoint = load_rssm_pretrain_checkpoint(checkpoint_path)
    world_model = rssm_from_pretrain_checkpoint(checkpoint).to(device).eval()
    padded_action_dim = int(checkpoint["metadata"]["hyperparameters"]["padded_action_dim"])

    embeddings: list[torch.Tensor] = []
    embodiments: list[str] = []
    episode_ids: list[int] = []
    action_dims: list[int] = []
    lengths: list[int] = []
    obs_key = None
    for part_dir in sequence_dataset_part_dirs(dataset_dir):
        manifest = read_part_manifest(part_dir)
        if obs_key is None:
            obs_key = manifest["obs_key"]
        elif manifest["obs_key"] != obs_key:
            raise ValueError(f"{part_dir}: obs_key {manifest['obs_key']!r} != {obs_key!r}.")
        if int(manifest["padded_action_dim"]) != padded_action_dim:
            raise ValueError(
                f"{part_dir}: padded_action_dim {manifest['padded_action_dim']} != checkpoint's {padded_action_dim}."
            )
        part_lengths = np.asarray(manifest["episode_lengths"], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(part_lengths)[:-1]])
        obs = np.load(part_dir / OBS_FILE, mmap_mode="r")
        for begin in range(0, starts.size, batch_size):
            first_frames = torch.from_numpy(np.ascontiguousarray(obs[starts[begin : begin + batch_size]]))
            embedding = retrieval_embedding(world_model, {obs_key: first_frames.to(device)})
            embeddings.append(embedding.to("cpu", torch.float16))
        embodiments += [manifest["embodiment"]] * starts.size
        episode_ids += list(range(starts.size))
        action_dims += [int(manifest["action_dim"])] * starts.size
        lengths += part_lengths.tolist()
        print(f"[retrieval_db] {manifest['embodiment']}: {starts.size} episodes", flush=True)

    embedding = torch.cat(embeddings)
    return {
        "embedding": embedding,
        "embodiment": embodiments,
        "episode_id": torch.tensor(episode_ids, dtype=torch.int64),
        "action_dim": torch.tensor(action_dims, dtype=torch.int64),
        "episode_length": torch.tensor(lengths, dtype=torch.int64),
        "metadata": {
            "format": RETRIEVAL_DB_FORMAT,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "obs_key": obs_key,
            "encoder_fingerprint": encoder_fingerprint(world_model.encoder),
            "padded_action_dim": padded_action_dim,
            "embedding_dim": int(embedding.shape[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, help="Converted dataset root (one subdirectory per embodiment).")
    parser.add_argument("--checkpoint", required=True, help="RSSMPretrain checkpoint whose encoder embeds the frames.")
    parser.add_argument("--out", required=True, help="Output .pt path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    db = build_retrieval_db(args.dataset_dir, args.checkpoint, device=args.device, batch_size=args.batch_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(db, out)
    print(f"[retrieval_db] {db['embedding'].shape[0]} episodes x {db['embedding'].shape[1]} dims -> {out}", flush=True)


if __name__ == "__main__":
    main()
