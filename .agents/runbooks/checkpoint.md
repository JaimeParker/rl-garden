# Checkpoint Runbook

The public checkpoint format and compatibility contract are documented in
[`docs/CHECKPOINT.md`](../../docs/CHECKPOINT.md). Read that document before changing
checkpoint code or relying on resume semantics.

## Save Location

Unless `--checkpoint_dir` is set, registry-managed training resolves checkpoints
under:

```text
{log_dir}/{run_name}/checkpoints/
```

Intermediate files use `checkpoint_<step>.pt`. Offline final filenames default to
`<algorithm>_offline_pretrained.pt`; `--save_filename` overrides the final name.
When both `--save_final_checkpoint=False` and `--checkpoint_freq=0`, checkpoint
output is disabled.

## Resume Checklist

1. Inspect the saved run's `config.json` and reconstruct compatible environment,
   observation, encoder, and policy settings.
2. Confirm that the target algorithm accepts the checkpoint class; legacy
   `OfflineCQL` and `OfflineCalQL` aliases are handled by checkpoint compatibility
   metadata.
3. For off-policy training, decide whether replay state is required. Model and
   optimizer state alone do not preserve replay distribution.
4. Save with `--save_replay_buffer` when the checkpoint is intended for exact
   off-policy continuation. Replay snapshots are separate
   `_replay_buffer.pt` files and can be large.
5. Pass `--load_checkpoint <path>` for loading. Keep replay loading enabled only
   when a matching replay snapshot should be restored.
6. Validate the resolved arguments with `--print-config` before a long resumed run.

Example offline load:

```bash
python examples/pretrain_offline.py calql \
  --offline_dataset_path demos/pickcube.h5 \
  --load_checkpoint runs/<run_name>/checkpoints/calql_offline_pretrained.pt
```

Do not commit checkpoints, replay snapshots, run directories, or generated logs.
