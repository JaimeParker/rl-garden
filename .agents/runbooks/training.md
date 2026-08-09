# Training Runbook

Use this runbook to select and validate a training entrypoint. For remote execution,
also read [the remote training SOP](../rules/remote-training-sop.md) and the ignored
`.agents/local/personal_config.md` before running any command.

## Registry Entrypoints

The first positional argument selects the registered algorithm:

```bash
# Online: sac, ppo, recurrent/transformer variants, drqv2, flash_sac,
# residual_sac, td3, rlpd, rlpd_hybrid, tdmpc2
python examples/train_online.py sac --env_id PickCube-v1 --obs_mode state

# Offline: bc, iql, cql, calql, wsrl, awac, td3_bc, tdmpc2_multitask
python examples/pretrain_offline.py calql \
  --offline_dataset_path demos/pickcube.h5 --num_offline_steps 700000

# Offline-to-online: wsrl, calql, iql, awac
python examples/train_off2on.py wsrl --env_id PickCube-v1 --obs_mode state
```

Use the scripts when their experiment defaults are wanted:

```bash
scripts/train_sac_state.sh
scripts/train_sac_rgbd.sh --encoder resnet10
scripts/train_ppo_state.sh
scripts/train_ppo_rgbd.sh --encoder plain_conv
scripts/train_drqv2_rgb.sh
scripts/train_wsrl.sh
scripts/train_wsrl_rgbd.sh
scripts/pretrain_offline.sh calql --offline_dataset_path demos/pickcube.h5
```

## Configuration Inspection

All registry-managed entrypoints accept a strict, args-only YAML preset. CLI flags
have final precedence:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml --gamma 0.95
```

Use `--print-config` for pure parsing and static validation. It exits before
loading a simulator or creating environments, loggers, agents, and run directories:

```bash
python examples/train_online.py sac --obs_mode state --print-config
python examples/pretrain_offline.py calql \
  --offline_dataset_path demos/pickcube.h5 --print-config
python examples/train_off2on.py wsrl --obs_mode rgb --print-config
```

Use `--dry-run` when the selected backend is available. It materializes the
environment request, backend configs, spaces, devices, and agent, then exits before
training, W&B initialization, full dataset/replay loading, or checkpoint writes.
If configured, checkpoint agent state is validated and loaded without its replay
buffer at the same phase as a normal run:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml --dry-run
```

Use `--explain-param FIELD` to inspect a value, source history, ownership, active
condition, definition location, and mapped consumer. The three inspection actions
are mutually exclusive.

Normal runs atomically save the same EffectiveConfig v2 schema to
`{log_dir}/{run_name}/config.json`: first `status: preflight`, then
`status: materialized` after environment and agent construction. Explicit CLI
flags override `RLG_*` logging environment defaults.

## Offline Evaluation

Offline pretraining does not create a simulator when `--env_id` is omitted. Pass an
environment to enable periodic evaluation:

```bash
python examples/pretrain_offline.py iql \
  --offline_dataset_path demos/pickcube.h5 \
  --env_id PickCube-v1 --eval_freq 10000 --num_eval_steps 50
```

## Peg Training

Use the dedicated entrypoint or launcher for `PegInsertionSidePegOnly-v1`:

```bash
scripts/train_sac_rgbd_peg.sh
# or
python examples/train_sac_rgbd_peg.py
```

It owns the peg-specific robot, controller, camera, and per-key image defaults. Do
not copy those defaults into `train_online.py`.

For a short local CPU simulator compatibility smoke test:

```bash
MPLCONFIGDIR=/tmp python examples/train_sac_rgbd_peg.py \
  --num_envs 1 \
  --num_eval_envs 1 \
  --total_timesteps 8 \
  --learning_starts 4 \
  --training_freq 4 \
  --batch_size 2 \
  --buffer_size 64 \
  --buffer_device cpu \
  --sim_backend physx_cpu \
  --render_backend gpu \
  --eval_freq 0 \
  --log_freq 4
```

This is only a smoke fallback. Keep normal training CUDA-first. Set
`MPLCONFIGDIR=/tmp` when matplotlib cache warnings appear.

## Preflight Checklist

1. Confirm the intended registry phase and algorithm.
2. Run `--print-config`; inspect selection, inputs, active backend, provenance,
   logging, checkpoint, and algorithm defaults.
3. Run `--explain-param` for every high-risk override.
4. Confirm dataset and checkpoint paths exist when applicable.
5. Run `--dry-run` when the backend and hardware dependencies are available.
6. Use the smallest relevant smoke before launching a long run.
7. For checkpoint behavior, follow [the checkpoint runbook](checkpoint.md).
