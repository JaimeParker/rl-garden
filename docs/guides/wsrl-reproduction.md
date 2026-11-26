# Reproducing WSRL Offline-to-Online Training

This guide walks through the state-based ManiSkill PickCube WSRL workflow:

1. Train a state SAC policy and save checkpoints.
2. Use SAC checkpoints to generate a WSRL-compatible H5 dataset.
3. Run WSRL offline pre-training followed by online fine-tuning.

It covers PickCube only. The paper-aligned presets for AntMaze, Adroit and
Kitchen are the `configs/off2on/wsrl_*_paper.yaml` files (for example
`configs/off2on/wsrl_antmaze_large_play_v2_paper.yaml`); they use their own
D4RL datasets and do not need Stages 1-3.

For algorithm details and individual configuration options, see
[`docs/design/wsrl-overview.md`](../design/wsrl-overview.md).

## Prerequisites

All commands run from the repository root in an environment with rl-garden and
its ManiSkill dependencies installed and a CUDA GPU available. Use
`CUDA_VISIBLE_DEVICES=<gpu-id>` to pick a GPU. Run directories are written to
`runs/<exp_name>/`.

## Stage 1: Train SAC Checkpoints

Train a state-based SAC policy on PickCube and save periodic checkpoints. The
working baseline used 2M environment steps; success was effectively saturated
after roughly 600k steps.

```bash
python -u examples/train_online.py sac \
  --env_id PickCube-v1 \
  --num_envs 16 \
  --total_timesteps 2000000 \
  --checkpoint_freq 200000 \
  --eval_freq 10000 \
  --num_eval_steps 50 \
  --capture_video \
  --video_fps 30 \
  --render_mode rgb_array \
  --exp_name pickcube_sac_state_2m_seed1 \
  --log_type wandb
```

Expected checkpoints:

```text
runs/pickcube_sac_state_2m_seed1/checkpoints/checkpoint_200000.pt
runs/pickcube_sac_state_2m_seed1/checkpoints/checkpoint_400000.pt
...
runs/pickcube_sac_state_2m_seed1/checkpoints/final.pt
```

For a long run, launch it under `tmux` or `nohup` and redirect output to a log
file.

## Stage 2: Select Checkpoints for Dataset Generation

Use checkpoint selection to score candidate policies before collection. The
dataset generator groups policies into failure, middle, near-success, and success
sources using success-rate thresholds. If no failure policy exists, random
actions are used as the failure fallback.

Recommended candidate checkpoints for the PickCube state baseline:

```text
checkpoint_200000.pt
checkpoint_400000.pt
checkpoint_600000.pt
checkpoint_800000.pt
checkpoint_1000000.pt
```

Create a source directory with the candidate checkpoints. Symlinks are fine:

```bash
mkdir -p runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints && \
  cd runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints && \
  ln -sf ../../checkpoints/checkpoint_200000.pt . && \
  ln -sf ../../checkpoints/checkpoint_400000.pt . && \
  ln -sf ../../checkpoints/checkpoint_600000.pt . && \
  ln -sf ../../checkpoints/checkpoint_800000.pt . && \
  ln -sf ../../checkpoints/checkpoint_1000000.pt . && \
  cd -
```

Run selection only:

```bash
python -u examples/generate.py wsrl_dataset \
  --checkpoint_dir runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints \
  --output_path runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.h5 \
  --total_transitions 200000 \
  --num_envs 16 \
  --eval_episodes 50 \
  --policy_mix 0.3 0.3 0.3 0.1 \
  --tier_thresholds 0.2 0.6 0.8 \
  --selection_only \
  --report_path runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.selection.json
```

`--policy_mix` gives the sampling weights of the failure, middle,
near-success, and success tiers. Review the report before collection:

```bash
python -m json.tool runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.selection.json | head -120
```

## Stage 3: Generate the WSRL Dataset

Use the selection report as the source policy plan. This avoids re-deciding the
policy tiers when generating the actual H5.

```bash
python -u examples/generate.py wsrl_dataset \
  --checkpoint_dir runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints \
  --output_path runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.h5 \
  --total_transitions 200000 \
  --num_envs 16 \
  --eval_episodes 50 \
  --policy_mix 0.3 0.3 0.3 0.1 \
  --tier_thresholds 0.2 0.6 0.8 \
  --source_report_path runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.selection.json \
  --report_path runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.collection.json
```

Expected H5 path:

```text
runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.h5
```

Inspect the dataset structure and one transition:

```python
import h5py
import numpy as np

path = "runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.h5"
np.set_printoptions(precision=4, suppress=True, linewidth=160)


def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}/ attrs={dict(obj.attrs)}")


with h5py.File(path, "r") as f:
    print("ROOT attrs:", dict(f.attrs))
    first = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[0]
    print("First trajectory:", first)
    f[first].visititems(walk)

    step = 0
    g = f[first]
    print("obs[step]:", g["obs/state"][step])
    print("action[step]:", g["actions"][step])
    print("reward[step]:", g["rewards"][step])
    print("terminated[step]:", g["terminated"][step])
    print("truncated[step]:", g["truncated"][step])
    print("next_obs[step]:", g["obs/state"][step + 1])
```

Observations are stored as an H5 group (`obs/state` for the state-based
dataset), not a single `obs` array.

## Stage 4: Run WSRL Offline-to-Online

CQL and Cal-QL are available as standalone algorithms as well as WSRL flow
components. Use one of two paths:

- **Standalone offline pretrain**: `examples/pretrain_offline.py cql`
  or `examples/pretrain_offline.py calql` trains `CQL` / `CalQL` without
  constructing a simulator env. The `wsrl` subcommand builds the
  WSRL-compatible Cal-QL pretrain path.
- **WSRL offline→online**: `examples/train_off2on.py wsrl` builds `WSRL`, runs
  offline updates, switches replay mode, then continues online interaction.

The smoke command below uses the WSRL offline→online path: 20k offline updates
and 50k online environment steps. The online step counter continues from the
offline step, so this run logs from 0 to roughly 70k global steps. The online
CQL loss is off by default (`--no-online_use_cql_loss` is the explicit form).

```bash
python -u examples/train_off2on.py wsrl \
  --env_id PickCube-v1 \
  --num_envs 16 \
  --offline_dataset runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.h5 \
  --num_offline_steps 20000 \
  --num_online_steps 50000 \
  --online_replay_mode empty \
  --n_critics 10 \
  --critic_subsample_size 2 \
  --use_calql \
  --checkpoint_freq 10000 \
  --log_freq 1000 \
  --eval_freq 10000 \
  --num_eval_steps 50 \
  --capture_video \
  --video_fps 30 \
  --render_mode rgb_array \
  --exp_name pickcube_wsrl_state_smoke_20koff_50kon_seed1 \
  --log_type wandb
```

Standalone CQL/Cal-QL pretrain smoke on the generated H5:

```bash
python -u examples/pretrain_offline.py cql \
  --offline_dataset runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k.h5 \
  --num_offline_steps 20000 \
  --checkpoint_dir runs/pickcube_cql_state_smoke_20koff_seed1/checkpoints \
  --buffer_device cuda \
  --n_critics 10 \
  --critic_subsample_size 2 \
  --checkpoint_freq 10000 \
  --log_freq 1000 \
  --log_type wandb
```

Switch the `cql` subcommand to `calql` to run standalone Cal-QL. Use `wsrl`
when the checkpoint should resume through the WSRL offline→online path. The
default final checkpoint names are `cql_offline_pretrained.pt`,
`calql_offline_pretrained.pt`, and `wsrl_offline_pretrained.pt`.

Expected checkpoint layout of the WSRL run:

```text
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/checkpoint_10000.pt
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/checkpoint_20000.pt
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/offline_final.pt
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/checkpoint_30000.pt
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/checkpoint_40000.pt
...
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/final.pt
```

The exact online checkpoint numbers can overshoot by up to `num_envs` because
the online loop advances in vectorized-environment increments.

## Monitoring and Artifacts

- Use `--log_type wandb` for shared experiment tracking, and run `wandb status`
  before a long run to confirm credentials.
- Logs, checkpoints, and videos are written under `runs/` and `wandb/`.

## Troubleshooting

- Online W&B metrics stop at the offline step: online training must continue
  from the offline `global_step`, which `examples/train_off2on.py wsrl` does.
- Dataset loading fails due to missing Monte Carlo returns: use the WSRL dataset
  generator from this repository; the ManiSkill H5 loader computes/caches MC
  returns when loading into MC replay buffers.
