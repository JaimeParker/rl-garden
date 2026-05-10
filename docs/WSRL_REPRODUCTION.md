# Reproducing WSRL Offline-to-Online Training

This guide describes the current state-based PickCube WSRL workflow:

1. Train a state SAC policy and save checkpoints.
2. Use SAC checkpoints to generate a WSRL-compatible H5 dataset.
3. Run WSRL offline pre-training followed by online fine-tuning.

For algorithm details and individual configuration options, see
[`docs/WSRL_README.md`](WSRL_README.md). This document is an operator guide for
collaborators who need to reproduce the workflow end to end.

## Environment Assumptions

The commands in this guide use the shared remote setup as an example. In the
examples below, the server is `6017`, but the same workflow can be run on any
server or directly inside a Docker container that has the same project,
dependencies, GPU access, and artifact paths.

Example remote setup:

- SSH alias: `6017`
- Container: `liuzhaohong_maniskill_rlgarden`
- Container workspace: `/workspace/rl-garden`
- Python environment: `/opt/venv/openvla/bin`
- Local-to-remote code sync: Mutagen session `rl-garden-code`

If using this remote setup, the container is expected to bind-mount the remote
host project into `/workspace/rl-garden`. After local edits, first verify that
Mutagen is healthy:

```bash
cd /home/hazyparker/Projects/rl-garden
bash scripts/mutagen_ensure.sh
```

Expected state:

```text
Status: Watching for changes
```

There should be no `Conflicts` section. `runs/`, `logs/`, and `wandb/` are
ignored by Mutagen because they are training artifacts.

Before launching a long run through `6017`, verify the container can see the
project:

```bash
ssh 6017 "docker exec liuzhaohong_maniskill_rlgarden bash -lc '
  cd /workspace/rl-garden &&
  pwd &&
  git status --short | head
'"
```

All training/data-generation commands should run inside the container with the
OpenVLA virtualenv on `PATH`. You can either use the `ssh 6017 "docker exec ..."`
wrappers shown below, or run the inner commands directly from a server shell or
an interactive Docker shell:

```bash
export PATH=/opt/venv/openvla/bin:$PATH
export PYTHONPATH=/workspace/rl-garden:${PYTHONPATH:-}
```

## Stage 1: Train SAC Checkpoints

Train a state-based SAC policy on PickCube and save periodic checkpoints. The
working baseline used 2M environment steps; in previous runs, success was
effectively saturated after roughly 600k steps.

Example long-run launch on GPU 1:

```bash
ssh 6017 "mkdir -p /data0/liuzhaohong/Projects/rl-garden/logs && \
  tmux new-session -d -s rlg_pickcube_sac_state_2m \
  \"docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc ' \
    cd /workspace/rl-garden && \
    export PATH=/opt/venv/openvla/bin:\$PATH && \
    export PYTHONPATH=/workspace/rl-garden:\${PYTHONPATH:-} && \
    MPLCONFIGDIR=/tmp python -u examples/train_sac_state.py \
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
      --log_type wandb \
  ' 2>&1 | tee /data0/liuzhaohong/Projects/rl-garden/logs/rlg_pickcube_sac_state_2m_\$(date +%Y%m%d_%H%M%S).log\""
```

Expected run directory:

```text
/workspace/rl-garden/runs/pickcube_sac_state_2m_seed1
```

Expected checkpoints:

```text
runs/pickcube_sac_state_2m_seed1/checkpoints/checkpoint_200000.pt
runs/pickcube_sac_state_2m_seed1/checkpoints/checkpoint_400000.pt
...
runs/pickcube_sac_state_2m_seed1/checkpoints/final.pt
```

Monitor the run:

```bash
ssh 6017 "tmux attach -t rlg_pickcube_sac_state_2m"
ssh 6017 "tail -f /data0/liuzhaohong/Projects/rl-garden/logs/rlg_pickcube_sac_state_2m_*.log"
```

## Stage 2: Select Checkpoints for Dataset Generation

Use checkpoint selection to score candidate policies before collection. The
dataset generator groups policies into failure, near-success, and success
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
ssh 6017 "mkdir -p /data0/liuzhaohong/Projects/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints && \
  cd /data0/liuzhaohong/Projects/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints && \
  ln -sf ../../checkpoints/checkpoint_200000.pt . && \
  ln -sf ../../checkpoints/checkpoint_400000.pt . && \
  ln -sf ../../checkpoints/checkpoint_600000.pt . && \
  ln -sf ../../checkpoints/checkpoint_800000.pt . && \
  ln -sf ../../checkpoints/checkpoint_1000000.pt ."
```

Run selection only:

```bash
ssh 6017 "docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc '
  cd /workspace/rl-garden &&
  export PATH=/opt/venv/openvla/bin:\$PATH &&
  export PYTHONPATH=/workspace/rl-garden:\${PYTHONPATH:-} &&
  MPLCONFIGDIR=/tmp python -u examples/generate_wsrl_dataset.py \
    --checkpoint_dir /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints \
    --output_path /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.h5 \
    --total_transitions 200000 \
    --obs_mode state \
    --num_envs 16 \
    --eval_episodes 50 \
    --policy_mix 0.3 0.3 0.4 \
    --tier_thresholds 0.2 0.8 \
    --selection_only \
    --report_path /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.selection.json
'"
```

Review the report before collection:

```bash
ssh 6017 "docker exec liuzhaohong_maniskill_rlgarden bash -lc '
  export PATH=/opt/venv/openvla/bin:\$PATH
  python -m json.tool /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.selection.json | head -120
'"
```

## Stage 3: Generate the WSRL Dataset

Use the selection report as the source policy plan. This avoids re-deciding the
policy tiers when generating the actual H5.

```bash
ssh 6017 "docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc '
  cd /workspace/rl-garden &&
  export PATH=/opt/venv/openvla/bin:\$PATH &&
  export PYTHONPATH=/workspace/rl-garden:\${PYTHONPATH:-} &&
  MPLCONFIGDIR=/tmp python -u examples/generate_wsrl_dataset.py \
    --checkpoint_dir /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_dataset_sources_200k_1m/checkpoints \
    --output_path /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.h5 \
    --total_transitions 200000 \
    --obs_mode state \
    --num_envs 16 \
    --eval_episodes 50 \
    --policy_mix 0.3 0.3 0.4 \
    --tier_thresholds 0.2 0.8 \
    --source_report_path /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.selection.json \
    --report_path /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.collection.json
'"
```

Expected H5 path:

```text
/workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.h5
```

Inspect the dataset structure:

```bash
ssh 6017 "docker exec liuzhaohong_maniskill_rlgarden bash -lc '
  export PATH=/opt/venv/openvla/bin:\$PATH
  python - <<\"PY\"
import h5py

path = \"/workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.h5\"

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f\"{name}: shape={obj.shape}, dtype={obj.dtype}\")
    else:
        print(f\"{name}/ attrs={dict(obj.attrs)}\")

with h5py.File(path, \"r\") as f:
    print(\"ROOT attrs:\", dict(f.attrs))
    first = sorted(f.keys(), key=lambda k: int(k.split(\"_\")[1]))[0]
    print(\"\\nFirst trajectory:\", first)
    f[first].visititems(walk)
PY
'"
```

Inspect one transition:

```bash
ssh 6017 "docker exec liuzhaohong_maniskill_rlgarden bash -lc '
  export PATH=/opt/venv/openvla/bin:\$PATH
  python - <<\"PY\"
import h5py
import numpy as np

path = \"/workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.h5\"
traj = \"traj_0\"
step = 0
np.set_printoptions(precision=4, suppress=True, linewidth=160)

with h5py.File(path, \"r\") as f:
    g = f[traj]
    print(\"traj attrs:\", dict(g.attrs))
    print(\"obs[step]:\", g[\"obs\"][step])
    print(\"action[step]:\", g[\"actions\"][step])
    print(\"reward[step]:\", g[\"rewards\"][step])
    print(\"terminated[step]:\", g[\"terminated\"][step])
    print(\"truncated[step]:\", g[\"truncated\"][step])
    print(\"next_obs[step]:\", g[\"obs\"][step + 1])
PY
'"
```

## Stage 4: Run WSRL Offline-to-Online

For a first smoke run, use 20k offline updates and 50k online environment steps.
The online step counter continues from the offline step, so this run logs from
0 to roughly 70k global steps. The offline phase saves `offline_final.pt`.

```bash
ssh 6017 "mkdir -p /data0/liuzhaohong/Projects/rl-garden/logs && \
  tmux new-session -d -s rlg_wsrl_pickcube_smoke \
  \"docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc ' \
    cd /workspace/rl-garden && \
    export PATH=/opt/venv/openvla/bin:\$PATH && \
    export PYTHONPATH=/workspace/rl-garden:\${PYTHONPATH:-} && \
    MPLCONFIGDIR=/tmp python -u examples/train_wsrl.py \
      --env_id PickCube-v1 \
      --num_envs 16 \
      --offline_dataset_path /workspace/rl-garden/runs/pickcube_sac_state_2m_seed1/wsrl_datasets/pickcube_state_wsrl_200k_mix_30_30_40_200k_1m.h5 \
      --num_offline_steps 20000 \
      --num_online_steps 50000 \
      --online_replay_mode empty \
      --online_use_cql_loss False \
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
      --log_type wandb \
  ' 2>&1 | tee /data0/liuzhaohong/Projects/rl-garden/logs/rlg_wsrl_pickcube_smoke_\$(date +%Y%m%d_%H%M%S).log\""
```

Expected checkpoint layout:

```text
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/offline_final.pt
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/checkpoint_30000.pt
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/checkpoint_40000.pt
...
runs/pickcube_wsrl_state_smoke_20koff_50kon_seed1/checkpoints/final.pt
```

The exact online checkpoint numbers can overshoot by up to `num_envs` because
the online loop advances in vectorized-environment increments.

Monitor the run:

```bash
ssh 6017 "tmux attach -t rlg_wsrl_pickcube_smoke"
ssh 6017 "tail -f /data0/liuzhaohong/Projects/rl-garden/logs/rlg_wsrl_pickcube_smoke_*.log"
```

## Monitoring and Artifacts

W&B:

- Use `--log_type wandb` for shared experiment tracking.
- After recreating the container, verify credentials before launching:

```bash
ssh 6017 "docker exec liuzhaohong_maniskill_rlgarden bash -lc '
  export PATH=/opt/venv/openvla/bin:\$PATH &&
  wandb status
'"
```

Remote artifacts:

```text
/data0/liuzhaohong/Projects/rl-garden/logs
/data0/liuzhaohong/Projects/rl-garden/runs
/data0/liuzhaohong/Projects/rl-garden/wandb
```

These directories are ignored by Mutagen. Inspect them on `6017` or copy
specific files back manually if needed.

If container-created artifacts are root-owned and host-side cleanup is blocked:

```bash
ssh 6017 "sudo chown -R liuzhaohong:liuzhaohong \
  /data0/liuzhaohong/Projects/rl-garden/runs \
  /data0/liuzhaohong/Projects/rl-garden/logs \
  /data0/liuzhaohong/Projects/rl-garden/wandb 2>/dev/null || true"
```

## Troubleshooting

- `mutagen_ensure.sh` reports conflicts: run `mutagen sync list rl-garden-code --long` and check for remote-only artifacts. `wandb/`, `runs/`, and `logs/` should stay ignored.
- `python` is missing in the container: prepend `export PATH=/opt/venv/openvla/bin:$PATH`.
- W&B login fails after container rebuild: copy or reconfigure credentials, then verify with `wandb status`.
- Container code differs from remote host code: verify the bind mount with `cd /workspace/rl-garden && git status --short`; do not use `docker cp` unless the bind mount is absent.
- Online W&B metrics stop at the offline step: ensure `examples/train_wsrl.py` is the version where online training continues from the offline `global_step`.
- Dataset loading fails due missing Monte Carlo returns: use the WSRL dataset generator from this repository; the ManiSkill H5 loader computes/caches MC returns when loading into MC replay buffers.
