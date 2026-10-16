# NCRL: Pretrained World Model + Rehearsal + Guidance

NCRL is an offline-to-online visual RL method built on Dreamer. It runs in two stages:

1. **Reward-free world-model pretraining** on a mixed-quality, multi-embodiment image
   dataset (`rssm_pretrain`, offline).
2. **Online fine-tuning** on one task (`ncrl`, online). The world model is the
   pretrained RSSM. The method adds two things:
   - **experience rehearsal**: offline episodes whose first frame resembles the task's
     are retrieved and mixed into every batch;
   - **execution guidance**: a behaviour-cloning prior takes over the actor inside
     random windows.

The actor-critic is DreamerV2-style. The actor is trained by backpropagating through
the learned dynamics, and there is no continue head.

Code:

- `rl_garden/algorithms/rssm_pretrain.py` (stage 1)
- `rl_garden/algorithms/ncrl.py` (stage 2)
- `rl_garden/buffers/rehearsal_sequence_buffer.py`
- `rl_garden/policies/bc_guide.py`
- `tools/conversion/{convert_ncrl_dataset,build_ncrl_retrieval_db}.py`

Deviations from the official code are listed in section 6.

## 1. Data preparation

The official dataset comes from HF `zhaoyi11/ncrl`. It is stored as one `.npz` file
per episode. Convert it once per benchmark:

```bash
python tools/conversion/convert_ncrl_dataset.py \
  --src_dir /data/ncrl/raw/ncrl_datasets/dmcontrol --dst_dir /data/ncrl/seq_v1/dmcontrol
```

This writes one directory per embodiment, each containing three files:

- `obs.npy`: uint8 `(N, 64, 64, 3)`, memory-mapped.
- `episodes.pt`
- `manifest.json`

Actions are zero-padded to the largest action dim in the benchmark (12 for dmcontrol;
deviation from the official code, which pads every embodiment to 21).

## 2. Stage 1: pretrain the RSSM

```bash
python examples/pretrain_offline.py rssm_pretrain \
  --dataset_dir /data/ncrl/seq_v1/dmcontrol --log_type none
```

Defaults:

| Setting | Value |
|---|---|
| Model size | `ncrl_m` (deter 4096, hidden 1024, 32x48, cnn 48) |
| Batch | 32 x 64 |
| Steps | 200k |
| Precision | bf16 |
| Free nats | `kl_free_mode=mean`, KL taken after 1% unimix (as official) |
| Stochastic-state gradient | probs straight-through (as official) |
| Backbone | `--backbone ncrl` |

`--backbone` picks the RSSM networks:

- `ncrl` (CLI default): the official NCRL networks. These are a stride-2 conv encoder,
  a transposed-conv decoder, a LayerNorm block-GRU without action normalization, and
  heads that read `deter` and `stoch` through separate branches, all with PyTorch
  default init. They need image-only 64x64 observations.
- `dreamer_v3`: the DreamerV3 encoder, decoder and RMSNorm networks. This is also
  the Python constructor default of `RSSMPretrain` and `NCRL`.

The checkpoint stores the RSSM under `policy.actor_extractor.*`, together with its
architecture metadata and its backbone.

## 3. Build the retrieval DB

The DB has to come from the same checkpoint that stage 2 loads:

```bash
python tools/conversion/build_ncrl_retrieval_db.py \
  --dataset_dir /data/ncrl/seq_v1/dmcontrol \
  --checkpoint <run>/checkpoints/rssm_pretrain_offline_pretrained.pt \
  --out /data/ncrl/seq_v1/dmcontrol_retrieval_db.pt
```

Each episode's first frame is embedded with the pretrained encoder. The DB stores the
float16 embeddings, the embodiment, the episode id, the action dim and the length of
each episode, and an encoder-weight fingerprint.

`ncrl` rejects a DB whose fingerprint does not match the loaded encoder. It reads the
episodes from the DB's recorded `dataset_dir`; use `--prior_dataset_dir` to override
that path, for example when the host and the container mount the data differently.

## 4. Stage 2: online fine-tuning

```bash
python examples/train_online.py ncrl \
  --env_id walker-walk \
  --pretrained_world_model <rssm_pretrain checkpoint> \
  --retrieval_db /data/ncrl/seq_v1/dmcontrol_retrieval_db.pt
```

`--backbone` (CLI default `ncrl`) must match the pretrained checkpoint's; otherwise
`ncrl` stops with an error. Checkpoints pretrained before the option existed are
`dreamer_v3` and need `--backbone dreamer_v3`.

These defaults match the official `finetune.yaml` with the `scripts/finetune.sh`
overrides:

| Setting | Value |
|---|---|
| Backend | dm_control, 1 env, image-only `rgb_default` 64x64 (no `state` key), `action_repeat` 2 |
| Run length | 250k policy steps (500k frames) |
| Batch | 16 x 64; `online_ratio` 0.75 (12 online + 4 prior rows) |
| Retrieval | 500 episodes |
| BC | 150k steps, batch 256 |
| Guidance schedule | `linear(1 -> 0.1, 100k policy steps)` |
| Warmup | `max(15k frames, 5 episodes)`, all BC actions |
| Learning starts | 10 episodes |
| Update ratio | `train_ratio` 256 |
| Imagination | horizon 16, discount 0.99, lambda 0.95, entropy 1e-4 |
| Reward head | 2 x 512, symlog two-hot (critic and slow critic use symlog two-hot too) |
| Policy | masked TruncatedNormal on the `ncrl` backbone, masked Normal on `dreamer_v3` |
| Learning rates | world model and reward head 1e-4; actor and critic 8e-5 |
| Optimizer | Adam eps 1e-5, clip 100, weight decay 1e-6 |
| Evaluation | 5 episodes every 10k frames, mean actions, no BC |

The environment's observations must match the pretrained encoder's key and shape
exactly. On backends other than dm_control, also pass `--episode_frames` and
`--action_repeat`.

Order of operations at the start of `learn()`:

1. The pretrained RSSM is loaded. The reward head, actor and critic are new.
2. Retrieval runs. Each iteration calls `env.reset()`, takes the first frame and finds
   the nearest same-action-dim episodes in the DB.
3. The prior buffer is filled with the retrieved episodes.
4. BC is trained on the prior buffer.
5. Online fine-tuning starts.

A resumed run (`--load_checkpoint`) skips steps 2–4. It reloads the checkpointed
episode ids and BC weights instead.

Memory: every stored step holds a uint8 frame (12 KiB) and the float32 RSSM carry
(`ncrl_m`: `(4096 + 32*48) * 4 B` = 22 KiB), about 34 KiB in total. The online buffer
(`--buffer_size 250000`) and the prior (500 episodes x 500 steps) are therefore about
8 GiB each, on top of roughly 12 GiB for the update itself (measured with B=16, T=64,
bf16). The defaults do not fit a 24 GB card on CUDA: use `--buffer_device cpu`, or a
smaller `--buffer_size` / `--num_retrieve_episodes`. The actor loss backpropagates through
16 imagined steps from B x T = 1024 starts, so the imagination pass dominates GPU
memory.

## 5. Ablation switches

| Variant | Flags | Behaviour |
|---|---|---|
| DreamerV3 from scratch | use `dreamer_v3` | `ncrl` refuses to run without `--pretrained_world_model` |
| +P (pretrained WM only) | `--online_ratio 1.0` | No retrieval DB needed, no prior rows, no BC (guidance needs the retrieved data, as in the official code). Warmup takes uniform-random actions |
| +P+ER | `--no-use_guidance` | Retrieval + 75/25 rehearsal batches; warmup is random, the actor acts afterwards |
| +P+ER+G | defaults | Full NCRL |

## 6. Deviations from the official code

Deviations specific to stage 2:

- **Rehearsal and retrieval**
  - Retrieval runs at most 10 iterations and never picks an episode twice.
    Ineligible action dims are filtered out before the top-k (officially after).
  - `torch.cdist` replaces faiss.
  - The replay buffer holds GPU tensors.
- **Optimizers**
  - The reward head has its own Adam.
- **Policy**
  - Actor samples are masked to the real action dims.
  - With `--backbone ncrl`, actions come from the official TruncatedNormal: `tanh`
    mean, std in [0.1, 1], straight-through clamp. With `dreamer_v3`, they come from
    a masked Normal clamped to [-1, 1]. Both backbones use an analytic entropy bonus
    instead of a single-sample estimate.
- **Training cadence**
  - Updates run every step, with the same replay ratio.
  - The online buffer keeps and writes back the RSSM carry (DreamerV3 latent replay).
    Prior rows start from a zero state, as every official chunk does.
- **Networks**
  - `--backbone ncrl` follows the official structure and init.
  - `--backbone dreamer_v3` keeps the DreamerV3 networks: RMSNorm, truncated-normal
    init, zero-initialised critic and reward outputs, and heads on the concatenated
    feature.
- **Precision**
  - `compute_dtype` defaults to bf16 on CUDA; the official fine-tuning runs in fp32.
