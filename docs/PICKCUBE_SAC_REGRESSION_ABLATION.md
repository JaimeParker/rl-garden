# PickCube SAC RGB ResNet Regression Ablation

Date: 2026-06-01

## Goal

Diagnose why the 4.29 branch run learned while the main branch run failed under
near-identical PickCube SAC RGB ResNet settings.

Reference runs:

- Working 4.29 run: `jp78745g`
  - W&B display name: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_3m__20260530_034230`
  - Final train success: `success_at_end=0.8125`, `success_once=0.875`
- Failed main run: `gdbiisvu`
  - W&B display name: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_3m_main__20260530_063953`
  - Final train success: `success_at_end=0`, `success_once=0.0625`

## Confirmed Issue: vmap Prototype CPU RNG Shift

The main-branch critic refactor from the old `nn.ModuleList` ensemble to the
fused `torch.func.vmap` implementation introduced an unintended CPU RNG shift.
The vmap path created a throwaway `_QHead` prototype after stacking the real
critic parameters:

```python
prototype = _QHead(**self._head_kwargs)
prototype.to("meta")
```

That prototype was never trained, but its Linear weight/bias initializers still
consumed CPU RNG. Because `DictReplayBuffer.sample()` uses CPU
`torch.randint()` for transition indices, the main branch sampled different
transitions from the first gradient update even with the same training seed.

The fix saves and restores CPU RNG around prototype construction:

```python
_cpu_rng = torch.get_rng_state()
prototype = _QHead(**self._head_kwargs)
torch.set_rng_state(_cpu_rng)
prototype.to("meta")
```

Later ablations showed that this fix is necessary for reproducibility but not
sufficient to recover the failing RGB ResNet training runs. The remaining issue
is tracked as a visual SAC training-dynamics problem in the sections below.

Other inspected paths looked equivalent for this experiment:

- Env wrapper: `FlattenRGBDObservationWrapper`, `per_camera_rgbd=False`
- Obs/control: `obs_mode=rgb`, `include_state=True`, `pd_joint_delta_pos`
- Encoder: `resnet10_pretrained_converted`, `stack_channels`, unfrozen
- Actor visual path: still uses stop-gradient for Dict observations
- SAC TD loss: still sums the two critic MSE losses in `SAC._td_loss`

## Code Change Under Test

Added `critic_impl` as an ablation switch:

- `critic_impl="vmap"`: current main behavior
- `critic_impl="legacy"`: pre-vmap `nn.ModuleList` critic behavior

Threaded through:

- `rl_garden/networks/actor_critic.py`
- `rl_garden/policies/sac_policy.py`
- `rl_garden/algorithms/sac.py`
- `rl_garden/common/cli_args.py`
- `examples/train_sac_rgbd.py`

Default remains `vmap` so existing main behavior is unchanged unless the CLI
explicitly passes `--critic-impl legacy`.

After the RNG fix, the default `vmap` path should preserve the same downstream
CPU RNG state as `legacy`. The switch remains useful for ablations and for
isolating any non-RNG behavioral difference in the fused critic.

## Fixed Training Parameters

The 1m ablations should match the successful 3m run except for
`total_timesteps=1000000` and `critic_impl`.

```text
env_id=PickCube-v1
obs_mode=rgb
include_state=True
control_mode=pd_joint_delta_pos
encoder=resnet10
encoder_features_dim=256
image_fusion_mode=stack_channels
pretrained_weights=resnet10_pretrained_converted
freeze_resnet_encoder=False
freeze_resnet_backbone=False
gamma=0.8
utd=0.25
num_envs=16
num_eval_envs=1
batch_size=512
buffer_size=200000
buffer_device=cuda
learning_starts=4000
training_freq=64
policy_lr=0.0003
q_lr=0.0003
tau=0.01
seed=1
eval_freq=99999999
num_eval_steps=50
log_freq=1000
log_type=wandb
wandb_entity=dalian0744-intel
wandb_project=rl-garden
wandb_group=rl-garden
capture_video=False
```

## Running Experiments

Started on 6017-nofwd at 2026-06-01 21:06 Asia/Shanghai, before the RNG fix was
confirmed:

- `critic_impl=vmap`
  - tmux: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_vmap`
  - GPU: 0
  - W&B run: `https://wandb.ai/dalian0744-intel/rl-garden/runs/53ojw3kc`
  - log: `/home/liuzhaohong/data0/Projects/rl-garden/logs/pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_vmap_20260601_210629.log`
- `critic_impl=legacy`
  - tmux: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_legacycritic`
  - GPU: 1
  - W&B run: `https://wandb.ai/dalian0744-intel/rl-garden/runs/a7zf90k7`
  - log: `/home/liuzhaohong/data0/Projects/rl-garden/logs/pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_legacycritic_20260601_210629.log`

Initial check: both runs reached training loop, first eval return was `1.7633`,
and both were around 1% progress without OOM. Because these runs were launched
before the fixed-vmap code was confirmed on the remote, use them as the
pre-fix/legacy ablation pair, not as the final fixed-vmap validation.

Started after syncing and verifying the RNG fix on 6017-nofwd:

- `critic_impl=vmap` with prototype CPU RNG preservation
  - tmux: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_vmap_rngfix`
  - GPU: 2
  - W&B run: `https://wandb.ai/dalian0744-intel/rl-garden/runs/sbux4y93`
  - log: `/home/liuzhaohong/data0/Projects/rl-garden/logs/pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_vmap_rngfix_20260601.log`

Initial check: run reached W&B and training loop, first eval return was
`1.7633`, GPU2 memory usage was about 10 GB, and no OOM occurred.

Exact fixed-vmap training command:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=2 liuzhaohong_maniskill_rlgarden bash -lc '
cd /workspace/rl-garden &&
/opt/venv/openvla/bin/python examples/train_sac_rgbd.py \
  --env-id PickCube-v1 \
  --obs-mode rgb \
  --include-state \
  --control-mode pd_joint_delta_pos \
  --encoder resnet10 \
  --encoder-features-dim 256 \
  --image-fusion-mode stack_channels \
  --pretrained-weights resnet10_pretrained_converted \
  --no-freeze-resnet-encoder \
  --no-freeze-resnet-backbone \
  --gamma 0.8 \
  --utd 0.25 \
  --num-envs 16 \
  --num-eval-envs 1 \
  --batch-size 512 \
  --buffer-size 200000 \
  --buffer-device cuda \
  --learning-starts 4000 \
  --training-freq 64 \
  --policy-lr 0.0003 \
  --q-lr 0.0003 \
  --tau 0.01 \
  --seed 1 \
  --eval-freq 99999999 \
  --num-eval-steps 50 \
  --log-freq 1000 \
  --log-type wandb \
  --wandb-entity dalian0744-intel \
  --wandb-project rl-garden \
  --wandb-group rl-garden \
  --exp-name pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_main_vmap_rngfix \
  --total-timesteps 1000000 \
  --critic-impl vmap \
  --no-capture-video
'
```

Seed sweep started on 6017-nofwd at 2026-06-01 23:26 Asia/Shanghai:

- tmux: `pickcube_sac_rgb_resnet10_seed_sweep_s2_s5`
- GPU: 3
- Driver log:
  `/home/liuzhaohong/data0/Projects/rl-garden/logs/pickcube_sac_rgb_resnet10_seed_sweep_s2_s5_driver_20260601.log`
- Runs are sequential to avoid OOM and compare seeds 2-5 for both
  `critic_impl=legacy` and `critic_impl=vmap`.
- First run:
  - exp: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_seed2_legacy`
  - W&B run: `https://wandb.ai/dalian0744-intel/rl-garden/runs/41x1o1jg`
  - log:
    `/home/liuzhaohong/data0/Projects/rl-garden/logs/pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_seed2_legacy_20260601.log`
  - initial check: entered training loop on GPU3, about 10 GB memory usage, no
    OOM.

## Verification Already Run

Local:

```bash
python -m py_compile rl_garden/networks/actor_critic.py rl_garden/policies/sac_policy.py rl_garden/algorithms/sac.py rl_garden/common/cli_args.py examples/train_sac_rgbd.py
pytest -q tests/test_networks.py tests/test_policy_kwargs.py tests/test_resnet_encoder.py
```

Result:

```text
70 passed, 1 skipped
```

Post-fix targeted check:

```bash
pytest -q tests/test_networks.py
```

Result:

```text
28 passed
```

The added test `test_ensemble_vmap_prototype_does_not_shift_cpu_rng` constructs
vmap and legacy critics under the same seed, then verifies that the CPU RNG
state and a replay-like `torch.randint()` sample are identical after
construction.

Remote smoke:

```text
PickCube-v1, rgb, include_state, resnet10_pretrained_converted,
stack_channels, critic_impl=legacy, total_timesteps=8, log_type=none
```

Result: exited successfully.

Remote post-sync RNG check:

```text
grep confirmed the fix in /workspace/rl-garden/rl_garden/networks/actor_critic.py
rng_equal True
sample_equal True
```

Full SAC vmap/legacy parity checks added in
`tests/test_sac_vmap_parity.py`. These tests compare policy initialization,
fixed-batch critic/actor/alpha/target updates, and CPU/CUDA RNG consumption.

Local:

```bash
pytest -q tests/test_sac_vmap_parity.py tests/test_networks.py
```

Result:

```text
32 passed, 1 skipped
```

Remote CUDA:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=3 liuzhaohong_maniskill_rlgarden \
  bash -lc 'cd /workspace/rl-garden && /opt/venv/openvla/bin/python -m pytest -q tests/test_sac_vmap_parity.py'
```

Result:

```text
5 passed
```

Interpretation: under fixed data and matched CPU/CUDA RNG streams, vmap and
legacy SAC updates are numerically equivalent. The remaining failure mode is
therefore likely rollout/replay trajectory sensitivity or a training-time
interaction not captured by fixed-batch parity, not a direct critic forward,
gradient, optimizer, or target-update mismatch.

## Training Curve Analysis (updated 2026-06-01)

All seed-1 runs are near completion (~760k–847k steps). Key findings:

### Runs start identically

Steps 8k–96k: legacy and vmap+RNG_fix produce bit-for-bit equivalent metrics.

| step | legacy α | vmap α | legacy Q | vmap Q |
|------|---------|-------|---------|-------|
| 8k | 0.7514 | 0.7514 | 15.28 | 15.27 |
| 32k | 0.1837 | 0.1837 | 4.627 | 4.632 |
| 96k | 0.00543 | 0.00544 | 0.678 | 0.671 |

This confirms the RNG fix is working: same initial weights, same first samples.

### Divergence begins at step 136k (~2100 gradient steps)

| metric | legacy | vmap+fix | delta |
|--------|--------|---------|-------|
| q/predicted | 0.829 | 0.916 | +11% |
| alpha | 0.000854 | 0.000928 | +9% |

vmap Q values trend 8–14% higher than legacy from this point and never
converge back.

### Observed failure pattern: low alpha and no breakthrough

| step | legacy α | vmap α | event |
|------|---------|-------|-------|
| 400k | 0.000697 | 0.000509 | vmap α begins sustained decline |
| 448k | 0.000658 | 0.000410 | **legacy first success (0.125)** |
| 480k | 0.000879 | 0.000407 | legacy α recovers upward |
| 840k | 0.001686 | 0.000378 | legacy 87.5% success; vmap still 0 |

Legacy alpha recovers around the first-success window. The likely interpretation
is that once the agent discovers a real grasping trajectory, the actor/critic
landscape changes and alpha starts recovering. In the failing vmap run, alpha
decays monotonically to about `0.0004` while success stays near zero. At that
point the actor loss is dominated by `-min_Q`, so the policy has little entropy
pressure left.

This is an observed failure pattern, not proof that low alpha is the sole cause.
Low alpha may be a cause, an effect of a bad replay/critic trajectory, or both.

### Current hypothesis: vmap changes long-horizon RGB training dynamics

`torch.func.vmap` + `functional_call` vectorizes the Q-head forward passes
into batched CUDA matmuls. Compared to legacy's sequential independent matmuls,
batched computation can use different CUDA kernels and floating-point reduction
orders. This can make vmap and legacy diverge over many rollout/update steps
even when fixed-batch one-step parity tests pass.

This numerical-path difference is a plausible contributor to the seed-1 RGB
ResNet divergence, but it is not proven to be the root cause. In particular,
the apparent Q differences in W&B may be downstream effects after replay and
rollout trajectories have already diverged, not necessarily the initial cause.
The observed failure mode is that vmap RGB ResNet runs remain around return
22-25 with near-zero success while legacy breaks through. In the failing vmap
runs, alpha stays low and the policy appears too deterministic to discover
stable grasping. The parity tests did not catch this because they use fixed
batches and short horizons; they validate local SAC math, not long-horizon
rollout/replay/encoder dynamics.

Treat this as a working hypothesis, not a closed diagnosis. The state-only
control below is important because it shows vmap SAC can train reliably when
ResNet and visual feature learning are removed.

## Decision Criteria

- Legacy wins seed 1 definitively (87.5% vs 0%).
- Both vmap variants (pre-fix and post-fix) fail identically — the RNG fix is
  necessary but not sufficient.
- RGB seed sweep confirms legacy is better at early breakthrough for seeds 1
  and 2; seed 3 is inconclusive at 1m.
- **Default PickCube RGB ResNet experiments to `--critic-impl legacy`** for
  recovery runs unless the experiment is explicitly testing vmap.
- Keep vmap as opt-in for RGB ResNet until fresh 2m/3m vmap runs recover
  consistently or a targeted mitigation is validated.
- If vmap recovery is desired, candidate mitigations (untested): entropy
  coefficient lower bound (`alpha_min=1e-4`), Q-value target clipping, or
  smaller `q_lr`.

## RGB ResNet Seed Sweep Snapshot (updated 2026-06-02)

Completed 1m paired runs show that `legacy` is clearly better on seeds 1 and 2,
while seed 3 is inconclusive because both implementations remain weak by 1m.

| run | final step | final return | final success | first success | max success | max return |
|---|---:|---:|---:|---:|---:|---:|
| seed1 legacy | 1,000,000 | 32.5802 | 0.8750 | 372,032 | 0.9375 | 34.6094 |
| seed1 vmap pre-fix | 972,032 | 23.3199 | 0.0000 | 548,032 | 0.0625 | 24.9298 |
| seed1 vmap rngfix | 924,032 | 22.7776 | 0.0000 | 436,032 | 0.0625 | 24.1512 |
| seed2 legacy | 1,000,000 | 31.8204 | 0.7500 | 272,000 | 0.7500 | 32.3508 |
| seed2 vmap | 1,000,000 | 21.8131 | 0.0000 | 748,032 | 0.0625 | 24.6455 |
| seed3 legacy | 1,000,000 | 21.9850 | 0.0625 | 492,032 | 0.0625 | 24.8044 |
| seed3 vmap | 1,000,000 | 24.0968 | 0.0000 | 384,000 | 0.0625 | 25.0126 |

Interpretation: the evidence supports `vmap` being worse at early breakthrough
on RGB ResNet PickCube, but 1m alone is not enough to separate implementation
failure from seed variance for all seeds. The 4.29 seed42 run reached first
success only at 816k and did not look good at 1m, but finished 3m with high
success. Therefore future long-run conclusions should use fresh continuous
2m/3m runs, not checkpoint continuation without replay buffer.

## Original Main vmap Run Reference

Reference W&B run:
`PickCube-v1__sac_rgbd_resnet10__1__1779939151__20260528_033231`
(`k1v7axnk`).

Key config:

```text
total_timesteps=2000000
critic_impl=vmap implicit main default
obs_mode=rgb
include_state=True
control_mode=pd_joint_delta_pos
encoder=resnet10
pretrained_weights=resnet10_pretrained_converted
image_fusion_mode=stack_channels
gamma=0.8
utd=0.25
num_envs=16
batch_size=512
eval_freq=10000
num_eval_envs=16
capture_video=True
```

Result: the run eventually learned, with final train `success_at_end=0.4375`
and `success_once=0.8125`. It did not show meaningful success by 1m; success
emerged much later, around 1.7m-1.8m.

Interpretation: vmap is not categorically unable to train RGB ResNet PickCube.
The stronger claim supported by current evidence is narrower: under the
no-video, very-low-eval-frequency 1m ablation settings, vmap has much worse
early breakthrough than legacy on seeds 1 and 2. Eval/video/RNG trajectory are
high-impact confounds because they can alter rollout/reset/sampling order, but
they are not proven root causes.

## State-only vmap/legacy Control

Purpose: remove ResNet and visual encoder training from the comparison. If
state-only `legacy` and `vmap` behave similarly, the RGB regression is likely a
visual SAC training-dynamics issue rather than a standalone critic API issue.
If state-only `vmap` is also consistently worse, keep vmap opt-in for SAC.

Implementation note: `examples/train_sac_state.py` now passes
`critic_impl=args.critic_impl` into `SAC`, so `--critic-impl legacy|vmap`
actually affects state-only runs.

Planned state-only runs:

```text
env_id=PickCube-v1
obs_mode=state
control_mode=pd_joint_delta_pos
gamma=0.8
utd=0.25
num_envs=16
num_eval_envs=1
batch_size=512
buffer_size=200000
buffer_device=cuda
learning_starts=4000
training_freq=64
policy_lr=0.0003
q_lr=0.0003
tau=0.01
total_timesteps=1000000
seeds=1,2,3
critic_impl=legacy,vmap
log_type=wandb
capture_video=False
```

Suggested run names:

```text
pickcube_sac_state_1m_seed1_legacy
pickcube_sac_state_1m_seed1_vmap
pickcube_sac_state_1m_seed2_legacy
pickcube_sac_state_1m_seed2_vmap
pickcube_sac_state_1m_seed3_legacy
pickcube_sac_state_1m_seed3_vmap
```

Started on 6017-nofwd at 2026-06-02 11:11 Asia/Shanghai:

| run | GPU | W&B |
|---|---:|---|
| `pickcube_sac_state_1m_seed1_legacy` | 0 | https://wandb.ai/dalian0744-intel/rl-garden/runs/akh4jqvk |
| `pickcube_sac_state_1m_seed1_vmap` | 1 | https://wandb.ai/dalian0744-intel/rl-garden/runs/rh7ughr2 |
| `pickcube_sac_state_1m_seed2_legacy` | 2 | https://wandb.ai/dalian0744-intel/rl-garden/runs/p2q85ofc |
| `pickcube_sac_state_1m_seed2_vmap` | 3 | https://wandb.ai/dalian0744-intel/rl-garden/runs/r9pv3ifp |
| `pickcube_sac_state_1m_seed3_legacy` | 0 | https://wandb.ai/dalian0744-intel/rl-garden/runs/qorqrbzm |
| `pickcube_sac_state_1m_seed3_vmap` | 1 | https://wandb.ai/dalian0744-intel/rl-garden/runs/yy038503 |

Initial check: all six runs reached W&B and entered the training loop. GPU0/1
run two state jobs each at about 8 GB; GPU2/3 run one state job each at about
4 GB. No OOM or CLI errors were observed.

State-only W&B analysis updated on 2026-06-02:

| seed | impl | first success | success >=0.75 | success ~=1 |
|---|---|---:|---:|---:|
| 1 | legacy | 176k | 244k | 260k |
| 1 | vmap | 216k | 300k | 0.9375 by 348k |
| 2 | legacy | 196k | 284k | 304k |
| 2 | vmap | 204k | 264k | 292k |
| 3 | legacy | 200k | 264k | 276k |
| 3 | vmap | 216k | 276k | 344k |

State-only interpretation:

- All six state-only runs reach high success early; `vmap` is not structurally
  broken for SAC on PickCube when the visual encoder is removed.
- The vmap/legacy difference in state mode is mostly first-breakthrough timing:
  seed 1 and seed 3 vmap are slower by about 40k-60k steps, while seed 2 vmap is
  slightly faster than legacy.
- `q/predicted` and `q/target` stay closely aligned in both implementations.
  There is no state-only evidence of persistent Q-target mismatch, Q explosion,
  or critic-target update failure.
- `entropy/alpha` follows the same healthy pattern in both implementations:
  it quickly falls from about `0.75` to about `1e-3`, then recovers toward
  `0.002-0.003` after success appears.
- Therefore the RGB ResNet regression is more likely an interaction between
  vmap, visual encoder training, replay trajectory, and early exploration than
  a standalone critic implementation bug.

Implication for next experiments: compare RGB vmap/legacy using visual-specific
diagnostics (`entropy/alpha`, policy entropy, feature norms, encoder gradient
norms, Q ensemble spread, and first-success timing). Do not use
`actor_entropy_coef` as an alpha-collapse fix; the branch below showed that it
can make alpha fall faster because alpha autotuning still reads the unscaled
policy `log_prob`.

## Actor Entropy Coefficient Branch

Purpose: test whether vmap RGB ResNet fails because the actor becomes too
deterministic before discovering stable grasping. This branch multiplies only
the actor-loss entropy term:

```python
actor_loss = (actor_entropy_coef * alpha * log_prob - min_q).mean()
```

It does not change critic target entropy backup or alpha autotuning.

Important interpretation: this is not a direct defense against alpha collapse.
If increasing `actor_entropy_coef` makes the actor choose higher-entropy
actions, the sampled `log_prob` becomes more negative. The alpha loss still uses
the raw `log_prob`:

```python
alpha_loss = -(alpha * (log_prob + target_entropy)).mean()
```

With PickCube's 7D action space, `target_entropy=-7`. More negative
`log_prob` makes `log_prob + target_entropy` more negative, so autotuning pushes
alpha down harder. Therefore an actor-only entropy coefficient can be
self-canceling or actively counterproductive unless alpha autotuning is also
changed.

Implementation note: `actor_entropy_coef` was added to `SACTrainingArgs` and
`SAC`; default is `1.0`, preserving current behavior.

Started on 6017-nofwd at 2026-06-02 11:41 Asia/Shanghai:

- exp: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_vmap_actorentropy5_seed1`
- W&B: https://wandb.ai/dalian0744-intel/rl-garden/runs/rc5h02n3
- tmux: `pickcube_sac_rgb_resnet10_actorentropy5_seed1_vmap`
- GPU: 2
- Key params: `critic_impl=vmap`, `actor_entropy_coef=5`, `seed=1`,
  `total_timesteps=1000000`, RGB ResNet10 converted stack-channel settings.

Initial check: run reached W&B and training loop, GPU2 memory was about 13.9 GB
while sharing the GPU with one state-only run, and no OOM or CLI error occurred.

Observed result before stopping:

| step | alpha |
|---:|---:|
| 22k | 0.323 |
| 72k | 0.0196 |
| 94k | 0.00576 |
| 127k | 0.000965 |
| 155k | 0.000254 |
| 172k | 0.000148 |

Interpretation: the branch moved in the wrong direction for preventing alpha
collapse. Alpha dropped faster than the ordinary vmap run at comparable steps.
This supports replacing this probe with direct alpha-autotune interventions
such as an `alpha_min` floor, target-entropy ablations, or explicit alpha-loss
changes.

Stopped on 6017-nofwd after the purpose was validated. The matching process was
`3502866` for exp
`pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_vmap_actorentropy5_seed1`.
Host-side termination did not work because the process was owned by root inside
the container; stopping from inside `liuzhaohong_maniskill_rlgarden` removed the
process. Follow-up `nvidia-smi` showed no remaining compute process for this
experiment.

## Alpha Min Probe

Purpose: test whether enforcing a hard lower bound on alpha prevents the
alpha-collapse failure mode in vmap RGB ResNet runs. Unlike `actor_entropy_coef`,
`alpha_min` acts directly on the output of `_current_alpha()` without altering the
alpha optimization or exploration target.

Implementation: `SAC._current_alpha()` returns
`self.log_alpha.exp().clamp(min=self.alpha_min)`. The `log_alpha` parameter
continues to be optimized via gradient descent; the clamp only affects the value
used in actor loss, entropy backup, and alpha loss. Default `alpha_min=0.0` is a
no-op, fully backward-compatible. Added to `SAC.__init__`, `SACTrainingArgs`, and
both `train_sac_rgbd.py` / `train_sac_state.py`. `OfflineSAC._current_alpha()` also
patched via `getattr(self, "alpha_min", 0.0)` for consistency.

CLI flag: `--alpha-min`.

Started on 6017-nofwd at 2026-06-02 Asia/Shanghai:

- exp: `pickcube_sac_rgb_resnet10_converted_gamma08_stack_1m_vmap_alphamin1e4_seed1`
- W&B: https://wandb.ai/dalian0744-intel/rl-garden/runs/xwr9ui1v
- Key params: `critic_impl=vmap`, `alpha_min=1e-4`, `seed=1`,
  `total_timesteps=1000000`, same RGB ResNet10 converted stack-channel settings
  as all other 1m ablations.
