# AGENTS.md

Brief for AI coding agents working on `rl-garden`. For broader context, read
[README.md](README.md) first.

---

## Execution Environment

`rl-garden` is a PyTorch-native SAC framework for ManiSkill. Training is GPU-first: normal rollouts, replay buffers, sampled batches, policy inference, and updates should stay on CUDA tensors.

CPU simulator paths are compatibility fallbacks only, mainly for short local smoke tests or CPU-backed ManiSkill envs such as `physx_cpu`. Do not make CPU the default training path.

For remote training/evaluation/debugging, read `.agents/rules/remote-training-sop.md`.
For Mutagen sync setup or repair, read `.agents/rules/mutagen-sync-sop.md`.
Agent-specific local runtime bindings may live in ignored files such as
`.agents/local/personal_config.md`; do not commit personal server, path, or
container details. Before any remote command, read that file. If it is missing,
stop and ask the user to create it from `.agents/local/personal_config.md.example`.

---

## Quick Orientation

`rl-garden` reorganizes ManiSkill SAC baselines into reusable algorithm/policy/encoder modules while keeping the hot path torch-native. It borrows SB3-style structure but does not import `stable_baselines3` in framework code.

Core design constraints:

- Keep ManiSkill GPU vectorization first-class.
- Avoid numpy handoff in rollout/replay/update hot paths.
- Store replay as torch tensors with `(T, N, ...)` layout.
- Keep RGBD actor/critic on a shared encoder; actor updates detach the encoder.
- Treat CPU env observations as fallback input that may be moved to the policy device for inference, not as the preferred training mode.

---

## Code Structure

- `rl_garden/algorithms/` - `BaseAlgorithm`, `OffPolicyAlgorithm`, `SAC`.
- `rl_garden/policies/` - SAC actor/critic policy modules.
- `rl_garden/buffers/` - GPU-native tensor and dict replay buffers.
- `rl_garden/encoders/` - flatten, plain CNN, combined RGBD/proprio, pooling, FiLM, augmentation, and ResNet encoders.
- `rl_garden/envs/` - ManiSkill env factory and vendored custom ManiSkill envs.
- `examples/` - Python training entrypoints.
- `scripts/` - shell launchers for common training runs.
- `tests/` - CPU unit tests and CUDA ManiSkill smoke tests.
- `3rd_party/` - reference submodules and external clones. Do not edit these unless the user explicitly asks.
- `pretrained_models/` - expected location for optional ResNet checkpoints.

Vendored custom envs live under `rl_garden/envs/custom/`.

---

## How Training Runs

State SAC:

```bash
scripts/train_sac_state.sh
```

Generic RGB SAC:

```bash
scripts/train_sac_rgbd.sh --encoder plain_conv
scripts/train_sac_rgbd.sh --encoder resnet10
```

Peg-only RGB SAC with GPU defaults:

```bash
scripts/train_sac_rgbd_peg.sh
```

The peg launcher uses:

- `PegInsertionSidePegOnly-v1`
- `sim_backend=gpu`
- `render_backend=gpu`
- `buffer_device=cuda`
- `panda_wristcam_gripper_closed_wo_norm`
- `pd_ee_delta_pose`
- RGB observations with state included

Short local peg smoke fallback, when CPU simulator mode is needed:

```bash
MPLCONFIGDIR=/tmp python \
  examples/train_sac_rgbd_peg.py \
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

Use `MPLCONFIGDIR=/tmp` when running ManiSkill commands if matplotlib cache warnings appear.

---

## Testing

Before finishing code changes:

- Run the smallest relevant test set first.
- For env/training changes, run a short smoke with tiny `total_timesteps`.
- Report exact commands and results.
- If a GPU/SAPIEN command fails because rendering or CUDA is unavailable, report the exact traceback and device status.

---

## Development Rules

- Keep `examples/train_sac_rgbd.py` generic for standard ManiSkill RGBD SAC. Put peg-specific defaults in `examples/train_sac_rgbd_peg.py`.
- Prefer extending `ManiSkillEnvConfig` and `make_maniskill_env()` for shared env features rather than duplicating wrapper logic in examples.
- Preserve CUDA-first behavior. Do not introduce CPU copies on the normal GPU path.
- Keep replay buffer storage/sample device behavior explicit: `buffer_device` controls storage, and samples move to the algorithm device.
- Use lazy imports for heavy ManiSkill/SAPIEN custom env dependencies so basic package import remains usable when optional simulator deps are absent.
- Use `rg` for search and inspect existing patterns before editing.
- Avoid modifying `3rd_party/` references except for explicit user requests.
- Do not add generated outputs (`runs/`, logs, caches, `__pycache__`) to git.

---

## Extending `rl-garden`

### New Algorithm

- Add the algorithm under `rl_garden/algorithms/`.
- Reuse `OffPolicyAlgorithm` when it fits the rollout/replay/update contract.
- Keep optimizer ownership clear: actor, critic, encoder, entropy coefficient.
- Add focused tests for construction, one update step, and edge cases.

### New Policy or Encoder

- Implement feature extractors under `rl_garden/encoders/` by subclassing `BaseFeaturesExtractor`.
- Use `policy_kwargs` with `features_extractor_class` and `features_extractor_kwargs` for injection.
- Add shape/device tests, and test CPU fallback only as compatibility behavior.

### New Environment

- Add reusable env factory support in `rl_garden/envs/maniskill.py` when the setting is broadly useful.
- Add custom ManiSkill env registrations under `rl_garden/envs/custom/`.
- Register custom envs lazily through `rl_garden.envs.register_custom_envs()`.
- Keep env-specific training defaults in a dedicated example/script.

---

## Current Custom Env Notes

`PegInsertionSidePegOnly-v1` is vendored from Residual_RL into `rl_garden/envs/custom/`. It depends on custom Panda wrist-camera/fixed-gripper agents and custom controller configs, also vendored under the same package.

The peg env registers two cameras (`base_camera` + `hand_camera`). To avoid ManiSkill's default channel-stacked single `rgb` key, `examples/train_sac_rgbd_peg.py` enables `per_camera_rgbd=True` and `image_fusion_mode="per_key"` so each camera gets its own `rgb_<camera>` key and its own 3-channel encoder. This is what makes ResNet + ImageNet pretrained weights load without channel-shape mismatch.

The generic RGBD trainer should remain independent of this env. Use:

```bash
examples/train_sac_rgbd_peg.py
scripts/train_sac_rgbd_peg.sh
```

for peg-only experiments.

---

## Git and Artifacts

- Check `git status --short` before and after changes.
