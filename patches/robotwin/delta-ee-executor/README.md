# RoboTwin delta-EE executor runtime patch

This bundle contains the RoboTwin-side runtime changes required by the current
`open_laptop` ACT + Residual SAC configuration.

The rl-garden files in this submission pass delta-EE execution settings into
RoboTwin, but the actual joint-settle loop and planner recovery live inside the
external RoboTwin runtime. Therefore this directory is included with the
submission package instead of silently assuming the reviewer already has the
same local RoboTwin checkout.

## Files

```text
robotwin-delta-ee-executor.patch
files/envs/_base_task.py
files/envs/robot/planner.py
apply_robotwin_delta_ee_executor_patch.ps1
MANIFEST.json
```

- `robotwin-delta-ee-executor.patch` is the standard Git patch from the
  recorded RoboTwin base file contents to the target runtime contents.
- `files/...` contains exact target copies of the two modified RoboTwin files.
  These are redundant with the patch, but useful for direct hash review.
- `apply_robotwin_delta_ee_executor_patch.ps1` is a dry-run-by-default helper.
  It checks exact file hashes first and writes only when `-Apply` is passed.

## RoboTwin provenance

Recorded base commit:

```text
0008ae6800df9f75fc8de7098bacb01735fd8fd2
```

Recorded target runtime commit:

```text
964a4e4b1c434d62a5d106a8fbc543210641a8d9
```

The local target copies included here match the recorded target SHA-256 values
for the two modified files.

## What this patch fixes

- Adds delta-EE command reference state so consecutive small deltas are applied
  relative to the last command target rather than only the lagging physical
  pose.
- Reanchors the command reference to the real end-effector pose when the
  command/reference discrepancy grows beyond the runtime threshold.
- Converts near-zero delta-EE commands and successful-but-empty plans into
  finite hold trajectories, avoiding zero-length trajectory synchronization.
- Adds terminal settle waiting after the final trajectory target is sent:
  RoboTwin keeps writing the terminal joint target with zero velocity and steps
  the physics scene until the real joint error is within tolerance, or the tick
  cap is reached.
- Improves `mplib_screw` planning robustness by trying bounded screw steps,
  preserving full articulation qpos for MPlib, and rejecting terminal plans
  whose FK endpoint misses the requested target beyond tolerance.

## Usage

Dry-run first:

```powershell
powershell -ExecutionPolicy Bypass -File .\patches\robotwin\delta-ee-executor\apply_robotwin_delta_ee_executor_patch.ps1 -RoboTwinPath C:\path\to\RoboTwin
```

Apply only after the dry-run reports that both files match the expected base
state:

```powershell
powershell -ExecutionPolicy Bypass -File .\patches\robotwin\delta-ee-executor\apply_robotwin_delta_ee_executor_patch.ps1 -RoboTwinPath C:\path\to\RoboTwin -Apply
```

If the target RoboTwin checkout already contains these changes, the helper
reports that it is already patched and does not rewrite the files.
