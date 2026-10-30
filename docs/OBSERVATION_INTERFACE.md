# open_laptop observation and action interfaces

This document defines the privileged-state and visual residual routes. Arrays use float32 unless stated otherwise, with batch first and left arm before right arm. Index ranges are half-open: 0-14 means entries 0 through 13.

## Inputs by module

| Module | Input | Notes |
| --- | --- | --- |
| ACT base policy | Three 320×240 RGB views and the original 14-dimensional control state | Same control state as ACT training |
| Privileged-state actor and critic | state[14] + state_open_laptop_robot_v5[132] | No appended image features |
| Visual actor and critic | state_encoder163[163] | No object ground truth, segmentation, asset IDs or original control state |
| Visual-route warmup policy | 146 privileged dimensions plus the ACT base action | Privileged observations do not enter visual policy inputs or replay |
| Perception network | RGBD, calibration, timing and validity; measured robot feedback for state assembly | No future frames or privileged object labels |
| Supervised loss | Predictions and separately stored labels | Labels are not passed to model forward |

Simulation rewards, success checks and the warmup policy may use ground truth. This does not expose it to the visual residual actor or critic; the latter also uses estimated state.

The 146 and 163 dimensions are observation features, not the full MLP input. The actor additionally receives the normalized 14-dimensional ACT base action: 160 dimensions for the privileged route and 177 for the visual route. The critic receives observation features plus the 14-dimensional final action. Base actions are distinct from ACT's control state; replay stores current and next base actions.

## ACT control state: 14 dimensions

Each arm contributes six joint reference angles in radians and one dimensionless gripper command, following the original ACT convention. Measured q and dq are read separately by joint name for visual-state assembly.

## Privileged-state route: 146 dimensions

Indices refer to the concatenated feature input. The original observation retains separate keys for the first 14 and remaining 132 dimensions.

| Indices | Contents | Units and source |
| --- | --- | --- |
| 0-14 | ACT-compatible control state | Six angles in rad and one dimensionless gripper command per arm |
| 14-46 | Measured joint positions and velocities | position8, velocity8 per arm; six rotary joints in rad and rad/s, two finger joints in m and m/s |
| 46-76 | TCP pose, reference pose and reference validity | 7+7+1 per arm; world xyz in m and unit wxyz quaternion |
| 76-90 | Hinge parent and child poses | World xyz in m and wxyz; privileged |
| 90-94 | Hinge position, velocity and limits | rad, rad/s, rad, rad; privileged |
| 94-100 | Hinge origin and axis | World m and unit vector; privileged |
| 100-114 | Two interaction-point poses | World xyz in m and wxyz; privileged |
| 114-120 | Two part collision dimensions | xyz sizes in m; privileged |
| 120-126 | Two part geometry centres | Part-local m; privileged |
| 126-138 | Vectors from two TCPs to two interaction points | Hinge-parent frame, m; left/right TCP, point0/1, xyz |
| 138-142 | Base/lid contact during the preceding action | Left/right × base/lid; contact at any physics substep |
| 142-146 | Base/lid contact at the final physics substep | Same order, binary |

Names and normalization divisors are defined by `rl_garden.encoders.scaled_task_state.scaling_spec()`; source fields are in `SOURCE_FIELDS['open_laptop_robot_v5']`. Measured joints are matched by name.

Normalization occurs only in the feature extractor: rotary joints and hinge angles use π; finger positions use 0.04765m and velocities 0.1m/s; dimensions, geometry centres and relative contact vectors use 0.25m; remaining fields use 1. There are no running statistics or clipping. Replay retains physical units.

## Visual route: 163 dimensions

The reference frame is the robot base. Predicted part frames follow supervised-label definitions and may differ from simulation link origins. `rot6(R)` concatenates the first two matrix columns.

| Indices | Contents | Units and source |
| --- | --- | --- |
| 0-12 | Measured arm q | rad; joint-name mapping |
| 12-24 | Measured arm dq | rad/s |
| 24-26 | Gripper widths | m; measured finger joints and calibration |
| 26-28 | Width rates | m/s |
| 28-30 | Target widths | m; limited to reachable gripper range |
| 30-32 | Target minus measured width | m |
| 32-44 | Arm q_target | rad; drive targets, separate from feedback |
| 44-56 | q_target−q | rad |
| 56-62 | Measured TCP positions | m; FK of measured q |
| 62-74 | Measured TCP orientations | Two rot6 vectors, including terminal-joint motion |
| 74-80 | TCP linear velocities | m/s; Jacobian×dq |
| 80-86 | TCP angular velocities | rad/s; Jacobian×dq |
| 86-92 | Reference minus measured TCP position | m; reference from drive-target FK |
| 92-104 | rot6(R_actualᵀR_reference)−rot6(I) | Dimensionless; invalid references masked |
| 104-110 | Base and lid positions | m; predicted |
| 110-122 | Base and lid orientations | Two rot6 vectors; predicted |
| 122-126 | Part widths and heights | m; outer dimensions including borders |
| 126-128 | Opening-angle sine and cosine | Predicted |
| 128-131 | Hinge origin | m; predicted |
| 131-134 | Hinge axis | Predicted unit vector |
| 134-146 | TCP-to-interaction-point vectors | Base-part frame, m; two arms × two points × xyz |
| 146-149 | View validity | Head, left, right; binary |
| 149-152 | Valid-depth fractions | 0–1 |
| 152-155 | Time elapsed since each image | Seconds; zero for the current exposure |
| 155-157 | TCP reference validity | Left, right; binary |
| 157-158 | Robot-feedback validity | Binary |
| 158-159 | Temporal-history validity | Binary; zero at the first window frame |
| 159-160 | Predicted position uncertainty | m |
| 160-161 | Predicted angular uncertainty | rad |
| 161-163 | Joint-target validity | Left, right; binary |

The first 104 dimensions use measured feedback and control references; remaining fields combine visual geometry and validity indicators. Measured q/dq remain separate from drive targets. Missing fields, nonfinite values, absent valid-depth features or invalid rotations raise errors.

Normalization divisors: π for 0-24, 32-56, 80-86 and 160-161; 0.1 for 24-32 and 159-160; 0.25 for 86-92, 122-126 and 134-146; 1 elsewhere. The source state is not modified in place.

## Actions and executor

Final actions have shape [batch,14], with [dx,dy,dz,rx,ry,rz,gripper] per arm. Entries 6 and 13 are grippers. Rotation increments follow RoboTwin delta_ee conventions, not rot6 or absolute orientations. ACT physical actions are normalized by the environment action scaler; residuals are added and clipped in that space before conversion back to physical units.

Both routes use position/rotation/gripper normalization units of 0.06m / 0.3rad / 0.2 and residual scales of 0.1 / 0.2 / 0.2. Without final clipping, maximum residual components are approximately 6mm / 0.06rad / 0.04. The warmup policy's rotation output is converted to preserve its original physical action rather than directly reusing the student's rotation scale.

The executor uses mplib_screw with command references. It reanchors at position error 5mm or rotation error 2 degrees. Terminal settling uses a joint threshold of 0.0005rad and at most 100 physics steps. Training and evaluation share these simulation execution settings.
