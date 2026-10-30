# Visual encoder and supervised learning

The encoder estimates laptop geometry from three views and camera calibration, then combines it with measured robot feedback into a 163-dimensional state. It is pretrained with supervision and frozen during residual training. Both 128×128 and 320×240 inputs have produced good geometric estimates.

This repository provides models, supervised losses and training interfaces, but not pretrained weights. See the [observation interface](OBSERVATION_INTERFACE.md).

This guide describes the currently used architecture and configuration, which has produced good open_laptop simulation performance. It is a starting point for improving architectures, supervision and preprocessing, not a requirement to use the same network for every task.

## 1. Architecture

The base encoder uses three independent CNN branches and calibrated 3D point features to extract global, base and lid features. A GRU processes the current frame and three preceding frames. Outputs include part poses, dimensions, hinge geometry, opening angle, and position and angle uncertainty. Joint and gripper feedback are measured directly.

Each RGBD branch receives RGB, depth in metres and a depth-validity channel: five channels in total. Each CNN has four 3×3 convolutions with 18 channels and strides 2/1/1/1. Point features and the GRU hidden state have 64 dimensions. The network learns spatial contributions to part features; segmentation labels are used only for supervision.

| Model | Inputs and training | Role |
| --- | --- | --- |
| Base visual encoder | Three RGBD views, four-frame history, geometry and depth-consistency losses | Pretrained architecture used by the visual residual route |
| Higher-resolution EMA encoder | 320×240 RGBD; same network and losses, with averaged weights | Best geometric estimates in existing RGBD comparisons |
| RGB-only encoder | 320×240 RGB, calibrated rays, four-frame history and averaged weights | Best existing RGB-only configuration |

The repository implements the first two models; RGB-only adaptation is described below. EMA copies online weights for the first 5000 updates, then averages parameters and buffers with decay 0.999. AdamW updates the online model.

## 2. Inputs and outputs

Call `prediction, hidden = model(obs, sensors, command, quality, hidden)`. B denotes batch size; view order is head, left wrist, right wrist.

| obs field | Shape and type | Meaning |
| --- | --- | --- |
| rgb | [B,3,3,H,W], float32 | RGB in 0–1 |
| depth_m | [B,3,1,H,W], float32 | Optical-axis depth in metres |
| depth_valid | Same shape as depth, bool | Valid-depth mask |
| view_valid | [B,3], bool | View validity |
| K | [B,3,3,3], float32 | Intrinsics for the current image size |
| T_BC | [B,3,4,4], float32 | CV camera-to-robot-base transform |
| dt_s | [B,1], float32 | Time since the preceding frame, seconds |
| reset | [B], bool | First-frame flag |

`sensors` contains measured `q,dq [B,12]`, `width,width_rate [B,2]`, `tcp_p [B,2,3]`, `tcp_R [B,2,3,3]` and `tcp_J [B,2,6,6]`. TCP poses and Jacobians use actual joints and tool calibration.

`command` contains `reference_p [B,2,3]`, `reference_R [B,2,3,3]`, `width_target [B,2]`, `q_target [B,12]` and `q_target_valid [B,2]`. Reference TCP poses use FK of joint drive targets.

`quality [B,15]` contains view validity (3), valid-depth fractions (3), elapsed time since each image (3), reference-pose validity (2), robot-feedback validity (1), history validity (1) and uncertainty (2).

Outputs include `geometry`, expanded geometric relations in `assembled`, `state [B,163]`, `sigma [B,2]` and auxiliary `dense` predictions. `hidden [B,64]` is passed only within the same four-frame window.

## 3. Calibration and preprocessing

Intrinsics define pixel rays; extrinsics locate cameras relative to the robot base. For homogeneous pixel coordinates p, `X_C = depth × inv(K) × p`; `T_BC` transforms this point to the base frame. Wrist-camera extrinsics update with arm motion.

Depth is quantized to 1mm. RGB uses bilinear resizing with `align_corners=False`; depth and masks use nearest-exact resizing. Intrinsics are adjusted with pixel-centre offsets.

Capture runs at approximately 5Hz during actions and includes decision-boundary images. Each prediction uses the latest four frames, resetting the GRU at the first frame. Object ground truth, segmentation and asset information are supervision labels only, never perception inputs.

## 4. Dataset and supervised training

The training set contains 1064 trajectories and 169,912 synchronized timestamps: 509,736 single-view RGBD images across three cameras. Two privileged-state collection policies contribute 500 trajectories each; ACT contributes 64. Data include successes, failures, approaches and occlusions.

The validation set is scene-disjoint from training: 100 fixed scenes, each with 16 four-frame windows, totaling 1600 windows. It is repeatedly used for model selection; no separate independent encoder test score is reported. The policy test set uses another 500 fixed scenes and measures complete ACT-plus-residual task success.

Half the training windows are sampled by choosing trajectories uniformly, then timestamps uniformly. The other half emphasizes object motion and surrounding context. Batch size is 8; AdamW learning rate is 3e-4, weight decay 1e-4, and gradient norm limit 10. Losses cover part poses, dimensions, opening angle, interaction points, TCP relations, uncertainty, part classification, local coordinates, task distances and observed-depth consistency.

Code: [model and losses](../rl_garden/encoders/laptop/model.py), [supervised updates](../rl_garden/encoders/laptop/supervised.py), [weight averaging](../rl_garden/encoders/laptop/ema.py). The caller prepares four synchronized frames with batched `obs`, `sensors`, `command`, `quality` and `label` tensors on one device.

```python
from rl_garden.encoders.laptop.model import LaptopEncoder
from rl_garden.encoders.laptop.supervised import make_optimizer, train_window
from rl_garden.encoders.laptop.ema import make_ema

model = LaptopEncoder().to('cuda')
optimizer = make_optimizer(model)
ema = make_ema(model)  # Optional for the base encoder.
loss = train_window(model, optimizer, frames, ema=ema, completed_updates=1)
```

Validation measures interaction-point, part-position, rotation, opening-angle and relative-position errors, including dynamic, static and occluded subsets. The selection score is `S = interaction-point error(m) + 0.5×position error(m) + 0.5×relative-position error(m) + 0.001×[rotation error(degrees)+opening-angle error(degrees)]`; lower is better. Report geometry errors separately from policy success.

## 5. Weight selection and RGB-only adaptation

`load_encoder(..., variant='rgbd')` reads online weights; `rgbd-ema` reads averaged weights. The residual CLI uses `--encoder-variant` (default `rgbd`) and `--encoder-resolution`. Both must match pretraining. New checkpoints should record their variant.

RGB-only models require a matching network and new pretraining, not zero-filled depth:

1. Use three RGB channels; remove depth and depth-validity inputs.
2. Replace depth-backprojected points with calibrated ray directions and camera positions.
3. Derive feature validity from view and pixel masks, not depth masks.
4. Remove observed-depth losses while retaining pose, angle, point and relation supervision.
5. Keep four-frame temporal processing, then freeze the pretrained model for residual learning.

An RGB pixel determines a ray, not distance. Multi-view and temporal learning must infer distance; scale and occlusion are generally harder than with RGBD.

## 6. Connecting another encoder

For the supplied architecture, provide trained weights directly. Ordinary weights may be stored under `model`, under `state_dict`, or as a raw state dict. Existing online and averaged checkpoint formats remain supported.

For another architecture, pass `encoder_factory=MyEncoder` to the Python builder. It constructs your `torch.nn.Module` and loads weights strictly. The residual algorithm does not need modification. The CLI uses the supplied architecture; custom architectures use the Python entrypoint.

Follow the call signature in section 2. The current residual capture path uses `prediction['state'] [B,163]` and the returned hidden state. Field order, physical meaning and units must match the observation interface, not merely the tensor dimension. Keep the model frozen and exclude privileged object information.

For a different call signature or output format, wrap the model in a `torch.nn.Module`: convert inputs in `forward`, invoke the original model, and return the required fields. Explicitly convert coordinate frames and units; do not substitute zeros for missing physical quantities. Supervised training additionally requires the geometry and auxiliary outputs described above; inference-only adapters need not provide them.
