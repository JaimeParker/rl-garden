# Pretrained model weights

Drop PyTorch checkpoints in this directory and reference them by name from
the CLI:

```bash
python examples/train_sac_rgbd.py --encoder resnet10 --pretrained_weights resnet10-imagenet
# loads ./pretrained_models/resnet10-imagenet.pt
```

Override the directory with `$RL_GARDEN_PRETRAINED_DIR`.

Expected file format: either a raw `state_dict` or a dict with a `state_dict`
key. Keys should match `ResNetEncoder.state_dict()`; pooling / bottleneck
heads are initialized fresh, so missing-key warnings for those heads are
expected (loads use `strict=False`).

Residual SAC ACT base policies use the same directory. For example:

```bash
scripts/train_residual_sac_rgbd_peg.sh --policy act --ckpt-path act-peg-only
# loads ./pretrained_models/act-peg-only.pt
```

ACT checkpoints may be `{"ema_agent": state_dict, "norm_stats": ...}`,
`{"agent": state_dict, "norm_stats": ...}`, or a raw ACT state dict.

## open_laptop visual encoder

The supervised laptop encoder is separate from the ResNet loader described above.
Its model and training code are included, but pretrained weights are not distributed.
See the [encoder guide](../docs/ENCODER_INTERFACE.md) for the architecture and
supervised training interface, and the [residual route guide](../docs/OPEN_LAPTOP_RECIPES.md)
for its use as a frozen state estimator.
