# Pretrained model weights

Externally-sourced weights (not checkpoints produced by rl-garden's own
training runs — those go under `--checkpoint_dir`). Override this directory
with `$RL_GARDEN_PRETRAINED_DIR`.

## ResNet backbones

```bash
python examples/train_online.py sac --encoder resnet10 --pretrained_weights resnet10-imagenet
# loads ./pretrained/resnet/resnet10-imagenet.pt
```

Expected file format: either a raw `state_dict` or a dict with a `state_dict`
key. Keys should match `ResNetEncoder.state_dict()`; pooling / bottleneck
heads are initialized fresh, so missing-key warnings for those heads are
expected (loads use `strict=False`). Torchvision-style checkpoints must be
converted first with `tools/conversion/convert_resnet_checkpoint.py`.

## ACT base policies

```bash
# loads ./pretrained/act/act-peg-only.pt
```

ACT checkpoints may be `{"ema_agent": state_dict, "norm_stats": ...}`,
`{"agent": state_dict, "norm_stats": ...}`, or a raw ACT state dict.
