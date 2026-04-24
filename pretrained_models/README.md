# Pretrained encoder weights

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
