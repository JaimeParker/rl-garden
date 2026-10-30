"""Supervised updates from prepared, synchronized four-frame batches."""
import torch

from .model import supervised_loss
from .ema import update_ema


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


def train_window(model, optimizer, frames, *, ema=None, completed_updates):
    """Train on the last frame's labels, retaining four-frame causal gradients.

    Each frame contains batched obs/sensors/command/quality tensors. The last
    frame also contains label. Sampling, preprocessing and augmentation belong
    to the caller's dataset. completed_updates includes this update.
    """
    if len(frames) != 4 or completed_updates < 1:
        raise ValueError('Expected four frames and a positive update count')
    if not frames[0]['obs']['reset'].all():
        raise ValueError('First frame must reset the causal history')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    hidden = None
    for frame in frames:
        prediction, hidden = model(frame['obs'], frame['sensors'], frame['command'],
                                   frame['quality'], hidden)
    value, _ = supervised_loss(prediction, frames[-1]['label'])
    if not torch.isfinite(value):
        raise ValueError('Nonfinite supervised loss')
    value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    if ema is not None:
        update_ema(ema, model, completed_updates)
    return float(value.detach())
