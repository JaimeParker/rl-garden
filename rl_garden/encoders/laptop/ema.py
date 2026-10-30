"""EMA: copy through update 5000; then decay 0.999, including buffers."""
from torch.optim.swa_utils import AveragedModel


def average(previous, current, count):
    return .999 * previous + (1 - .999) * current


def make_ema(model):
    return AveragedModel(model, avg_fn=average, use_buffers=True).requires_grad_(False)


def update_ema(ema, model, completed_updates):
    if completed_updates <= 5000:
        ema.n_averaged.zero_()
    ema.update_parameters(model)
