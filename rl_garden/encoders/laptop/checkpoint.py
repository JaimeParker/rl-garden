"""Explicit inference-weight selection; never treats EMA as an Adam resume state."""
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(path, *, variant, expected_sha256, device='cpu', model_factory=None):
    import torch
    from .model import LaptopEncoder

    if variant not in ('rgbd', 'rgbd-ema'):
        raise ValueError('Supported encoders: rgbd, rgbd-ema')
    if not expected_sha256 or sha256(path) != expected_sha256:
        raise ValueError('Encoder SHA256 mismatch')
    # Only load trusted local artifacts. torch serialization is not a safe file format
    # for untrusted uploads, even when the caller supplied their checksum.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    identity = saved.get('variant', saved.get('config', {}).get('variant'))
    if identity in ('rgbd', 'rgbd-ema') and identity != variant:
        raise ValueError(f'Expected {variant}, found {identity}')
    if variant == 'rgbd':
        if 'model' in saved:
            state = saved['model']
        elif 'state_dict' in saved:
            state = saved['state_dict']
        else:
            state = saved  # Plain torch state_dict.
    else:
        ema = saved['ema']
        if 'n_averaged' not in ema or any(
                k != 'n_averaged' and not k.startswith('module.') for k in ema):
            raise ValueError('Expected AveragedModel EMA state with module.* keys')
        state = {k[7:]: v for k, v in ema.items() if k.startswith('module.')}
    model = (LaptopEncoder if model_factory is None else model_factory)().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)

