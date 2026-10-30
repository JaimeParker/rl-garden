"""The validated 6D rotation reconstruction used by the visual encoders.

Preserves the 6D rotation convention and rejects degenerate/nonfinite inputs.
Uses cross products to re-orthogonalize instead of relaxing geometry checks.
"""
import torch
from torch.nn import functional as F


def rotation6(raw):
    if raw.shape[-1] != 6 or not torch.isfinite(raw).all():
        raise ValueError('finite 6D rotation required')
    a = F.normalize(raw[..., :3], dim=-1)
    b = raw[..., 3:]-(a*raw[..., 3:]).sum(-1, keepdim=True)*a
    if (a.norm(dim=-1) < .9).any() or (b.norm(dim=-1) < 1e-6).any():
        raise ValueError('degenerate rotation')
    c = F.normalize(torch.linalg.cross(a, b, dim=-1), dim=-1)
    b = F.normalize(torch.linalg.cross(c, a, dim=-1), dim=-1)
    return torch.stack((a, b, c), -1)
