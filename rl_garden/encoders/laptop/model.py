"""Shared RGBD architecture for compact and high-resolution inputs."""
import torch
from torch.nn import functional as F

from .parts import PartSupervisedEncoder, part_supervised_loss


class LaptopEncoder(PartSupervisedEncoder):
    def __init__(self):
        super().__init__('joint-supervision')
        self.experiment_variant = 'rgbd'


def supervised_loss(pred, label):
    """Geometry losses, including observed-depth/part consistency; labels are offline only."""
    total, terms = part_supervised_loss(pred, label, 'joint-supervision')
    target = label['parts']
    index = (target['seg'] - 1).clamp_min(0)
    valid = target['valid'] & pred['dense']['valid'] & (target['seg'] > 0)
    local = pred['dense']['local']
    batch = torch.arange(local.shape[0], device=local.device)[:, None]
    point = torch.arange(local.shape[1], device=local.device)[None]
    selected = local[batch, point, index]
    rotation = pred['assembled']['part_R'][batch, index]
    position = pred['assembled']['part_p'][batch, index]
    xyz = torch.einsum('bnij,bnj->bni', rotation, selected) + position
    error = F.smooth_l1_loss(xyz / .1, label['observed_xyz'] / .1,
                           reduction='none').mean(-1)
    terms['depth_part_consistency'] = .2 * (
        (error * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()
    return total + terms['depth_part_consistency'], terms

