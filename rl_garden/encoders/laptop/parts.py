"""Part-aware encoder variants. Privilege enters losses, never forward()."""
import torch
from torch import nn
from torch.nn import functional as F
from .full_encoder import FullEncoder, supervised_loss
from .full_temporal import CausalRGBD, KEYS


def camera_points(obs, height, width):
    """Calibrated feature-centre points, matching the frozen four-conv trunks."""
    ys = torch.arange(height, device=obs['rgb'].device)*2+7
    xs = torch.arange(width, device=obs['rgb'].device)*2+7
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).to(obs['rgb'])
    valid = obs['depth_valid'] & obs['view_valid'][:, :, None, None, None]
    depth = torch.where(valid, obs['depth_m'], 0.)[:, :, 0, ys[:, None], xs[None, :]]
    ray = torch.einsum('bvij,hwj->bvhwi', torch.linalg.inv(obs['K']), pixels)
    T = obs['T_BC']
    xyz = torch.einsum('bvij,bvhwj->bvhwi', T[:, :, :3, :3], ray*depth[..., None])+T[:, :, None, None, :3, 3]
    return xyz, valid[:, :, 0, ys[:, None], xs[None, :]], ys, xs


class PartEncoder(CausalRGBD):
    """Global/base/lid pooling with predicted membership and calibrated XYZ.

    The frozen base forward validates all observable inputs. Its token and score
    modules are reused; a single scoped hook replaces only the pooling-to-GRU
    input. This avoids a second copy of calibration/validity rules or trunks.
    Hooks are removed even when forward raises. No labels enter these hooks.
    """
    def __init__(self):
        super().__init__()
        self.part_logits = nn.Linear(64, 3)
        self.local_coordinates = nn.Linear(64, 6)
        self.part_fusion = nn.Sequential(nn.Linear(192, 64), nn.LayerNorm(64), nn.SiLU())
        self.last_aux = None

    def forward_step(self, obs, hidden=None, *, features_only=False):
        captured = {}
        def tokens_hook(module, args, output):
            captured['tokens'] = output
        def gru_input(module, args):
            tokens = captured['tokens']
            h = (obs['rgb'].shape[-2]-15)//2+1
            w = (obs['rgb'].shape[-1]-15)//2+1
            xyz, mask, _, _ = camera_points(obs, h, w)
            mask = mask.flatten(1)
            assert tokens.shape[:2] == mask.shape
            logits = self.part_logits(tokens)
            log_prob = logits.log_softmax(-1)
            score = self.score(tokens).squeeze(-1)
            pooled = [args[0][:, :64]]
            for part in (1, 2):
                weights = (score+log_prob[..., part]).masked_fill(~mask, float('-inf')).softmax(-1)
                pooled.append((tokens*weights[..., None]).sum(1))
            self.last_aux = dict(logits=logits, local=self.local_coordinates(tokens).reshape(*tokens.shape[:2], 2, 3), valid=mask)
            return torch.cat((self.part_fusion(torch.cat(pooled, -1)), args[0][:, 64:]), -1), args[1]
        a = self.tokens.register_forward_hook(tokens_hook)
        b = self.temporal.register_forward_pre_hook(gru_input)
        try:
            return super().forward_step(obs, hidden, features_only=features_only)
        finally:
            a.remove(); b.remove()


class PartSupervisedEncoder(FullEncoder):
    def __init__(self, variant):
        if variant not in ('task-distance', 'part-supervision', 'joint-supervision'):
            raise ValueError(variant)
        super().__init__()
        self.variant = variant
        if variant in ('part-supervision', 'joint-supervision'):
            common = self.perception.encoder.state_dict()
            # Additional heads must not advance the shared augmentation RNG.
            with torch.random.fork_rng(devices=[]):
                self.perception.encoder = PartEncoder()
            self.perception.encoder.load_state_dict(common, strict=False)
            self.perception.encoder.head = nn.Identity()

    def forward(self, obs, sensors, command, quality, hidden=None):
        result, hidden = super().forward(obs, sensors, command, quality, hidden)
        if self.variant in ('part-supervision', 'joint-supervision'):
            result['dense'] = self.perception.encoder.last_aux
        return result, hidden


def part_supervised_loss(pred, label, variant):
    total, terms = supervised_loss(pred, label)
    terms = dict(terms)
    if variant in ('part-supervision', 'joint-supervision'):
        d = pred['dense']; target = label['parts']
        mask = d['valid'] & target['valid']
        # Explicit log-softmax/gather avoids nondeterministic CUDA NLL kernels.
        nll = -d['logits'].log_softmax(-1).gather(-1, target['seg'][..., None]).squeeze(-1)
        counts = []; classes = []
        for k in range(3):
            use = mask & (target['seg'] == k)
            count = use.sum(1)
            classes.append((nll*use).sum(1)/count.clamp_min(1))
            counts.append(count > 0)
        present = torch.stack(counts, -1)
        semantic = (torch.stack(classes, -1)*present).sum(-1)/present.sum(-1).clamp_min(1)
        terms['part_semantic'] = .2*semantic.mean()
        foreground = mask & (target['seg'] > 0)
        index = (target['seg']-1).clamp_min(0)
        selected = d['local'].gather(2, index[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        error = F.smooth_l1_loss(selected/.1, target['local']/.1, reduction='none').mean(-1)
        terms['part_local'] = .2*((error*foreground).sum(1)/foreground.sum(1).clamp_min(1)).mean()
    if variant in ('task-distance', 'joint-supervision'):
        # All four TCP/operation pairs, with a smooth task-distance weighting.
        tcp = label['tcp_p']
        actual = (pred['assembled']['operation_p'][:, None]-tcp[:, :, None]).norm(dim=-1)
        truth = label['tcp_distance']
        weight = 1+torch.exp(-truth/.05)
        terms['tcp_distance'] = .5*(F.smooth_l1_loss(actual/.05, truth/.05, reduction='none')*weight).mean()
    extra = sum(v for k,v in terms.items() if k not in ('base_p','hinge_base','hinge_lid','points_lid','size_wh','base_R','closed_R','axis_base','opening','parts','operations','relative','uncertainty'))
    return total+extra, terms
