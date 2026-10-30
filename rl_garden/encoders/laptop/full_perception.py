"""Articulated geometry decoder; object labels never enter forward."""
import torch
from torch import nn
from torch.nn import functional as F
from .full_temporal import CausalRGBD
from .articulated_geometry import assemble
from . import state163

from .rotation import rotation6

class Perception163(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=CausalRGBD();self.encoder.head=nn.Identity();self.head=nn.Linear(64,36);self.uncertainty=nn.Linear(64,2)
        with torch.no_grad():
            self.head.bias[3:9]=torch.tensor([1.,0,0,0,1,0]);self.head.bias[12:15]=torch.tensor([1.,0,0]);self.head.bias[15:17]=torch.tensor([0.,1.]);self.head.bias[17:23]=torch.tensor([1.,0,0,0,1,0])
    def forward(self,obs,hidden=None):
        h,aux=self.encoder.forward_step(obs,hidden,features_only=True);z=self.head(h)
        if (z[:,12:15].norm(dim=-1)<1e-6).any() or (z[:,15:17].norm(dim=-1)<1e-6).any():raise ValueError('degenerate hinge')
        e=dict(base_p=z[:,:3],base_R=rotation6(z[:,3:9]),hinge_base=z[:,9:12],axis_base=F.normalize(z[:,12:15],dim=-1),theta=torch.atan2(z[:,15],z[:,16]),closed_R=rotation6(z[:,17:23]),hinge_lid=z[:,23:26],points_lid=z[:,26:32].reshape(-1,2,3),size_wh=F.softplus(z[:,32:36]).reshape(-1,2,2)*.2+.001)
        sigma=F.softplus(self.uncertainty(h))+z.new_tensor([.001,.005])
        return e,h,aux,sigma
