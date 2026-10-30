"""Complete articulated perception + measured proprio -> 163 deployable features."""
import torch
from torch import nn
from torch.nn import functional as F
from .full_perception import Perception163
from .articulated_geometry import assemble
from . import state163

class FullEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.perception=Perception163()
        # Avoid arbitrary random degenerate rotations/axes at the first update.
        # Small asymmetry avoids zero rotation gradient at an exact 180-degree
        # mismatch. Canonical visual x is opposite the positive hinge direction.
        nn.init.normal_(self.perception.head.weight,std=.001);nn.init.zeros_(self.perception.head.bias)
        with torch.no_grad():
            self.perception.head.bias[3:9]=torch.tensor([1.,0,0,0,1,0])
            self.perception.head.bias[12:15]=torch.tensor([-1.,0,0])
            self.perception.head.bias[15:17]=torch.tensor([0.,1.])
            self.perception.head.bias[17:23]=torch.tensor([1.,0,0,0,1,0])
    def forward(self,obs,sensors,command,quality,hidden=None):
        e,h,aux,sigma=self.perception(obs,hidden)
        g=assemble(**{k:v for k,v in e.items() if k!='size_wh'})
        q=quality.clone();q[:,3:6]=aux['depth_fraction'];q[:,12]=aux['temporal_valid'];q[:,13:15]=sigma
        state=state163.build(sensors,command,e,q)
        return dict(geometry=e,assembled=g,state=state,sigma=sigma),h

def supervised_loss(pred,label):
    e=pred['geometry'];truth=label['geometry'];g=pred['assembled'];full=label['state_truth']
    terms={}
    for key in ('base_p','hinge_base','hinge_lid','points_lid','size_wh'):
        terms[key]=F.smooth_l1_loss(e[key]/.1,truth[key]/.1)
    for key in ('base_R','closed_R','axis_base'):terms[key]=F.mse_loss(e[key],truth[key])
    sc=lambda angle:torch.stack((angle.sin(),angle.cos()),-1)
    terms['opening']=F.mse_loss(sc(e['theta']),sc(truth['theta']))
    # Geometry and mixed visual/proprio relations are trained jointly, with
    # targets independently checked against original saved object labels.
    truth_g=assemble(**{k:v for k,v in truth.items() if k!='size_wh'})
    terms['parts']=F.smooth_l1_loss(g['part_p']/.1,truth_g['part_p']/.1)+F.mse_loss(g['part_R'],truth_g['part_R'])
    terms['operations']=F.smooth_l1_loss(g['operation_p']/.1,truth_g['operation_p']/.1)
    terms['relative']=F.smooth_l1_loss(pred['state'][:,134:146]/.1,label['relative_truth']/.1)
    assert torch.allclose(pred['state'][:,:104],full[:,:104],atol=2e-5),'Visual estimates changed measured proprio/control'
    position=(g['operation_p']-truth_g['operation_p']).square().sum(-1).mean(-1)
    angle=torch.atan2((e['theta']-truth['theta']).sin(),(e['theta']-truth['theta']).cos()).square()
    sigma=pred['sigma']
    terms['uncertainty']=.01*((position.detach()/sigma[:,0].square()+2*sigma[:,0].log())+(angle.detach()/sigma[:,1].square()+2*sigma[:,1].log())).mean()
    return sum(terms.values()),terms
