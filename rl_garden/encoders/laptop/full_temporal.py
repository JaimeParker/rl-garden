"""Causal calibrated RGBD backbone used by the laptop encoders.

Three independent convolutional trunks, metric calibrated point tokens, learned
pooling and GRU. No asset IDs, privileged labels, simulator handles or future
frames enter forward_step. RGB-D samples with invalid depth are excluded in
this backbone; all-depth-missing explicitly fails instead of inventing state.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .rgbd_trunks import RGBDTrunks

KEYS={'rgb','depth_m','depth_valid','view_valid','K','T_BC','dt_s','reset'}

class CausalRGBD(nn.Module):
    def __init__(self,depth_scale_m=2.):
        super().__init__();self.depth_scale_m=depth_scale_m
        self.trunks=RGBDTrunks(depth_scale_m).trunks
        self.tokens=nn.Sequential(nn.Linear(24,64),nn.LayerNorm(64),nn.SiLU())
        self.score=nn.Linear(64,1);self.temporal=nn.GRUCell(71,64)
        self.head=nn.Linear(64,35)
        with torch.no_grad():
            self.head.bias[6:18]=torch.tensor([1.,0,0,0,1,0]*2)
    def forward_step(self,obs,hidden=None,*,features_only=False):
        if set(obs)!=KEYS:raise ValueError('Only declared observable inputs accepted')
        rgb=obs['rgb'];depth=obs['depth_m'];valid=obs['depth_valid'];views=obs['view_valid']
        if rgb.ndim!=5 or rgb.shape[1:3]!=(3,3):raise ValueError('RGB BVCHW')
        b,_,_,h,w=rgb.shape
        if min(h,w)<17 or depth.shape!=(b,3,1,h,w) or valid.shape!=depth.shape or views.shape!=(b,3):raise ValueError('shape')
        if valid.dtype!=torch.bool or views.dtype!=torch.bool or obs['reset'].dtype!=torch.bool:raise ValueError('boolean masks')
        if obs['K'].shape!=(b,3,3,3) or obs['T_BC'].shape!=(b,3,4,4) or obs['dt_s'].shape!=(b,1) or obs['reset'].shape!=(b,):raise ValueError('calibration/time shape')
        if not torch.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any():raise ValueError('RGB range')
        if not torch.isfinite(obs['K']).all() or not torch.isfinite(obs['T_BC']).all():raise ValueError('calibration finite')
        if (obs['K'][...,0,0]<=0).any() or (obs['K'][...,1,1]<=0).any():raise ValueError('focal length')
        T=obs['T_BC'];R=T[...,:3,:3];eye=torch.eye(3,device=R.device,dtype=R.dtype).expand_as(R)
        if not torch.allclose(R.transpose(-1,-2)@R,eye,atol=1e-4) or not torch.allclose(torch.linalg.det(R),torch.ones_like(R[...,0,0]),atol=1e-4):raise ValueError('Rigid calibration rotation required')
        if not torch.allclose(T[...,3,:],T.new_tensor([0.,0,0,1]).expand_as(T[...,3,:]),atol=1e-6):raise ValueError('Homogeneous calibration')
        if not torch.allclose(obs['K'][...,2,:],T.new_tensor([0.,0,1]).expand_as(obs['K'][...,2,:])):raise ValueError('Pinhole calibration')
        if not torch.isfinite(obs['dt_s']).all() or (obs['dt_s']<0).any() or (obs['dt_s']>120).any():raise ValueError('time gap outside validated range')
        if ((~obs['reset'])&(obs['dt_s'][:,0]<=0)).any():raise ValueError('Nonreset time must advance')
        effective=valid & views[:,:,None,None,None]
        if (effective & (~torch.isfinite(depth)|(depth<=0))).any():raise ValueError('bad valid depth')
        safe_depth=torch.where(effective,depth,torch.zeros_like(depth))
        feats=[];coords=[];masks=[]
        for i,trunk in enumerate(self.trunks):
            x=trunk(torch.cat((rgb[:,i],safe_depth[:,i]/self.depth_scale_m,effective[:,i].to(rgb.dtype)),1))
            # Four 3x3 convolutions, strides 2/1/1/1: centre=7+2*index.
            hh,ww=x.shape[-2:];ys=torch.arange(hh,device=x.device)*2+7;xs=torch.arange(ww,device=x.device)*2+7
            yy,xx=torch.meshgrid(ys,xs,indexing='ij')
            pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).to(rgb.dtype)
            d=safe_depth[:,i,0][:,ys[:,None],xs[None,:]]
            m=effective[:,i,0][:,ys[:,None],xs[None,:]]
            ray=torch.einsum('bij,hwj->bhwi',torch.linalg.inv(obs['K'][:,i]),pixels)
            xyz=ray*d[...,None];T=obs['T_BC'][:,i]
            xyz=torch.einsum('bij,bhwj->bhwi',T[:,:3,:3],xyz)+T[:,None,None,:3,3]
            onehot=F.one_hot(torch.tensor(i,device=x.device),3).to(rgb.dtype).expand(b,hh,ww,3)
            feats.append(torch.cat((x.permute(0,2,3,1),xyz,onehot),-1).reshape(b,-1,24))
            coords.append(xyz);masks.append(m.reshape(b,-1))
        mask=torch.cat(masks,1)
        if not mask.any(-1).all():raise ValueError('No valid calibrated RGBD tokens')
        tokens=self.tokens(torch.cat(feats,1));logits=self.score(tokens).squeeze(-1).masked_fill(~mask,float('-inf'))
        weight=logits.softmax(-1);pooled=(tokens*weight[...,None]).sum(1)
        if hidden is None:
            if not obs['reset'].all():raise ValueError('History absent without explicit reset')
            hidden=pooled.new_zeros(b,64)
        if hidden.shape!=(b,64) or not torch.isfinite(hidden).all():raise ValueError('hidden')
        hidden=torch.where(obs['reset'][:,None],torch.zeros_like(hidden),hidden)
        fractions=effective.to(rgb.dtype).mean((2,3,4))
        hidden=self.temporal(torch.cat((pooled,obs['dt_s'].log1p(),views.to(rgb.dtype),fractions),-1),hidden)
        if features_only:
            return hidden,{'temporal_valid':~obs['reset'],'depth_fraction':fractions,'point_coordinates':coords}
        z=self.head(hidden);raw=z[:,6:18].reshape(b,2,6)
        c1=F.normalize(raw[...,:3],dim=-1);c2=raw[...,3:]-(c1*raw[...,3:]).sum(-1,keepdim=True)*c1
        if (c2.norm(dim=-1)<1e-6).any():raise ValueError('Degenerate predicted rotation')
        c2=F.normalize(c2,dim=-1);R=torch.stack((c1,c2,torch.cross(c1,c2,dim=-1)),-1)
        v={'part_p':z[:,:6].reshape(b,2,3),'part_R':R,'size_wh':F.softplus(z[:,18:22]).reshape(b,2,2)*.2+.001,
           'operation_p':z[:,22:28].reshape(b,2,3),'operation_v':z[:,28:34].reshape(b,2,3),'opening_rate':z[:,34:35]}
        # On reset velocity has no temporal evidence; caller receives explicit flag.
        aux={'temporal_valid':~obs['reset'],'depth_fraction':fractions,'point_coordinates':coords}
        return v,hidden,aux
