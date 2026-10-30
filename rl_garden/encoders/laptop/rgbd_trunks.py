"""Three-view convolutional image features.
Three independent 5-channel (RGB, depth/declared metric scale, validity) trunks.
Uses 18-channel x4 branches and 54->32 pointwise late fusion.
View masks gate maps, not learned confidence. No privileged state inputs.
"""
import torch
from torch import nn

VIEW_ORDER=('head','left_wrist','right_wrist')

class RGBDTrunks(nn.Module):
    def __init__(self,depth_scale_m):
        super().__init__()
        if depth_scale_m<=0:raise ValueError('Positive fixed metric depth scale required')
        self.depth_scale_m=float(depth_scale_m)
        self.trunks=nn.ModuleList([nn.Sequential(nn.Conv2d(5,18,3,2),nn.ReLU(),nn.Conv2d(18,18,3),nn.ReLU(),nn.Conv2d(18,18,3),nn.ReLU(),nn.Conv2d(18,18,3),nn.ReLU()) for _ in VIEW_ORDER])
        self.fusion=nn.Conv2d(54,32,1)
    def forward(self,rgb,depth_m,depth_valid,view_valid):
        # B,V,C,H,W; RGB float in[0,1], metric depth, explicit boolean masks.
        if rgb.ndim!=5 or rgb.shape[1:3]!=(3,3) or min(rgb.shape[-2:])<17:raise ValueError('RGB BVCHW, three views, min17px')
        b,_,_,h,w=rgb.shape
        if depth_m.shape!=(b,3,1,h,w) or depth_valid.shape!=depth_m.shape or view_valid.shape!=(b,3):raise ValueError('input shape')
        if depth_valid.dtype!=torch.bool or view_valid.dtype!=torch.bool:raise ValueError('boolean masks required')
        if not torch.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any():raise ValueError('RGB scale')
        if not view_valid.any(-1).all():raise ValueError('All views missing; caller must handle safely')
        valid=depth_valid & view_valid[:,:,None,None,None]
        if (valid & (~torch.isfinite(depth_m)|(depth_m<=0))).any():raise ValueError('invalid metric depth marked valid')
        d=torch.where(valid,depth_m,torch.zeros_like(depth_m))/self.depth_scale_m
        maps=[]
        for i,trunk in enumerate(self.trunks):
            x=torch.cat((rgb[:,i],d[:,i],valid[:,i].to(rgb.dtype)),1)
            maps.append(trunk(x)*view_valid[:,i,None,None,None])
        return torch.relu(self.fusion(torch.cat(maps,1))),tuple(maps)
