"""Shared rotation representation and skew matrix (column vectors)."""
import torch

def skew(v):
    x,y,z=v.unbind(-1); o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)


def rot6(r):
    return r[...,:,:2].transpose(-1,-2).flatten(-2)
