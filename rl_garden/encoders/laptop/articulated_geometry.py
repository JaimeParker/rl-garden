"""Deployable geometry algebra. Inputs are estimates/calibration, never asset IDs.

All world quantities use one fixed robot base; hinge_* are local metric points.
Velocities are optional diagnostics, NOT policy features. No static-base assumption.
"""
import torch
from .state106 import _rotation
from .state147 import skew

def unit_axis(axis):
    if not torch.isfinite(axis).all() or not torch.allclose(axis.norm(dim=-1),torch.ones_like(axis[...,0]),atol=1e-4):
        raise ValueError('Declared directed unit hinge axis required')

def axis_rotation(axis,theta):
    unit_axis(axis)
    if not torch.isfinite(theta).all():raise ValueError('angle')
    x,y,z=axis.unbind(-1);o=torch.zeros_like(x)
    k=torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*axis.shape[:-1],3,3)
    eye=torch.eye(3,device=axis.device,dtype=axis.dtype)
    return eye+theta[...,None,None].sin()*k+(1-theta[...,None,None].cos())*(k@k)

def assemble(base_p,base_R,hinge_base,axis_base,theta,closed_R,hinge_lid,points_lid):
    """closed_R maps lid axes to base-part axes at declared zero opening.
    points_lid: two estimated/calibrated points fixed to the lid. No hidden lookup.
    Supports translating/rotating laptop base. Outputs positions, not contact.
    """
    for x in (base_p,hinge_base,hinge_lid,points_lid):
        if not torch.isfinite(x).all():raise ValueError('nonfinite geometry')
    _rotation(base_R);_rotation(closed_R)
    lid_R=base_R@axis_rotation(axis_base,theta)@closed_R
    hinge_p=base_p+(base_R@hinge_base[...,None]).squeeze(-1)
    lid_p=hinge_p-(lid_R@hinge_lid[...,None]).squeeze(-1)
    points=lid_p[...,None,:]+(lid_R[...,None,:,:]@points_lid[...,None]).squeeze(-1)
    return dict(part_p=torch.stack((base_p,lid_p),-2),part_R=torch.stack((base_R,lid_R),-3),
                operation_p=points,hinge_p=hinge_p,hinge_axis=(base_R@axis_base[...,None]).squeeze(-1),
                opening_sincos=torch.stack((theta.sin(),theta.cos()),-1))

def opening_angle(base_R,lid_R,closed_R,axis_base,tolerance=1e-3):
    """Signed angle; directed axis and closed-frame calibration are indispensable.
    Reject off-axis relative rotations rather than calling arbitrary pose angle opening.
    """
    for r in (base_R,lid_R,closed_R):_rotation(r)
    unit_axis(axis_base);d=base_R.transpose(-1,-2)@lid_R@closed_R.transpose(-1,-2)
    vee=torch.stack((d[...,2,1]-d[...,1,2],d[...,0,2]-d[...,2,0],d[...,1,0]-d[...,0,1]),-1)/2
    sine=(vee*axis_base).sum(-1);cosine=(d.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    theta=torch.atan2(sine,cosine)
    if ((axis_rotation(axis_base,theta)-d).abs().amax(dim=(-1,-2))>tolerance).any():
        raise ValueError('Relative pose violates calibrated hinge model')
    return theta

def diagnostic_point_velocity(geometry,base_p,base_v,base_omega,opening_rate):
    """Instantaneous rigid-body relationship; not a visual speed predictor.
    base_v is velocity of base_p, NOT COM velocity unless origins coincide.
    Never equate this with backward interval-average labels.
    """
    hp=geometry['hinge_p'];axis=geometry['hinge_axis'];points=geometry['operation_p']
    hv=base_v+torch.linalg.cross(base_omega,hp-base_p,dim=-1)
    omega=base_omega+axis*opening_rate[...,None]
    return hv[...,None,:]+torch.linalg.cross(omega[...,None,:].expand_as(points),points-hp[...,None,:],dim=-1)

def tcp_velocity(jacobian,dq):
    """Calibrated geometric TCP Jacobian [linear; angular], fixed common base."""
    if jacobian.shape[-2:]!=(6,6) or dq.shape!=jacobian.shape[:-2]+(6,):raise ValueError('Jacobian shape')
    if not torch.isfinite(jacobian).all() or not torch.isfinite(dq).all():raise ValueError('feedback finite')
    velocity=(jacobian@dq[...,None]).squeeze(-1)
    return velocity[...,:3],velocity[...,3:]
