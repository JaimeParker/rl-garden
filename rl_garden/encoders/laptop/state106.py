"""Geometric feature assembly shared by the 163-dimensional visual state.

All poses common fixed robot base; part frames are SEMANTIC visual frames,
not arbitrary simulator link origins. Labels and predicted inputs stay separate.
The caller provides calibrated measured gripper opening and its target in metres.
No simulator APIs, hidden IDs, contact truth or privileged observation accepted.
"""
import torch
from .state147 import rot6

GROUPS=(('proprio',32),('control_error',18),('visual_geometry',22),('relative_points',12),('visual_motion',7),('quality',15))

def _rotation(r):
    eye=torch.eye(3,dtype=r.dtype,device=r.device).expand_as(r)
    if not torch.isfinite(r).all() or not torch.allclose(r.transpose(-1,-2)@r,eye,atol=1e-4) or not (torch.linalg.det(r)>0).all():
        raise ValueError('Invalid rotation')

def build(s,v,c,quality):
    """s: q,dq[B,12]; width,width_rate[B,2]; tcp_p[B,2,3],tcp_R[B,2,3,3].
    v: part_p[B,2,3],part_R[B,2,3,3],size_wh[B,2,2],operation_p[B,2,3],
       operation_v[B,2,3],opening_rate[B,1]: causal visual estimates, not labels.
       Visible BASE/LID exterior width/height, not glass or collision bounds.
    c: reference_p[B,2,3],reference_R[B,2,3,3],width_target[B,2].
    quality[B,15]: view_valid3,depth_valid_fraction3,image_age_s3,
        command_valid2,proprio_valid1,temporal_valid1,position_std_m1,angle_std_rad1.
    Missing observations must be safety-handled upstream, never filled from truth.
    Geometry is estimated at capture time and aligned to feedback time.
    """
    b=s['q'].shape[0]
    shapes_s={'q':(b,12),'dq':(b,12),'width':(b,2),'width_rate':(b,2),'tcp_p':(b,2,3),'tcp_R':(b,2,3,3)}
    shapes_v={'part_p':(b,2,3),'part_R':(b,2,3,3),'size_wh':(b,2,2),'operation_p':(b,2,3),'operation_v':(b,2,3),'opening_rate':(b,1)}
    shapes_c={'reference_p':(b,2,3),'reference_R':(b,2,3,3),'width_target':(b,2)}
    for values,shapes in ((s,shapes_s),(v,shapes_v),(c,shapes_c)):
        if set(values)!=set(shapes):raise ValueError('Only declared deployable inputs accepted')
        for key,shape in shapes.items():
            if values[key].shape!=shape or not torch.isfinite(values[key]).all():raise ValueError(key)
    if quality.shape!=(b,15) or not torch.isfinite(quality).all():raise ValueError('quality')
    flags=quality[:,[0,1,2,9,10,11,12]]
    if not ((flags==0)|(flags==1)).all():raise ValueError('binary validity flags')
    if (quality[:,3:6]<0).any() or (quality[:,3:6]>1).any() or (quality[:,6:9]<0).any() or (quality[:,13:]<0).any():raise ValueError('quality range')
    if (v['size_wh']<=0).any() or (s['width']<0).any() or (c['width_target']<0).any():raise ValueError('physical lengths')
    for r in (s['tcp_R'],v['part_R'],c['reference_R']):_rotation(r)
    flat=lambda x:x.reshape(b,-1)
    prop=torch.cat((s['q'],s['dq'],s['width'],s['width_rate'],c['width_target'],c['width_target']-s['width']),-1)
    rr=s['tcp_R'].transpose(-1,-2)@c['reference_R']
    identity=torch.eye(3,device=rr.device,dtype=rr.dtype).expand_as(rr)
    control=torch.cat((flat(c['reference_p']-s['tcp_p']),flat(rot6(rr)-rot6(identity))),-1)
    visual=torch.cat((flat(v['part_p']),flat(rot6(v['part_R'])),flat(v['size_wh'])),-1)
    delta=v['operation_p'][:,None]-s['tcp_p'][:,:,None]
    relative=(v['part_R'][:,0,None,None].transpose(-1,-2)@delta[...,None]).squeeze(-1)
    values=(prop,control,visual,flat(relative),torch.cat((flat(v['operation_v']),v['opening_rate']),-1),quality)
    for x,(_,n) in zip(values,GROUPS):
        if x.shape!=(b,n):raise ValueError('group dimension')
    return torch.cat(values,-1)

def calibrated_aperture(joint_position,*,slope,offset):
    """Only for independently verified AFFINE gripper mechanisms, not universal.
    Coefficients must come from hardware/URDF pad geometry audit, never defaults.
    Nonlinear linkages require a different calibrated forward-kinematics model.
    """
    if slope is None or offset is None:raise ValueError('Calibration required')
    width=joint_position*slope+offset
    if not torch.isfinite(width).all() or (width<0).any():raise ValueError('Invalid calibrated aperture')
    return width

def parallel_finger_feedback(q_pair,dq_pair,raw_target_pair,lower,upper,*,verified_offset_m):
    """Audited opposite prismatic fingers only, last axis is the TWO fingers.

    Actual aperture=offset+q7+q8, not 2*q7 or drive_target. The requested
    attainable aperture bounds raw actuator targets by mechanical joint limits;
    this explicitly separates an overdrive/preload target from physical width.
    Caller retains raw targets for diagnostics. No force/contact inference.
    verified_offset_m must come from this embodiment's opposing-face audit.
    """
    if verified_offset_m is None or q_pair.shape[-1]!=2:raise ValueError('Verified two-finger geometry required')
    if any(x.shape!=q_pair.shape or not torch.isfinite(x).all() for x in (dq_pair,raw_target_pair,lower,upper)) or not torch.isfinite(q_pair).all():raise ValueError('feedback shape/finite')
    if (lower>upper).any() or (q_pair<lower-1e-6).any() or (q_pair>upper+1e-6).any():raise ValueError('feedback outside audited limits')
    width=q_pair.sum(-1)+verified_offset_m
    target=torch.minimum(torch.maximum(raw_target_pair,lower),upper).sum(-1)+verified_offset_m
    if (width<0).any() or (target<0).any():raise ValueError('Negative physical aperture')
    return {'width':width,'width_rate':dq_pair.sum(-1),'width_target':target,
            'raw_target_outside_limits':((raw_target_pair<lower)|(raw_target_pair>upper)).any(-1)}
