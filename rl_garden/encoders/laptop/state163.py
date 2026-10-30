"""163D measured proprioception + estimated articulated geometry, no object speeds.
Input provenance is enforced by the runtime adapter as well as this schema.
"""
import torch
from . import state106
from .state147 import rot6
from .articulated_geometry import assemble,tcp_velocity

GROUPS=(('proprio',86),('tcp_control_error',18),('visual_geometry',30),('relative_points',12),('quality',17))

def build(s,c,estimate,quality):
    # New sensor quantities are feedback/FK and geometric Jacobian, never object truth.
    old_s={'q','dq','width','width_rate','tcp_p','tcp_R'}
    if set(s)!=old_s|{'tcp_J'}:raise ValueError('sensor schema')
    if set(c)!={'reference_p','reference_R','width_target','q_target','q_target_valid'}:raise ValueError('command schema')
    keys={'base_p','base_R','hinge_base','axis_base','theta','closed_R','hinge_lid','points_lid','size_wh'}
    if set(estimate)!=keys:raise ValueError('estimated geometry only; no object velocities/truth/IDs')
    b=s['q'].shape[0];valid=c['q_target_valid']
    if valid.shape!=(b,2) or valid.dtype!=torch.bool or c['q_target'].shape!=(b,12):raise ValueError('joint command')
    if not torch.isfinite(c['q_target']).all():raise ValueError('joint target finite')
    if s['tcp_J'].shape!=(b,2,6,6):raise ValueError('TCP geometric Jacobian')
    geometry=assemble(**{k:estimate[k] for k in keys-{'size_wh'}})
    v={k:geometry[k] for k in ('part_p','part_R','operation_p')}
    v.update(size_wh=estimate['size_wh'],operation_v=s['q'].new_zeros(b,2,3),opening_rate=s['q'].new_zeros(b,1))
    old=state106.build({k:s[k] for k in old_s},v,{k:c[k] for k in ('reference_p','reference_R','width_target')},quality)
    mask=valid.repeat_interleave(6,-1);target=torch.where(mask,c['q_target'],s['q'])
    linear,angular=tcp_velocity(s['tcp_J'],s['dq'].reshape(b,2,6))
    flat=lambda x:x.reshape(b,-1)
    prop=torch.cat((old[:,:32],target,target-s['q'],flat(s['tcp_p']),flat(rot6(s['tcp_R'])),flat(linear),flat(angular)),-1)
    cv=quality[:,9:11].bool();cm=torch.cat((cv.repeat_interleave(3,-1),cv.repeat_interleave(6,-1)),-1)
    control=torch.where(cm,old[:,32:50],torch.zeros_like(old[:,32:50]))
    visual=torch.cat((old[:,50:72],geometry['opening_sincos'],geometry['hinge_p'],geometry['hinge_axis']),-1)
    groups=(prop,control,visual,old[:,72:84],torch.cat((quality,valid.float()),-1))
    for value,(_,n) in zip(groups,GROUPS):
        if value.shape!=(b,n) or not torch.isfinite(value).all():raise ValueError('dimension/finite')
    return torch.cat(groups,-1)
