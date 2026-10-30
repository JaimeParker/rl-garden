"""Treat official RGBA-scaled partial-opacity depth as invalid, never metric."""
import torch

def normalize(frame,rgba):
    for name,v in frame['views'].items():
        alpha=torch.as_tensor(rgba[name]['rgba'][...,3]).detach().cpu().clone()
        assert alpha.dtype==torch.uint8 and alpha.shape==v['depth_mm'].shape
        pos=v['position_gl'];z=-pos[...,2]
        valid=torch.isfinite(pos).all(-1)&(pos[...,3]>=0)&(pos[...,3]<1)&(z>0)
        expected=z*(alpha.float()/255.)
        if valid.any():
            assert ((v['depth_mm']*.001-expected)[valid]).abs().max()<1e-5,'Official depth/actual opacity disagree'
        else:
            assert torch.isfinite(pos).all() and torch.isfinite(v['depth_mm']).all(), 'Nonfinite empty-depth view'
            v['depth_mm']=torch.zeros_like(v['depth_mm'])
        v['alpha_uint8']=alpha
        # Installed rt/camera.rmiss sets ray.origin=world(0,0,0) on a miss;
        # camera.rgen transforms it into camera coordinates before writing
        # Position. Thus w<1 alone does NOT reject every background pixel.
        # Detect the documented renderer sentinel, not semantic segmentation.
        twc=v['cam2world_gl'];world=pos[...,:3]@twc[:3,:3].T+twc[:3,3]
        hit=world.norm(dim=-1)>1e-5;v['render_hit_valid']=hit
        # This is a validity mask, not a fitted calibration or truth depth.
        v['depth_mm']=torch.where((alpha==255)&hit,v['depth_mm'],torch.zeros_like(v['depth_mm']))
    return frame
