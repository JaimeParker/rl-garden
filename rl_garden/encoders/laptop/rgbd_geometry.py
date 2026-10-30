"""Isolated torch geometry contract. No simulator, actor, or truth-label access.

Column-vector convention: T_AB maps frame B into A. Camera uses OpenCV
x right/y down/z forward, metric AXIAL depth (not ray range). Rectified images
only. Calibration must correspond to the same capture and pixel grid.
"""
import torch


def homogeneous_extrinsic(extrinsic):
    """SAPIEN intrinsic world-to-CV extrinsic is 3x4; no fitted correction."""
    if extrinsic.shape==(3,4):
        extrinsic=torch.cat((extrinsic,extrinsic.new_tensor([[0.,0.,0.,1.]])),dim=0)
    check_transform(extrinsic)
    return extrinsic


def check_transform(t):
    if t.shape != (4,4) or not torch.isfinite(t).all():
        raise ValueError('Expected finite 4x4 transform')
    r=t[:3,:3]; eye=torch.eye(3,dtype=t.dtype,device=t.device)
    if not torch.allclose(r.T@r,eye,atol=1e-5) or not torch.allclose(torch.linalg.det(r),t.new_tensor(1.),atol=1e-5):
        raise ValueError('Invalid rotation')
    if not torch.allclose(t[3],t.new_tensor([0,0,0,1])):
        raise ValueError('Invalid homogeneous row')


def transform_points(t_ab, points_b):
    check_transform(t_ab)
    return points_b@t_ab[:3,:3].T+t_ab[:3,3]


def world_from_cv(extrinsic_cv, cam2world_gl=None):
    """extrinsic_cv = T_CW. Optional GL matrix is independent consistency check."""
    check_transform(extrinsic_cv)
    # Validated rigid transform: analytical inverse preserves exact homogeneous
    # row and avoids generic float32 matrix-inverse roundoff (~1.5e-8 here).
    result=torch.eye(4,dtype=extrinsic_cv.dtype,device=extrinsic_cv.device)
    result[:3,:3]=extrinsic_cv[:3,:3].T
    result[:3,3]=-result[:3,:3]@extrinsic_cv[:3,3]
    if cam2world_gl is not None:
        check_transform(cam2world_gl)
        flip=torch.diag(result.new_tensor([1.,-1.,-1.,1.]))
        if not torch.allclose(result,cam2world_gl@flip,atol=1e-4):
            raise ValueError('CV/GL calibration conventions disagree')
    return result


def depth_metres(raw, *, unit, valid, near, far):
    if unit not in ('m','mm') or not 0 < near < far:
        raise ValueError('Explicit depth units and ordered positive clipping bounds required')
    if raw.ndim!=2 or valid.shape!=raw.shape or valid.dtype!=torch.bool:
        raise ValueError('Depth HW and explicit boolean validity mask required')
    d=raw.to(torch.float32)*(0.001 if unit=='mm' else 1.)
    mask=valid & torch.isfinite(d) & (d>near) & (d<far)
    return torch.where(mask,d,torch.zeros_like(d)),mask


def check_k(k):
    if k.shape!=(3,3) or not torch.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:
        raise ValueError('Invalid intrinsic matrix')
    if not torch.allclose(k[2],k.new_tensor([0.,0.,1.])):
        raise ValueError('Expected pinhole intrinsic matrix')


def deproject(depth, k, valid):
    check_k(k)
    if depth.ndim!=2 or valid.shape!=depth.shape or valid.dtype!=torch.bool:
        raise ValueError('Expected depth HW with boolean mask')
    if (valid & (~torch.isfinite(depth)|(depth<=0))).any():
        raise ValueError('Invalid depth marked valid')
    h,w=depth.shape
    y,x=torch.meshgrid(torch.arange(h,device=depth.device,dtype=k.dtype),torch.arange(w,device=depth.device,dtype=k.dtype),indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),-1)
    xyz=(pixels@torch.linalg.inv(k).T)*depth[...,None]
    return torch.where(valid[...,None],xyz,torch.zeros_like(xyz))


def project(points_cv,k):
    check_k(k)
    good=torch.isfinite(points_cv).all(-1)&(points_cv[...,2]>0)
    p=points_cv@k.T
    safe=torch.where(good,p[...,2],torch.ones_like(p[...,2]))
    uv=p[...,:2]/safe[...,None]
    return torch.where(good[...,None],uv,torch.zeros_like(uv)),good


def crop_resize_k(k, *, x0,y0,sx,sy):
    """Crop then resize with half-pixel-center convention (align_corners=False).
    Caller MUST use this image-resize convention; no image resize performed here.
    """
    check_k(k)
    if sx<=0 or sy<=0:raise ValueError('Positive scales required')
    affine=k.new_tensor([[sx,0,sx*(.5-x0)-.5],[0,sy,sy*(.5-y0)-.5],[0,0,1]])
    return affine@k


def prepare_view(rgb,depth_raw,k,extrinsic_cv,*,depth_unit,depth_valid,near,far,cam2world_gl=None):
    """Same-grid rectified RGB HWC uint8 and depth HW only; no unproven alignment.
    Distinct real RGB/depth sensors require upstream calibrated registration.
    Returns separate RGB, metric depth, mask, K and T_WC; never generic /255 depth.
    """
    if rgb.dtype!=torch.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.shape[:2]!=depth_raw.shape:
        raise ValueError('Expected same-grid HWC uint8 RGB and HW depth')
    check_k(k)
    d,valid=depth_metres(depth_raw,unit=depth_unit,valid=depth_valid,near=near,far=far)
    twc=world_from_cv(extrinsic_cv,cam2world_gl)
    return dict(rgb=rgb.permute(2,0,1).to(torch.float32)/255.,depth_m=d[None],depth_valid=valid[None],K=k,T_WC=twc)
