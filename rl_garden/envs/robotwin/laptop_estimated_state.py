"""Observable-only RGBD/FK adapter. No laptop handle, labels, IDs or segmentation."""
from collections import deque
import weakref
import numpy as np
import torch
from torch.nn import functional as F
from rl_garden.encoders.laptop.robot_calibration import RobotCalibration
from rl_garden.encoders.laptop.rgbd_geometry import prepare_view
from rl_garden.encoders.laptop.depth_alpha import normalize


def resize_rgbd(rgb, depth, valid, intrinsics, image_size):
    """Resize synchronized views and adjust the pixel-centre camera intrinsics."""
    image_size = tuple(image_size)
    if image_size not in ((128, 128), (240, 320)):
        raise ValueError('Encoder image size must be (128, 128) or (240, 320)')
    h, w = rgb.shape[-2:]
    oh, ow = image_size
    transform = intrinsics.new_tensor([[ow/w, 0, (ow/w)*.5-.5],
                                        [0, oh/h, (oh/h)*.5-.5], [0, 0, 1.]])
    quant = (depth*1000).round().to(torch.int32).float()*.001
    return dict(rgb=F.interpolate(rgb.float()/255, size=image_size, mode='bilinear', align_corners=False),
                depth_m=F.interpolate(quant, size=image_size, mode='nearest-exact'),
                depth_valid=F.interpolate(valid.float(), size=image_size, mode='nearest-exact').bool(),
                K=transform@intrinsics)


class LiveState:
    def __init__(self, task, model, calibration, image_size=(128, 128)):
        self.image_size = tuple(image_size)
        if self.image_size not in ((128, 128), (240, 320)):
            raise ValueError('Encoder image size must be (128, 128) or (240, 320)')
        self.task = weakref.ref(task)
        self.model = model
        self.robot = calibration
        self.rows = deque(maxlen=4)
        self.tick = 0
        self.last_tick = -1
        self.last_dense = -1e9
        self.busy = False
        self.dt = float(task.scene.get_timestep())
        self.original_step, self.original_render = task.scene.step, task._update_render
        def step(*args, **kwargs):
            result = self.original_step(*args, **kwargs)
            self.tick += 1
            return result
        def render(*args, **kwargs):
            result = self.original_render(*args, **kwargs)
            if not self.busy and self.tick*self.dt-self.last_dense >= .2-1e-8:
                self.capture()
                self.last_dense = self.tick*self.dt
            return result
        task.scene.step, task._update_render = step, render

    def capture(self):
        if self.last_tick == self.tick:
            return
        self.busy = True
        try:
            task = self.task()
            task.cameras.update_picture()
            names = ('head_camera', 'left_camera', 'right_camera')
            cameras = dict(zip(task.cameras.static_camera_name, task.cameras.static_camera_list))
            cameras.update(left_camera=task.cameras.left_camera, right_camera=task.cameras.right_camera)
            configs, rgbs, depths = task.cameras.get_config(), task.cameras.get_rgb(), task.cameras.get_depth()
            tensor = lambda x:torch.as_tensor(np.asarray(x).copy()).cpu()
            from rl_garden.encoders.laptop.rgbd_geometry import homogeneous_extrinsic
            frame = dict(views={})
            for name in names:
                k = tensor(configs[name]['intrinsic_cv']).float()
                k[0,2] -= .5;k[1,2] -= .5
                frame['views'][name] = dict(rgb=tensor(rgbs[name]['rgb']),
                    depth_mm=tensor(depths[name]['depth']).float(),
                    position_gl=tensor(cameras[name].get_picture('Position')).float(), K=k,
                    extrinsic_cv=homogeneous_extrinsic(tensor(configs[name]['extrinsic_cv']).float()),
                    cam2world_gl=tensor(configs[name]['cam2world_gl']).float())
            frame = normalize(frame, task.cameras.get_rgba())
            BW = torch.linalg.inv(tensor(task.robot.left_entity.get_root_pose().to_transformation_matrix()).float())
            views = []
            for name in names:
                v = frame['views'][name];pos = v['position_gl']
                valid = torch.isfinite(pos).all(-1)&(pos[...,3]>=0)&(pos[...,3]<1)&(-pos[...,2]>0)&(v['alpha_uint8']==255)&v['render_hit_valid']
                views.append(prepare_view(v['rgb'], v['depth_mm'], v['K'], v['extrinsic_cv'],
                    depth_unit='mm', depth_valid=valid, near=.01, far=10., cam2world_gl=v['cam2world_gl']))
            rgb = torch.stack([frame['views'][name]['rgb'].permute(2,0,1) for name in names])
            depth = torch.stack([view['depth_m'] for view in views])
            obs = resize_rgbd(rgb, depth, torch.stack([v['depth_valid'] for v in views]),
                              torch.stack([v['K'] for v in views]), self.image_size)
            obs.update(view_valid=torch.ones(3,dtype=torch.bool),
                       T_BC=torch.stack([BW@v['T_WC'] for v in views]))
            measured = {}
            for side in ('left','right'):
                robot = getattr(task.robot, side+'_entity');joints=robot.get_active_joints()
                measured[side] = dict(qpos=robot.get_qpos().tolist(), qvel=robot.get_qvel().tolist(),
                    names=[j.get_name() for j in joints], targets=[float(j.get_drive_target()[0]) for j in joints])
            sensors, command = self.robot.feedback(measured)
            quality = torch.zeros(1,15);quality[:,:3]=1;quality[:,9:12]=1
            stamp = self.tick*self.dt
            gap = 0. if not self.rows else stamp-self.rows[-1]['time']
            assert not self.rows or gap > 0
            self.rows.append(dict(obs=obs, sensors=sensors, command=command, quality=quality,
                                  time=stamp, dt_s=gap, reset=not self.rows))
            self.last_tick = self.tick
        finally:
            self.busy = False

    @torch.inference_mode()
    def predict(self):
        self.capture()
        device = next(self.model.parameters()).device
        hidden = None
        for i,row in enumerate(self.rows):
            obs = {k:v[None].to(device) for k,v in row['obs'].items()}
            obs.update(dt_s=torch.tensor([[0. if i==0 else row['dt_s']]], device=device),
                       reset=torch.tensor([i==0], dtype=torch.bool, device=device))
            pred,hidden = self.model(obs, {k:v.to(device) for k,v in row['sensors'].items()},
                {k:v.to(device) for k,v in row['command'].items()}, row['quality'].to(device), hidden)
        assert pred['state'].shape == (1,163) and torch.isfinite(pred['state']).all()
        return pred['state'][0].cpu().numpy().copy()

    def close(self):
        task = self.task()
        if task is not None:
            task.scene.step, task._update_render = self.original_step, self.original_render
        self.rows.clear()
        self.original_step = self.original_render = None


def install(env, model, *, assets_path, gripper_calibration, image_size=(128, 128)):
    """Instance-local hooks, with terminal prediction before the normal reset."""
    calibration = RobotCalibration(assets_path, gripper_calibration)
    from gymnasium import spaces
    import types
    assert env.num_envs == 1
    adapter = env.executor.envs[0].adapter
    original_get, original_close = adapter.get_obs, adapter.close
    state = dict(reader=None, task=None)
    def get_obs(_):
        obs = original_get()
        if state['task'] is not adapter.task:
            if state['reader'] is not None:state['reader'].close()
            state.update(reader=LiveState(adapter.task, model, calibration, image_size), task=adapter.task)
        obs['state_encoder163'] = state['reader'].predict()
        return obs
    def close(_, *args, **kwargs):
        if state['reader'] is not None:state['reader'].close()
        state.update(reader=None, task=None)
        return original_close(*args, **kwargs)
    adapter.get_obs,adapter.close = types.MethodType(get_obs, adapter),types.MethodType(close, adapter)
    original_tensor = env._tensor_obs
    def tensor_obs(raw):
        # Privilege is retained outside policy observations only for the fixed
        # precollection teacher; neither actor/critic nor replay receives it.
        env._warmup_teacher_observation = {k:torch.as_tensor(np.stack([row[k] for row in raw]),
            dtype=torch.float32, device=env.device) for k in ('state','state_open_laptop_robot_v5')}
        return original_tensor(raw)
    env._tensor_obs = tensor_obs
    env.single_observation_space = spaces.Dict(dict(state=env.single_observation_space['state'],
        state_encoder163=spaces.Box(-np.inf,np.inf,(163,),np.float32)))
    env.observation_space = env.single_observation_space
    return state
