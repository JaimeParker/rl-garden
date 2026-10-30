import copy

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms.laptop_estimated_residual import EstimatedStateExtractor
from rl_garden.algorithms.laptop_privileged_residual import TeacherAdapter
from rl_garden.encoders.laptop.checkpoint import load_encoder, sha256
from rl_garden.encoders.laptop.ema import make_ema, update_ema
from rl_garden.encoders.laptop.model import LaptopEncoder
from rl_garden.training.online.open_laptop_recipes import recipe_args


def observable_sample(height=128, width=128):
    """Synthetic observable input, without object labels or simulator handles."""
    eye = torch.eye(3).reshape(1, 1, 3, 3)
    obs = dict(rgb=torch.rand(1, 3, 3, height, width),
               depth_m=torch.full((1, 3, 1, height, width), .7),
               depth_valid=torch.ones(1, 3, 1, height, width, dtype=torch.bool),
               view_valid=torch.ones(1, 3, dtype=torch.bool),
               K=eye.expand(1, 3, 3, 3).clone(),
               T_BC=torch.eye(4).reshape(1, 1, 4, 4).expand(1, 3, 4, 4).clone(),
               dt_s=torch.zeros(1, 1), reset=torch.ones(1, dtype=torch.bool))
    obs['K'][..., 0, 0] = 100
    obs['K'][..., 1, 1] = 100
    sensor = dict(q=torch.zeros(1, 12), dq=torch.zeros(1, 12),
                  width=torch.full((1, 2), .04), width_rate=torch.zeros(1, 2),
                  tcp_p=torch.zeros(1, 2, 3), tcp_R=eye.expand(1, 2, 3, 3).clone(),
                  tcp_J=torch.zeros(1, 2, 6, 6))
    command = dict(reference_p=torch.zeros(1, 2, 3), reference_R=sensor['tcp_R'].clone(),
                   width_target=sensor['width'].clone(), q_target=torch.zeros(1, 12),
                   q_target_valid=torch.ones(1, 2, dtype=torch.bool))
    quality = torch.zeros(1, 15)
    quality[:, :3] = 1
    quality[:, 9:12] = 1
    return obs, sensor, command, quality


@pytest.mark.parametrize('height,width', [(128, 128), (240, 320)])
def test_four_frame_forward_and_no_privileged_keys(height, width):
    model = LaptopEncoder().eval()
    obs, sensor, command, quality = observable_sample(height, width)
    hidden = None
    with torch.no_grad():
        for frame in range(4):
            obs['reset'].fill_(frame == 0)
            obs['dt_s'].fill_(0 if frame == 0 else .2)
            result, hidden = model(obs, sensor, command, quality, hidden)
            assert result['state'].shape == (1, 163)
            assert torch.isfinite(result['state']).all()
        with pytest.raises(ValueError, match='observable inputs'):
            model(dict(obs, object_pose=torch.zeros(1, 7)), sensor, command, quality, hidden)


@pytest.mark.parametrize('recipe', ['privileged-state', 'visual-state'])
def test_recipe_preserves_training_protocol(recipe):
    args = recipe_args(recipe, act_checkpoint='act.pt', teacher_checkpoint='teacher.pt',
                       robotwin_root='robotwin', assets_path='assets')
    assert args.robotwin.step_lim == 200
    assert (args.total_timesteps, args.learning_starts, args.critic_only_steps) == (220000, 10000, 20000)
    assert (args.batch_size, args.training_freq, args.utd) == (64, 64, .25)
    assert args.bootstrap_at_done == 'truncated'
    assert args.log_type == 'none' and args.load_checkpoint is None
    assert args.checkpoint_freq == 0 and not args.save_final_checkpoint
    assert args.robotwin.random_head_camera_dis == 0
    assert args.robotwin.ee_delta_pos_scale == .06
    assert args.robotwin.ee_delta_rot_scale == .3


def test_estimated_features_do_not_read_act_state_or_privilege():
    space = spaces.Dict(state=spaces.Box(-np.inf, np.inf, (14,), np.float32),
                        state_encoder163=spaces.Box(-np.inf, np.inf, (163,), np.float32))
    model = EstimatedStateExtractor(space)
    state = torch.ones(2, 163)
    first = model(dict(state=torch.zeros(2, 14), state_encoder163=state))
    second = model(dict(state=torch.full((2, 14), float('nan')), state_encoder163=state))
    assert torch.equal(first, second)
    assert torch.equal(state, torch.ones_like(state))
    extra = dict(space.spaces, privileged=spaces.Box(-1, 1, (3,), np.float32))
    with pytest.raises(AssertionError):
        EstimatedStateExtractor(spaces.Dict(extra))


def test_teacher_compensation_preserves_physical_action():
    class Policy(torch.nn.Module):
        def predict(self, obs, **kwargs):
            return torch.ones(1, 14)
    scales = torch.tensor([.1, .1, .1, .2, .2, .2, .2] * 2)
    adapter = TeacherAdapter(Policy(), scales)
    got = adapter.predict(None, base_actions=torch.zeros(1, 14)) * scales
    assert torch.equal(got, torch.tensor([[.1] * 6 + [.2] + [.1] * 6 + [.2]]))


def test_visual_boundary_requires_released_history():
    from rl_garden.training.online.open_laptop_recipes import PerceptionBoundary
    model = torch.nn.Linear(1, 1).requires_grad_(False)
    live = dict(reader=None, task=None)
    boundary = PerceptionBoundary(model, live)
    boundary.validate(None)
    state = boundary.state_dict()
    boundary.load_state_dict(state)
    with pytest.raises(ValueError, match='identity'):
        boundary.load_state_dict(dict(state, encoder_sha='different'))
    live['reader'] = object()
    with pytest.raises(ValueError, match='history'):
        boundary.state_dict()


def test_ema_warmup_and_resume_next_update():
    online = torch.nn.Linear(2, 1)
    ema = make_ema(online)
    update_ema(ema, online, 5000)
    for a, b in zip(online.parameters(), ema.module.parameters()):
        assert torch.equal(a, b)
    saved = copy.deepcopy(ema.state_dict())
    resumed = make_ema(online)
    resumed.load_state_dict(saved)
    with torch.no_grad():
        online.weight.add_(1)
    update_ema(ema, online, 5001)
    update_ema(resumed, online, 5001)
    for key, value in ema.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])


def test_ema_loader_never_selects_online_weights(tmp_path):
    online = LaptopEncoder()
    ema = make_ema(online)
    update_ema(ema, online, 5001)
    saved = copy.deepcopy(ema.state_dict())
    with torch.no_grad():
        next(online.parameters()).add_(1)
    path = tmp_path / 'encoder.pt'
    torch.save(dict(config=dict(variant='rgbd-ema'),
                    model=online.state_dict(), ema=saved), path)
    loaded = load_encoder(path, variant='rgbd-ema', expected_sha256=sha256(path))
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, saved['module.' + key])
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError, match='SHA256'):
        load_encoder(path, variant='rgbd-ema', expected_sha256='bad')
    with pytest.raises(ValueError, match='Expected rgbd'):
        load_encoder(path, variant='rgbd', expected_sha256=sha256(path))


def test_supervised_window_updates_once_and_keeps_temporal_gradients(monkeypatch):
    from rl_garden.encoders.laptop import supervised

    class Temporal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, obs, sensors, command, quality, hidden):
            hidden = self.weight * sensors + (0 if hidden is None else hidden)
            return hidden, hidden

    model = Temporal()
    optimizer = supervised.make_optimizer(model)
    frames = [dict(obs=dict(reset=torch.tensor([i == 0])), sensors=torch.tensor(1.),
                   command=None, quality=None) for i in range(4)]
    frames[-1]['label'] = torch.tensor(0.)
    monkeypatch.setattr(supervised, 'supervised_loss',
                        lambda prediction, label: ((prediction - label).square(), {}))
    ema = make_ema(model)
    loss = supervised.train_window(model, optimizer, frames, ema=ema, completed_updates=1)
    assert loss == 16.
    assert model.weight.item() < 1.
    assert optimizer.state[model.weight]['step'].item() == 1
    assert torch.equal(model.weight, ema.module.weight)
    with pytest.raises(ValueError, match='four frames'):
        supervised.train_window(model, optimizer, frames[:3], completed_updates=2)
