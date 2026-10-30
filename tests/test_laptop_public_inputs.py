import copy
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.common.checkpoint import space_metadata, validate_checkpoint_metadata, FORMAT_VERSION
from rl_garden.algorithms.laptop_privileged_residual import validate_warmup, PrivilegedStateResidualSAC
from rl_garden.algorithms.laptop_estimated_residual import VisualStateResidualSAC
from rl_garden.envs.robotwin.laptop_estimated_state import resize_rgbd
from rl_garden.training.online.open_laptop_recipes import PerceptionBoundary


def test_warmup_accepts_new_checkpoint_but_checks_interface_and_units():
    obs = spaces.Dict(state=spaces.Box(-1, 1, (14,), np.float32))
    action = spaces.Box(-1, 1, (14,), np.float32)
    checkpoint = dict(metadata=dict(global_step=12345, algorithm_class='UserTrainedPolicy',
        observation_space=space_metadata(obs), action_space=space_metadata(action),
        hyperparameters=dict(base_action_alignment=dict(ee_delta_pos_scale=.06,
            ee_delta_rot_scale=.3, gripper_delta_scale=.2, residual_position_ratio=.1,
            residual_rotation_ratio=.1, residual_gripper_ratio=.2))))
    validate_warmup(checkpoint, obs, action)
    invalid = copy.deepcopy(checkpoint)
    invalid['metadata']['hyperparameters']['base_action_alignment']['ee_delta_rot_scale'] = .6
    with pytest.raises(ValueError, match='units'):
        validate_warmup(invalid, obs, action)
    with pytest.raises(ValueError, match='observation'):
        validate_warmup(checkpoint, spaces.Box(-1, 1, (2,), np.float32), action)


@pytest.mark.parametrize('size', [(128, 128), (240, 320)])
def test_resize_keeps_depth_validity_and_calibration(size):
    rgb = torch.randint(0, 256, (3, 3, 240, 320), dtype=torch.uint8)
    depth = torch.full((3, 1, 240, 320), .1234)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[..., :30, :40] = False
    intrinsics = torch.tensor([[200., 0, 159.5], [0, 200., 119.5], [0, 0, 1.]]).repeat(3, 1, 1)
    result = resize_rgbd(rgb, depth, valid, intrinsics, size)
    assert result['rgb'].shape == (3, 3, *size)
    assert torch.equal(result['depth_m'], torch.full_like(result['depth_m'], .123))
    assert not result['depth_valid'][..., 0, 0].any()
    if size == (240, 320):
        assert torch.equal(result['rgb'], rgb.float()/255)
        assert torch.equal(result['K'], intrinsics)
    else:
        transform = torch.tensor([[128/320, 0, (128/320)*.5-.5],
                                  [0, 128/240, (128/240)*.5-.5], [0, 0, 1.]])
        assert torch.equal(result['K'], transform@intrinsics)
        assert torch.equal(result['rgb'], torch.nn.functional.interpolate(
            rgb.float()/255, size=size, mode='bilinear', align_corners=False))


def test_perception_boundary_legacy_and_changed_inputs():
    model = torch.nn.Linear(1, 1).requires_grad_(False)
    live = dict(reader=None, task=None)
    compact = PerceptionBoundary(model, live, 'weights')
    legacy = dict(encoder_sha='weights', reset_history=True)
    compact.load_state_dict(legacy)
    for boundary in (PerceptionBoundary(model, live, 'other'),
                     PerceptionBoundary(model, live, 'weights', image_size=(240, 320))):
        with pytest.raises(ValueError, match='identity'):
            boundary.load_state_dict(legacy)


@pytest.mark.parametrize('cls,old_name', [(VisualStateResidualSAC, 'B17SAC'),
                                        (PrivilegedStateResidualSAC, 'B24ResidualSAC')])
def test_old_and_new_algorithm_names_load(cls, old_name):
    obs = spaces.Box(-1, 1, (3,), np.float32)
    action = spaces.Box(-1, 1, (14,), np.float32)
    for name in (old_name, cls.__name__):
        checkpoint = dict(format_version=FORMAT_VERSION, metadata=dict(algorithm_class=name,
            observation_space=space_metadata(obs), action_space=space_metadata(action)))
        validate_checkpoint_metadata(checkpoint, algorithm_class=cls.__name__,
            compatible_algorithms=cls._compatible_checkpoint_algorithms,
            observation_space=obs, action_space=action, strict=True)


@pytest.mark.parametrize('variant,size', [('rgbd', (128, 128)), ('rgbd-ema', (240, 320))])
def test_builder_accepts_user_encoder_and_records_inputs(tmp_path, monkeypatch, variant, size):
    from types import SimpleNamespace
    from rl_garden.encoders.laptop.model import LaptopEncoder
    from rl_garden.encoders.laptop.ema import make_ema
    from rl_garden.training.online import open_laptop_recipes, residual_sac
    from rl_garden.envs.robotwin import laptop_estimated_state

    model = LaptopEncoder()
    encoder = tmp_path / 'my_encoder.pt'
    torch.save(dict(variant=variant, model=model.state_dict(), ema=make_ema(model).state_dict()), encoder)
    act, warmup, calibration = [tmp_path / name for name in ('act.pt', 'warmup.pt', 'calibration.json')]
    for path in (act, warmup, calibration):
        path.write_text('test fixture')
    args = SimpleNamespace(load_checkpoint=None, base_ckpt_path=str(act),
        residual_warmup_policy_checkpoint=str(warmup), robotwin=SimpleNamespace(assets_path='assets'))
    def install(env, loaded, **kwargs):
        assert kwargs['image_size'] == size
        assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
        return dict(reader=None, task=None)
    monkeypatch.setattr(laptop_estimated_state, 'install', install)
    monkeypatch.setattr(residual_sac, 'build_residual_sac', lambda *args, **kwargs: SimpleNamespace())
    rng = torch.get_rng_state().clone()
    agent = open_laptop_recipes.build_agent('visual-state', args, SimpleNamespace(device='cpu'), None,
        None, encoder_checkpoint=str(encoder), gripper_calibration=str(calibration),
        encoder_variant=variant, encoder_image_size=size)
    assert torch.equal(rng, torch.get_rng_state())
    assert agent._laptop_inputs['image_size'] == list(size)
    assert agent._laptop_inputs['variant'] == variant


def test_evaluation_rejects_replaced_encoder_before_rollout(tmp_path):
    from types import SimpleNamespace
    from rl_garden.encoders.laptop.checkpoint import sha256
    from rl_garden.training.online.laptop_entry import evaluate
    checkpoint = tmp_path / 'policy.pt'
    torch.save(dict(metadata=dict(hyperparameters=dict(laptop_inputs=dict(encoder='original')))), checkpoint)
    options = SimpleNamespace(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(SimpleNamespace(_laptop_inputs=dict(encoder='replacement')), options, tmp_path)


@pytest.mark.parametrize('wrapper', ['raw', 'model', 'state_dict'])
def test_user_model_factory_and_weight_formats(tmp_path, wrapper):
    from rl_garden.encoders.laptop.checkpoint import load_encoder, sha256
    # Loader only owns weight compatibility; the runtime owns the forward contract.
    source = torch.nn.Linear(2, 3)
    state = source.state_dict()
    path = tmp_path / 'encoder.pt'
    torch.save(state if wrapper == 'raw' else {wrapper: state}, path)
    model = load_encoder(path, variant='rgbd', expected_sha256=sha256(path),
                         model_factory=lambda: torch.nn.Linear(2, 3))
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    with pytest.raises(RuntimeError):
        load_encoder(path, variant='rgbd', expected_sha256=sha256(path),
                     model_factory=lambda: torch.nn.Linear(2, 4))
