"""Privileged-state and frozen visual-state residual training configurations."""
from dataclasses import fields
from importlib.resources import files
import json

from .task_state_residual_sac_v5_scaled import ScaledTaskStateResidualSACV5Args

RECIPES = ('privileged-state', 'visual-state')
_LEGACY_RECIPES = {'B24D': 'privileged-state', 'B17B-B1E': 'visual-state'}


def recipe_args(recipe, *, act_checkpoint, teacher_checkpoint, robotwin_root,
                assets_path, log_dir='runs/open_laptop'):
    """Load the recorded scientific configuration; paths are supplied by the caller.

    No downloads, GPU access, environment creation, logging service or training.
    Caller must use build_agent below to select the matching policy class.
    """
    recipe = _LEGACY_RECIPES.get(recipe, recipe)
    if recipe not in RECIPES:
        raise ValueError(f'Expected one of {RECIPES}, got {recipe!r}')
    values = json.loads(files(__package__).joinpath('open_laptop_defaults.json').read_text())
    args = ScaledTaskStateResidualSACV5Args()
    allowed = {f.name for f in fields(args)}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f'Recipe fields not supported by this version: {sorted(unknown)}')
    for key, value in values.items():
        if key in ('robotwin', 'maniskill', 'minari'):
            config = getattr(args, key)
            unknown = value.keys() - {f.name for f in fields(config)}
            if unknown:
                raise ValueError(f'Unknown {key} fields: {sorted(unknown)}')
            for name, item in value.items():
                setattr(config, name, item)
        else:
            setattr(args, key, value)
    args.base_ckpt_path = str(act_checkpoint)
    args.residual_warmup_policy_checkpoint = str(teacher_checkpoint)
    args.robotwin.robotwin_root = str(robotwin_root)
    args.robotwin.assets_path = str(assets_path)
    args.log_dir = str(log_dir)
    args.exp_name = recipe.lower() + '-open-laptop-train220k'
    from .task_state_residual_sac import validate_task_state_args
    validate_task_state_args(args)
    return args


def build_agent(recipe, args, env, logger, checkpoint_dir, *,
                encoder_checkpoint=None, gripper_calibration=None,
                encoder_variant='rgbd', encoder_image_size=(128, 128), encoder_factory=None):
    """Build a fresh learner; never silently resume a model-only checkpoint.

    Call before the first environment reset. Attach a validated BoundaryCheckpoints
    codec before learn() and restore a verified complete bundle for continuation.
    Visual hooks only affect this environment instance.
    """
    import torch
    from rl_garden.encoders.laptop.checkpoint import sha256, load_encoder
    from .residual_sac import build_residual_sac
    recipe = _LEGACY_RECIPES.get(recipe, recipe)
    if recipe not in RECIPES:
        raise ValueError(f'Unsupported recipe: {recipe}')
    if args.load_checkpoint is not None:
        raise ValueError('Use the complete boundary codec, not model-only resume')
    weights = dict(act=sha256(args.base_ckpt_path),
                   warmup=sha256(args.residual_warmup_policy_checkpoint))
    if recipe == 'privileged-state':
        if encoder_checkpoint is not None or gripper_calibration is not None:
            raise ValueError('The privileged-state route does not use a visual encoder')
        from rl_garden.algorithms.laptop_privileged_residual import PrivilegedStateResidualSAC
        algorithm = PrivilegedStateResidualSAC
    else:
        if encoder_checkpoint is None or gripper_calibration is None:
            raise ValueError('The visual-state route requires encoder weights and gripper calibration')
        from rl_garden.algorithms.laptop_estimated_residual import VisualStateResidualSAC
        from rl_garden.envs.robotwin.laptop_estimated_state import install
        # Loading a frozen model must not perturb the learner's initialization RNG.
        with torch.random.fork_rng(devices=[]):
            encoder_identity = sha256(encoder_checkpoint)
            model = load_encoder(encoder_checkpoint, variant=encoder_variant,
                                 expected_sha256=encoder_identity, device='cpu',
                                 model_factory=encoder_factory)
        model.to(env.device)
        live = install(env, model, assets_path=args.robotwin.assets_path,
                       gripper_calibration=gripper_calibration, image_size=encoder_image_size)
        algorithm = VisualStateResidualSAC
    agent = build_residual_sac(args, env, None, logger, checkpoint_dir,
                               algorithm_class=algorithm)
    agent._laptop_recipe = recipe
    agent._laptop_inputs = dict(weights=weights)
    if recipe == 'visual-state':
        agent._laptop_inputs.update(encoder=encoder_identity, variant=encoder_variant,
                                    image_size=list(encoder_image_size),
                                    calibration=sha256(gripper_calibration))
        agent._laptop_perception_boundary = PerceptionBoundary(model, live, encoder_identity,
                                                              encoder_variant, encoder_image_size)
    return agent


class PerceptionBoundary:
    """The visual history is empty only at the released-task episode boundary."""
    def __init__(self, model, live, encoder_identity=None, variant='rgbd', image_size=(128, 128)):
        self.model, self.live = model, live
        self.encoder_identity = encoder_identity
        self.variant, self.image_size = variant, tuple(image_size)

    def validate(self, agent):
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('The pretrained encoder must stay frozen')

    def prepare_boundary(self, agent):
        pass  # The runtime closes the task after all components have prepared.

    def state_dict(self):
        if self.live['reader'] is not None or self.live['task'] is not None:
            raise ValueError('Visual history was not released at the episode boundary')
        return dict(encoder_sha=self.encoder_identity, reset_history=True,
                    variant=self.variant, image_size=self.image_size)

    def load_state_dict(self, value):
        # Older bundles used the original compact RGBD preprocessing only.
        value = dict(value)
        value.setdefault('variant', 'rgbd')
        value.setdefault('image_size', (128, 128))
        value['image_size'] = tuple(value['image_size'])
        if value != self.state_dict():
            raise ValueError('Visual boundary identity mismatch')


def attach_boundaries(agent, directory, identity, *, external, torch_generators,
                      numpy_generators, fingerprint, resume=None):
    """Attach existing boundary machinery with explicit planner/sampler inventories.

    This does not supply, infer or fake the external planner's RNG implementation.
    The caller owns its validated lifecycle. No training starts in this function.
    """
    from rl_garden.common.boundary_checkpoint import BoundaryCheckpoints, BoundaryStore
    from rl_garden.common.boundary_torch import RoboTwinBoundaryRuntime, TorchBoundaryCodec
    components = dict(external)
    if agent._laptop_recipe == 'visual-state':
        if 'perception' in components:
            raise ValueError('Do not replace the recipe visual-history contract')
        components['perception'] = agent._laptop_perception_boundary
    runtime = RoboTwinBoundaryRuntime(components)
    codec = TorchBoundaryCodec(runtime, torch_generators=torch_generators,
                               numpy_generators=numpy_generators, fingerprint=fingerprint)
    codec.validate(agent)
    store = BoundaryStore(directory, identity)
    manager = BoundaryCheckpoints(store, codec, total_steps=220000, resume=resume)
    if agent._laptop_recipe == 'visual-state':
        manager.thresholds = list(range(10000, 70001, 10000)) + list(range(75000, 220001, 5000))
        manager.pins = set(manager.thresholds)
    agent.boundary_checkpoints = manager
    return manager
