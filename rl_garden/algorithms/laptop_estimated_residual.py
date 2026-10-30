"""Residual SAC using a frozen supervised visual state estimator."""
import math
import weakref
import numpy as np
import torch
from gymnasium import spaces
from rl_garden.algorithms.residual import ResidualSAC
from rl_garden.encoders.base import BaseFeaturesExtractor


class EstimatedStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        assert set(observation_space.spaces) == {'state','state_encoder163'}
        assert observation_space['state_encoder163'].shape == (163,)
        super().__init__(observation_space, features_dim=163)
        scale = torch.ones(163)
        scale[:24] = math.pi
        scale[24:32] = .1
        scale[32:56] = math.pi
        scale[80:86] = math.pi
        scale[86:92] = .25
        scale[122:126] = .25  # two part width/height pairs
        scale[134:146] = .25
        scale[159] = .1  # predicted position std in metres
        scale[160] = math.pi  # predicted angle std in radians
        self.register_buffer('state_divisor', scale)

    def forward(self, obs):
        value = obs['state_encoder163']
        assert value.shape[-1] == 163 and torch.isfinite(value).all()
        return value/self.state_divisor


class PrecollectionTeacher(torch.nn.Module):
    def __init__(self, policy, env, scales):
        super().__init__()
        self.policy = policy.eval().requires_grad_(False)
        self.environment = weakref.ref(env)
        old = torch.tensor([.1]*6+[.2]+[.1]*6+[.2],device=scales.device)
        self.register_buffer('compensation', old/scales)

    def predict(self, ignored_student_obs, *, base_actions, deterministic=True):
        privileged = self.environment()._warmup_teacher_observation
        assert set(privileged) == {'state','state_open_laptop_robot_v5'}
        return self.policy.predict(privileged,base_actions=base_actions,deterministic=deterministic)*self.compensation


class VisualStateResidualSAC(ResidualSAC):
    _compatible_checkpoint_algorithms = ('VisualStateResidualSAC', 'B17SAC')

    def _build_features_extractor(self):
        return EstimatedStateExtractor(self.env.single_observation_space)

    def _residual_scale_tensor(self, reference):
        scale = super()._residual_scale_tensor(reference)
        scale[..., [3,4,5,10,11,12]] = .2
        return scale

    def _sample_warmup_policy_selection(self, env_ids=None):
        self._warmup_use_policy = None

    def _rollout_action(self, obs, learning_has_started):
        if learning_has_started:
            return super()._rollout_action(obs, learning_has_started)
        base = self._base_naction(obs) if self._cached_base_actions is None else self._cached_base_actions
        self._cached_base_actions = None
        with torch.no_grad():
            unit = self._warmup_policy.predict(obs,base_actions=base,deterministic=True)
        final = self._combine_base_residual(base, unit)
        return final,self.action_scaler.unscale(final),{'base_actions':base}

    def _load_state_warmup_policy(self, path):
        from .laptop_privileged_residual import validate_warmup
        from rl_garden.encoders.scaled_task_state import ScaledTaskStateExtractor
        from rl_garden.policies.residual_policy import ResidualSACPolicy
        ck = torch.load(path,map_location=self.device,weights_only=False,mmap=True)
        hp = ck['metadata']['hyperparameters']
        observation = spaces.Dict(dict(state=spaces.Box(-np.inf,np.inf,(14,),np.float32),
            state_open_laptop_robot_v5=spaces.Box(-np.inf,np.inf,(132,),np.float32)))
        validate_warmup(ck, observation, self.env.single_action_space)
        policy = ResidualSACPolicy(observation, self._residual_action_space,
            ScaledTaskStateExtractor(observation), net_arch=hp['net_arch'],
            n_critics=hp['n_critics'], critic_subsample_size=hp['critic_subsample_size'],critic_impl=hp['critic_impl'],
            actor_use_layer_norm=hp['actor_use_layer_norm'],critic_use_layer_norm=hp['critic_use_layer_norm'],
            log_std_min=hp['actor_log_std_min'],log_std_mode=hp['actor_log_std_mode'],
            residual_actor_zero_init=hp['residual_actor_zero_init'],residual_log_std_init=hp['residual_log_std_init']).to(self.device)
        expected = policy.features_extractor.state_dict()
        for key, value in expected.items():
            if not torch.equal(value, ck['state']['policy']['features_extractor.' + key]):
                raise ValueError('Warmup state normalization mismatch')
        policy.load_state_dict(ck['state']['policy'],strict=True)
        return PrecollectionTeacher(policy,self.env,self._residual_scale_tensor(torch.zeros(14,device=self.device)))

    def _checkpoint_metadata(self):
        meta = super()._checkpoint_metadata()
        if hasattr(self, '_laptop_inputs'):
            meta['laptop_inputs'] = self._laptop_inputs
        meta['visual_state'] = dict(actor='estimated163_only', critic='estimated163_only',
            actual_joint_velocity=True, frozen_encoder=True, privilege='precollection teacher only; not replay',
            ACT_state14_unchanged=True, scaling='fixed physical divisors, no clipping/running stats',
            observation='threeRGBD/calibration + measuredq/dq/targets/FK/J, four causal real frames')
        return meta


# Import compatibility for older launchers and pickled class references.
B17SAC = VisualStateResidualSAC
