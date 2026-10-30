"""Privileged-state residual SAC, with explicit warmup action units."""
import copy
import torch
from rl_garden.algorithms.scaled_task_state_residual import ScaledTaskStateResidualSACV5
from rl_garden.common.checkpoint import load_checkpoint_file, space_metadata

def validate_warmup(checkpoint, observation_space, action_space):
    """Accept compatible trained policies, independently of file name or training step."""
    metadata = checkpoint['metadata']
    hp = metadata['hyperparameters']
    if metadata['observation_space'] != space_metadata(observation_space):
        raise ValueError('Warmup observation interface mismatch')
    if metadata['action_space'] != space_metadata(action_space):
        raise ValueError('Warmup action interface mismatch')
    alignment = hp['base_action_alignment']
    expected = dict(ee_delta_pos_scale=.06, ee_delta_rot_scale=.3, gripper_delta_scale=.2,
                    residual_position_ratio=.1, residual_rotation_ratio=.1,
                    residual_gripper_ratio=.2)
    if any(alignment.get(key) != value for key, value in expected.items()):
        raise ValueError('Warmup action units or residual scale mismatch')
    return hp

class TeacherAdapter(torch.nn.Module):
    def __init__(self,policy,student_scales):
        super().__init__();self.policy=policy
        old=torch.full((14,),.1,device=student_scales.device);old[[6,13]]=.2
        self.register_buffer('compensation',old/student_scales)
    def predict(self,obs,*,base_actions,deterministic=True):
        return self.policy.predict(obs,base_actions=base_actions,deterministic=True)*self.compensation

class PrivilegedStateResidualSAC(ScaledTaskStateResidualSACV5):
    _compatible_checkpoint_algorithms=('PrivilegedStateResidualSAC', 'B24ResidualSAC')
    def _residual_scale_tensor(self,reference):
        out=super()._residual_scale_tensor(reference)
        out[..., [3,4,5,10,11,12]]=.2
        return out
    def _load_state_warmup_policy(self,path):
        checkpoint=load_checkpoint_file(path,map_location=self.device)
        hp=validate_warmup(checkpoint, self.env.single_observation_space, self.env.single_action_space)
        assert hp['selected_state_schema']==super()._checkpoint_metadata()['selected_state_schema']
        assert hp['base_action_alignment']['residual_position_ratio']==.1
        assert hp['base_action_alignment']['residual_rotation_ratio']==.1
        policy=copy.deepcopy(self.policy);target=policy.state_dict();source=checkpoint['state']['policy']
        prefixes=('features_extractor.','actor.','_actor_adapter.')
        keys={k for k in target if k.startswith(prefixes)}
        assert keys=={k for k in source if k.startswith(prefixes)}
        for k in keys:
            assert source[k].shape==target[k].shape and torch.isfinite(source[k]).all(),k
            if k.startswith('features_extractor.'):assert torch.equal(source[k],target[k]),k
            target[k]=source[k]
        policy.load_state_dict(target,strict=True);policy.eval().requires_grad_(False)
        return TeacherAdapter(policy,self._residual_scale_tensor(torch.zeros(14,device=self.device)))
    def _rollout_action(self,obs,learning_has_started):
        if learning_has_started:return super()._rollout_action(obs,learning_has_started)
        base=self._base_naction(obs) if self._cached_base_actions is None else self._cached_base_actions
        self._cached_base_actions=None
        with torch.no_grad():
            unit=torch.zeros_like(base) if self._warmup_policy is None else self._warmup_policy.predict(self._obs_to_policy_device(obs),base_actions=base)
        final=self._combine_base_residual(base,unit)
        return final,self.action_scaler.unscale(final),{'base_actions':base}
    def _sample_warmup_policy_selection(self,env_ids=None):
        # Both arms of this experiment use deterministic, 100% chosen collectors.
        # No unused Bernoulli draws that would confound pure-ACT vs teacher groups.
        self._warmup_use_policy=None
    def _checkpoint_metadata(self):
        meta=super()._checkpoint_metadata()
        if hasattr(self, '_laptop_inputs'):
            meta['laptop_inputs'] = self._laptop_inputs
        meta['base_action_alignment'] = dict(profile='delta_ee_arm_units_v1',
            base='ACT', ee_delta_pos_scale=.06, ee_delta_rot_scale=.30,
            gripper_delta_scale=.2, residual_gripper_ratio=.2, gripper_unit_ratio=1.,
            residual_position_ratio=.1,residual_rotation_ratio=.2,
            warmup_arm_base_input_ratio=1.,warmup_arm_output_ratio=None,warmup_base_input_clipping=None)
        meta['base_action_alignment']['residual_arm_ratio']='split: position .1 / rotation .2'
        meta['collector']={'teacher':'privileged-state' if self._warmup_policy is not None else 'pure_ACT',
            'probability':1.,'teacher_pos_rot_gripper':[.1,.1,.2] if self._warmup_policy is not None else None,
            'student_pos_rot_gripper':[.1,.2,.2],'historical_replay_loaded':False}
        return meta


# Import compatibility for older launchers and pickled class references.
B24ResidualSAC = PrivilegedStateResidualSAC
