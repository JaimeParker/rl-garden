import json
from types import SimpleNamespace

import pytest
import torch

from rl_garden.encoders.laptop.checkpoint import sha256
from rl_garden.training.online.laptop_entry import evaluate


@pytest.mark.parametrize('recipe', ['privileged-state', 'visual-state'])
@pytest.mark.parametrize('mode', ['evaluate', 'train', 'resume', 'smoke'])
def test_entrypoint_environment_count_is_scoped_to_evaluation(tmp_path, monkeypatch, recipe, mode):
    import sys
    from rl_garden.envs import backend_registry
    from rl_garden.envs.robotwin import isolated_planner, capture_reuse
    from rl_garden.training.online.laptop_entry import run

    class RequestChecked(Exception):
        pass

    def check_request(backend, request):
        assert request.num_envs == 1
        assert request.num_eval_envs == (1 if mode == 'evaluate' else 0)
        raise RequestChecked()

    monkeypatch.setattr(backend_registry, 'make_evaluation_env', check_request)
    monkeypatch.setattr(backend_registry, 'make_training_envs', check_request)
    monkeypatch.setattr(isolated_planner, 'install_planner', lambda **kw: (None, {}, lambda: None))
    monkeypatch.setattr(capture_reuse, 'install', lambda: lambda: None)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setenv('ASSETS_PATH', str(tmp_path))
    options = SimpleNamespace(mode=mode, recipe=recipe, output=str(tmp_path), smoke=False,
        act_checkpoint='act.pt', teacher_checkpoint='teacher.pt', robotwin_root=str(tmp_path),
        assets_path=str(tmp_path), planner_sha256='a' * 64)
    with pytest.raises(RequestChecked):
        run(options)


def test_fresh_training_rejects_existing_directory_before_simulation(tmp_path):
    from rl_garden.training.online.laptop_entry import run
    (tmp_path / 'configuration.json').write_text('{}')
    with pytest.raises(ValueError, match='use resume'):
        run(SimpleNamespace(output=str(tmp_path), mode='train'))


def test_evaluation_preserves_completed_prefix_and_rejects_changed_scenes(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'trusted test fixture')
    seeds = tmp_path / 'scenes.json'
    seeds.write_text('[11, 12]')
    options = SimpleNamespace(recipe='privileged-state', checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), seeds=str(seeds), planner_sha256='a' * 64,
        encoder_checkpoint=None, gripper_calibration=None)
    class Env:
        resets = 0
        def reset(self):
            self.resets += 1
            return {}, {'env_seed': self._success_seeds}
        def step(self, action):
            return {}, torch.tensor([1.]), torch.tensor([True]), torch.tensor([False]), {'success': torch.tensor([True])}
    env = Env()
    agent = SimpleNamespace(env=env, policy=SimpleNamespace(eval=lambda: None),
        base_action_provider=SimpleNamespace(reset=lambda: None),
        load=lambda *a, **kw: None, get_action=lambda *a, **kw: torch.zeros(1, 14))
    evaluate(agent, options, tmp_path)
    first = json.loads((tmp_path / 'evaluation.json').read_text())
    assert env.resets == 2 and first['finished'] and first['success_rate'] == 1
    assert first['episodes'][0]['return'] == 1
    evaluate(agent, options, tmp_path)
    assert env.resets == 2
    first['episodes'] = first['episodes'][:1]
    first['finished'] = False
    (tmp_path / 'evaluation.json').write_text(json.dumps(first))
    evaluate(agent, options, tmp_path)
    assert env.resets == 3
    seeds.write_text('[11, 13]')
    with pytest.raises(ValueError, match='identity or scene list'):
        evaluate(agent, options, tmp_path)
