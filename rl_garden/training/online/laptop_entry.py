"""Explicit standalone training, boundary resume and fixed-scene evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'resume', 'smoke', 'evaluate'))
    parser.add_argument('--recipe', choices=('privileged-state', 'visual-state'), required=True)
    for name in ('act-checkpoint', 'teacher-checkpoint', 'robotwin-root', 'assets-path', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--encoder-checkpoint')
    parser.add_argument('--encoder-variant', choices=('rgbd', 'rgbd-ema'), default='rgbd')
    parser.add_argument('--encoder-resolution', choices=('128x128', '320x240'), default='128x128')
    parser.add_argument('--gripper-calibration')
    parser.add_argument('--planner-sha256', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--seeds', help='JSON list of explicit environment seeds for evaluation')
    parser.add_argument('--smoke', action='store_true', help='Use a 32-step integration check configuration')
    options = parser.parse_args()
    run(options)


def run(options):
    import fcntl
    import torch
    from rl_garden.common import seed_everything
    from rl_garden.common.logger import Logger
    from rl_garden.encoders.laptop.checkpoint import sha256
    from rl_garden.envs.backend_registry import make_training_envs, make_evaluation_env
    from .open_laptop_recipes import recipe_args, build_agent, attach_boundaries
    from .task_state_residual_sac import _task_state_env_request
    from .laptop_runtime import PlannerState, SamplerState, inventory

    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options.mode in ('train', 'smoke') and (output / 'configuration.json').exists():
            raise ValueError('Existing training directory: use resume or a new output directory')
        args = recipe_args(options.recipe, act_checkpoint=options.act_checkpoint,
                           teacher_checkpoint=options.teacher_checkpoint,
                           robotwin_root=options.robotwin_root, assets_path=options.assets_path,
                           log_dir=str(output))
        if options.mode == 'smoke' or options.smoke:
            args.total_timesteps = 32
            args.robotwin.step_lim = 4
            args.buffer_size = 128
            args.learning_starts, args.critic_only_steps, args.training_freq = 8, 16, 8
        if options.mode == 'evaluate':
            args.num_eval_envs = 1  # Standalone evaluation; training disables periodic evaluation.
            args.robotwin.step_lim = 700
            args.robotwin.clear_cache_freq = 5
            args.buffer_size = 1  # inference does not use replay
        sys.path.insert(0, str(Path(options.robotwin_root).resolve()))
        # Configure assets before importing the simulator's planner module.
        import os
        os.environ['ASSETS_PATH'] = str(options.assets_path)
        from rl_garden.envs.robotwin.isolated_planner import install_planner
        worker, counters, close_planner = install_planner(
            planner_source=Path(options.robotwin_root) / 'envs/robot/planner.py',
            planner_sha256=options.planner_sha256)
        env = logger = close_capture = None
        try:
            torch.set_num_threads(1)
            seed_everything(args.seed)
            req = _task_state_env_request(args, args.exp_name)
            if options.mode == 'evaluate':
                if options.recipe == 'visual-state':
                    from rl_garden.envs.robotwin.capture_reuse import install
                    close_capture = install()
                env = make_evaluation_env(args.env_backend, req)
                env.cfg.auto_reset = False
            else:
                env, unused = make_training_envs(args.env_backend, req)
                if unused is not None:
                    unused.close()
            logger = Logger.create(log_type='none', log_dir=str(output / 'logs'), run_name=args.exp_name)
            agent = build_agent(options.recipe, args, env, logger, None,
                                encoder_checkpoint=options.encoder_checkpoint,
                                gripper_calibration=options.gripper_calibration,
                                encoder_variant=getattr(options, 'encoder_variant', 'rgbd'),
                                encoder_image_size=(240, 320) if getattr(options, 'encoder_resolution', '128x128') == '320x240' else (128, 128))
            if options.mode == 'evaluate':
                evaluate(agent, options, output)
                return
            identity = dict(recipe=options.recipe, args=asdict(args),
                            inputs=agent._laptop_inputs,
                            planner_sha256=options.planner_sha256,
                            source_sha256={str(p.relative_to(Path(__file__).parents[2])): sha256(p)
                                for p in sorted(Path(__file__).parents[2].rglob('*.py'))},
                            encoder_sha256=sha256(options.encoder_checkpoint) if options.encoder_checkpoint else None,
                            calibration_sha256=sha256(options.gripper_calibration) if options.gripper_calibration else None)
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            config = output / 'configuration.json'
            if config.exists():
                if json.loads(config.read_text()) != identity:
                    raise ValueError('Configuration differs from this output directory')
            else:
                if options.mode == 'resume':
                    raise ValueError('Resume requires the original configuration')
                write_json(config, identity)
            checkpoint_dir = output / 'checkpoints'
            resume = None
            if options.mode == 'resume':
                resume = json.loads((checkpoint_dir / 'index.json').read_text())['latest']
                if not resume:
                    raise ValueError('No complete checkpoint to resume')
            tg, ng = inventory(agent)
            manager = attach_boundaries(agent, checkpoint_dir, fingerprint,
                external=dict(planner=PlannerState(worker, counters), sampler=SamplerState()),
                torch_generators=tg, numpy_generators=ng, fingerprint=fingerprint, resume=resume)
            if options.mode == 'smoke' or options.smoke:
                manager.total_steps = 32
                manager.thresholds = [8, 16, 24, 32]
                manager.pins = set(manager.thresholds)
            agent.learn(total_timesteps=args.total_timesteps)
            write_json(output / 'result.json', dict(step=agent._global_step,
                updates=agent._global_update, finished=manager.finished, planner=counters))
        finally:
            from contextlib import ExitStack
            # Release every owned resource even if a renderer close raises.
            with ExitStack() as cleanup:
                if logger is not None:
                    cleanup.callback(logger.close)
                cleanup.callback(close_planner)
                if close_capture is not None:
                    cleanup.callback(close_capture)
                if env is not None:
                    cleanup.callback(env.close)


def evaluate(agent, options, output):
    """Resume only complete episodes; no synthetic completion of interrupted rows."""
    import torch
    from rl_garden.encoders.laptop.checkpoint import sha256
    if not options.checkpoint_sha256 or sha256(options.checkpoint) != options.checkpoint_sha256:
        raise ValueError('Evaluation checkpoint SHA256 mismatch')
    if hasattr(agent, '_laptop_inputs'):
        checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        recorded = checkpoint.get('metadata', {}).get('hyperparameters', {}).get('laptop_inputs')
        if recorded is not None and recorded != agent._laptop_inputs:
            raise ValueError('Evaluation inputs differ from the trained policy')
        del checkpoint
    seeds = json.loads(Path(options.seeds).read_text())
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Expected distinct nonnegative environment seeds')
    manifest = dict(recipe=options.recipe, checkpoint_sha256=options.checkpoint_sha256,
                    inputs=getattr(agent, '_laptop_inputs', {}),
                    source_sha256={str(p.relative_to(Path(__file__).parents[2])): sha256(p)
                        for p in sorted(Path(__file__).parents[2].rglob('*.py'))},
                    planner_sha256=options.planner_sha256, seeds=seeds, step_limit=700,
                    encoder_sha256=sha256(options.encoder_checkpoint) if options.encoder_checkpoint else None,
                    calibration_sha256=sha256(options.gripper_calibration) if options.gripper_calibration else None)
    path = output / 'evaluation.json'
    saved = json.loads(path.read_text()) if path.exists() else dict(manifest=manifest, episodes=[])
    if saved['manifest'] != manifest:
        raise ValueError('Evaluation identity or scene list changed')
    rows = saved['episodes']
    if len(rows) > len(seeds) or any(r['episode'] != i + 1 or r['env_seed'] != seeds[i]
                                    for i, r in enumerate(rows)):
        raise ValueError('Evaluation ledger is not a contiguous scene prefix')
    agent.load(options.checkpoint, strict=True, load_replay_buffer=False, load_optimizers=False)
    agent.policy.eval()
    env = agent.env
    for i in range(len(rows), len(seeds)):
        env._success_seeds = torch.tensor([seeds[i]], dtype=torch.long)
        env._current_seed_index = 0
        agent.base_action_provider.reset()
        agent._cached_base_actions = None
        obs, info = env.reset()
        if int(info['env_seed'][0]) != seeds[i]:
            raise ValueError('Simulator substituted the requested scene')
        reward_sum, success = 0., False
        for length in range(1, 701):
            with torch.inference_mode():
                action = agent.get_action(obs, deterministic=True)
            if not torch.isfinite(action).all():
                raise ValueError('Nonfinite evaluation action')
            obs, rewards, terms, truncs, info = env.step(action)
            reward_sum += float(rewards[0])
            success |= bool(info['success'][0])
            if bool((terms | truncs)[0]):
                break
        else:
            raise RuntimeError('Environment did not terminate at the declared horizon')
        rows.append({'episode': i + 1, 'env_seed': seeds[i], 'success': success,
                     'length': length, 'return': reward_sum})
        write_json(path, dict(manifest=manifest, episodes=rows, finished=len(rows) == len(seeds),
                             success_rate=sum(r['success'] for r in rows) / len(rows)))


if __name__ == '__main__':
    main()
