#!/usr/bin/env python3
"""Train official JAX Cal-QL on rl-garden's canonical Minari arrays.

This file owns experiment orchestration and Cal-QL/AntMaze-specific pieces
only. The policy, critics, losses, optimizers, RNG utilities, and batch
conversion are imported from the official Cal-QL checkout supplied with
``--calql-source``. Shared, non-Cal-QL-specific infra (the env bridge, the
wire protocol, dataset loading, evaluation loops, result I/O) lives in
``baselines.core``.

Invoked as ``python -m baselines.cal_ql.run_offline`` with the repo root
on PYTHONPATH -- see ``.agents/runbooks/baseline-install.md``.
"""

from __future__ import print_function

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

from baselines.core.dataset import load_npz_dataset, source_metadata
from baselines.core.evaluation import evaluate_bridge_policy, evaluate_legacy_gym_policy
from baselines.core.reporting import (
    append_jsonl,
    json_scalar,
    load_pickle_checkpoint,
    save_pickle_checkpoint,
    write_json,
)

DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "dones",
    "mc_returns",
)

OBSERVATION_KEYS = ("achieved_goal", "desired_goal", "observation")


def flatten_legacy_observation(observation, target_goal):
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    target_goal = np.asarray(target_goal, dtype=np.float32).reshape(-1)
    if observation.size != 29 or target_goal.size != 2:
        raise ValueError(
            "legacy AntMaze adapter expected obs=29 and target_goal=2, got {} and {}".format(
                observation.size, target_goal.size
            )
        )
    return np.concatenate((observation[:2], target_goal, observation[2:])).astype(
        np.float32, copy=False
    )


def _legacy_observation_adapter(raw_observation, env):
    return flatten_legacy_observation(raw_observation, env.target_goal)


def _load_dataset(path):
    return load_npz_dataset(
        path,
        keys=DATASET_KEYS,
        expected_observation_keys=OBSERVATION_KEYS,
        expected_reward_values={-5.0, 5.0},
    )


def _source_metadata(source):
    tracked_files = (
        "JaxCQL/conservative_sac.py",
        "JaxCQL/model.py",
        "JaxCQL/jax_utils.py",
        "JaxCQL/replay_buffer.py",
    )
    return source_metadata(source, tracked_files)


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calql-source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-python", required=True)
    parser.add_argument("--secondary-env-python")
    parser.add_argument("--minari-datasets-path", required=True)
    parser.add_argument("--dataset-id", default="D4RL/antmaze/large-diverse-v2")
    parser.add_argument("--legacy-env-id", default="antmaze-large-diverse-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-updates", type=int, default=1_000_000)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--final-eval-episodes", type=int, default=1000)
    parser.add_argument("--wandb-entity", default="jaimezhao")
    parser.add_argument("--wandb-project", default="d4rl")
    parser.add_argument("--wandb-name", default="calql_jax_minari_v4_s0")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def main():
    args = _build_parser().parse_args()
    source = Path(args.calql_source).resolve()
    sys.path.insert(0, str(source))

    import jax
    import wandb
    from JaxCQL.conservative_sac import ConservativeSAC
    from JaxCQL.jax_utils import batch_to_jax
    from JaxCQL.model import FullyConnectedQFunction, SamplerPolicy, TanhGaussianPolicy
    from JaxCQL.replay_buffer import subsample_batch
    from JaxCQL.utils import set_random_seed

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    dataset, dataset_manifest = _load_dataset(args.dataset)
    observation_dim = int(dataset["observations"].shape[1])
    action_dim = int(dataset["actions"].shape[1])
    if observation_dim != 31 or action_dim != 8:
        raise ValueError(
            "expected Minari AntMaze dimensions (31, 8), got ({}, {})".format(
                observation_dim, action_dim
            )
        )

    source_meta = _source_metadata(source)
    config = {
        "implementation": "official-jax-calql-with-minari-orchestration",
        "source": source_meta,
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_manifest": str(Path(args.dataset).with_suffix(".manifest.json")),
        "dataset_array_sha256": {
            key: value["sha256"]
            for key, value in dataset_manifest["arrays"].items()
        },
        "seed": args.seed,
        "offline_updates": args.offline_updates,
        "batch_size": args.batch_size,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "policy_arch": "256-256",
        "qf_arch": "256-256-256-256",
        "orthogonal_init": True,
        "reward_scale": 10.0,
        "reward_bias": -5.0,
        "discount": 0.99,
        "soft_target_update_rate": 0.005,
        "policy_lr": 1e-4,
        "qf_lr": 3e-4,
        "cql_n_actions": 10,
        "cql_lagrange": True,
        "cql_target_action_gap": 0.8,
        "cql_min_q_weight": 5.0,
        "enable_calql": True,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "jax_version": jax.__version__,
    }
    write_json(output_dir / "config.json", config)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        config=config,
        dir=str(output_dir),
        mode=args.wandb_mode,
    )
    print("WANDB_RUN_ID={}".format(run.id), flush=True)

    set_random_seed(args.seed)
    policy = TanhGaussianPolicy(
        observation_dim,
        action_dim,
        "256-256",
        True,
        1.0,
        -1.0,
    )
    qf = FullyConnectedQFunction(
        observation_dim, action_dim, "256-256-256-256", True
    )
    cql_config = ConservativeSAC.get_default_config(
        {
            "discount": 0.99,
            "target_entropy": -float(action_dim),
            "policy_lr": 1e-4,
            "qf_lr": 3e-4,
            "soft_target_update_rate": 5e-3,
            "cql_n_actions": 10,
            "cql_importance_sample": True,
            "cql_lagrange": True,
            "cql_target_action_gap": 0.8,
            "cql_temp": 1.0,
            "cql_max_target_backup": True,
            "cql_clip_diff_min": -np.inf,
            "cql_clip_diff_max": np.inf,
        }
    )
    sac = ConservativeSAC(cql_config, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params["policy"])

    primary_bridge = {
        "python_executable": args.env_python,
        "dataset_id": args.dataset_id,
        "datasets_path": args.minari_datasets_path,
        "observation_keys": OBSERVATION_KEYS,
        "initial_seed": args.seed,
    }

    def evaluate_and_log(step, episodes, label):
        result = evaluate_bridge_policy(
            sampler_policy.update_params(sac.train_params["policy"]),
            primary_bridge,
            episodes,
        )
        metrics = {
            "grad_steps": step,
            "evaluation/backend": label,
            "evaluation/average_return": result["average_return"],
            "evaluation/success_rate": result["success_rate"],
            "evaluation/average_traj_length": result["average_traj_length"],
            "evaluation/elapsed_seconds": result["elapsed_seconds"],
        }
        run.log(metrics, step=step)
        append_jsonl(metrics_path, {**metrics, "evaluation": result})
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return result

    evaluate_and_log(0, args.eval_episodes, "minari_v4_mujoco_3.11")
    last_metrics = None
    started = time.time()
    for step in range(1, args.offline_updates + 1):
        batch = batch_to_jax(subsample_batch(dataset, args.batch_size))
        last_metrics = sac.train(
            batch,
            use_cql=True,
            cql_min_q_weight=5.0,
            enable_calql=True,
        )

        if step % args.steps_per_epoch == 0:
            train_metrics = {
                "grad_steps": step,
                "epoch": step // args.steps_per_epoch,
                "elapsed_seconds": time.time() - started,
            }
            train_metrics.update(
                {"sac/" + key: json_scalar(value) for key, value in last_metrics.items()}
            )
            run.log(train_metrics, step=step)
            append_jsonl(metrics_path, train_metrics)

        if step % args.eval_every == 0 or step == args.offline_updates:
            evaluate_and_log(
                step, args.eval_episodes, "minari_v4_mujoco_3.11"
            )
            checkpoint = checkpoint_dir / "model_{:07d}.pkl".format(step)
            save_pickle_checkpoint(
                checkpoint, {"sac": sac, "metadata": {**config, "grad_steps": step}}
            )

    final_checkpoint = output_dir / "model_final.pkl"
    save_pickle_checkpoint(
        final_checkpoint,
        {"sac": sac, "metadata": {**config, "grad_steps": args.offline_updates}},
    )
    loaded = load_pickle_checkpoint(final_checkpoint)
    loaded_sac = loaded["sac"]
    final_policy = SamplerPolicy(
        loaded_sac.policy, loaded_sac.train_params["policy"]
    )

    final_results = {
        "checkpoint": str(final_checkpoint),
        "minari_v4_mujoco_3_11": evaluate_bridge_policy(
            final_policy, primary_bridge, args.final_eval_episodes
        ),
    }
    if args.secondary_env_python:
        secondary_bridge = dict(primary_bridge)
        secondary_bridge["python_executable"] = args.secondary_env_python
        final_results["minari_v4_mujoco_3_1_6"] = evaluate_bridge_policy(
            final_policy, secondary_bridge, args.final_eval_episodes
        )
    final_results["legacy_d4rl_mujoco_2_1"] = evaluate_legacy_gym_policy(
        final_policy,
        args.legacy_env_id,
        args.final_eval_episodes,
        args.seed,
        observation_adapter=_legacy_observation_adapter,
    )
    write_json(output_dir / "final_evaluation.json", final_results)
    for backend, result in final_results.items():
        if not isinstance(result, dict):
            continue
        run.log(
            {
                "grad_steps": args.offline_updates,
                "final/{}/success_rate".format(backend): result["success_rate"],
                "final/{}/average_return".format(backend): result["average_return"],
                "final/{}/average_traj_length".format(backend): result[
                    "average_traj_length"
                ],
            },
            step=args.offline_updates,
        )
    print(json.dumps(final_results, indent=2, sort_keys=True), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
