#!/usr/bin/env python3
"""Train official JAX Cal-QL on rl-garden's canonical Minari arrays.

This file owns experiment orchestration only. The policy, critics, losses,
optimizers, RNG utilities, and batch conversion are imported from the official
Cal-QL checkout supplied with ``--calql-source``.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

try:
    from .calql_minari_protocol import (
        CLOSE_REQUEST,
        HANDSHAKE,
        MAGIC,
        OP_CLOSE,
        OP_RESET,
        OP_STEP,
        RESET_REQUEST,
        STEP_REQUEST,
        STEP_RESULT,
        read_exact,
        read_status,
    )
except ImportError:  # Direct script execution.
    from calql_minari_protocol import (
        CLOSE_REQUEST,
        HANDSHAKE,
        MAGIC,
        OP_CLOSE,
        OP_RESET,
        OP_STEP,
        RESET_REQUEST,
        STEP_REQUEST,
        STEP_RESULT,
        read_exact,
        read_status,
    )


DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "dones",
    "mc_returns",
)


class GymnasiumBridge(object):
    def __init__(
        self,
        python_executable,
        server_script,
        dataset_id,
        datasets_path,
        observation_keys,
        initial_seed,
    ):
        env = os.environ.copy()
        env["MINARI_DATASETS_PATH"] = datasets_path
        command = [
            python_executable,
            server_script,
            "--dataset-id",
            dataset_id,
            "--observation-keys",
            ",".join(observation_keys),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open environment bridge pipes")
        self.reader = self.process.stdout
        self.writer = self.process.stdin
        magic, self.observation_dim, self.action_dim, self.horizon = HANDSHAKE.unpack(
            read_exact(self.reader, HANDSHAKE.size)
        )
        if magic != MAGIC:
            raise RuntimeError("invalid environment bridge handshake: {!r}".format(magic))
        self.initial_seed = int(initial_seed)
        self._reset_count = 0
        self._closed = False

    def reset(self):
        seed = self.initial_seed if self._reset_count == 0 else -1
        self._reset_count += 1
        self.writer.write(RESET_REQUEST.pack(OP_RESET, seed))
        self.writer.flush()
        read_status(self.reader)
        return np.frombuffer(
            read_exact(self.reader, self.observation_dim * 4), dtype="<f4"
        ).copy()

    def step(self, action):
        action = np.asarray(action, dtype="<f4").reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(
                "expected action dimension {}, got {}".format(
                    self.action_dim, action.size
                )
            )
        self.writer.write(STEP_REQUEST.pack(OP_STEP))
        self.writer.write(action.tobytes())
        self.writer.flush()
        read_status(self.reader)
        observation = np.frombuffer(
            read_exact(self.reader, self.observation_dim * 4), dtype="<f4"
        ).copy()
        reward, terminated, truncated = STEP_RESULT.unpack(
            read_exact(self.reader, STEP_RESULT.size)
        )
        return observation, float(reward), bool(terminated), bool(truncated)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None:
                self.writer.write(CLOSE_REQUEST.pack(OP_CLOSE))
                self.writer.flush()
                read_status(self.reader)
        finally:
            self.writer.close()
            self.reader.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


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


def wilson_interval(successes, episodes, z=1.959963984540054):
    if episodes <= 0:
        return [float("nan"), float("nan")]
    proportion = float(successes) / float(episodes)
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return [center - radius, center + radius]


def summarize_episodes(returns, lengths, normalized_returns=None):
    successes = int(sum(value > 0.5 for value in returns))
    result = {
        "episodes": len(returns),
        "successes": successes,
        "success_rate": successes / float(len(returns)),
        "success_rate_wilson95": wilson_interval(successes, len(returns)),
        "average_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "average_traj_length": float(np.mean(lengths)),
        "traj_length_std": float(np.std(lengths)),
    }
    if normalized_returns is not None:
        result["average_normalized_return"] = float(np.mean(normalized_returns))
    return result


def evaluate_bridge_policy(policy, bridge_kwargs, episodes):
    returns = []
    lengths = []
    started = time.time()
    with GymnasiumBridge(**bridge_kwargs) as env:
        for _ in range(episodes):
            observation = env.reset()
            episode_return = 0.0
            episode_length = 0
            for _ in range(env.horizon):
                action = policy(
                    observation.reshape(1, -1), deterministic=True
                ).reshape(-1)
                observation, reward, terminated, truncated = env.step(action)
                episode_return += reward
                episode_length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
    result = summarize_episodes(returns, lengths)
    result["elapsed_seconds"] = time.time() - started
    return result


def evaluate_legacy_policy(policy, env_id, episodes, seed):
    import d4rl  # noqa: F401
    import gym

    env = gym.make(env_id).unwrapped
    if hasattr(env, "seed"):
        env.seed(seed)
    returns = []
    normalized_returns = []
    lengths = []
    started = time.time()
    try:
        for _ in range(episodes):
            raw_observation = env.reset()
            observation = flatten_legacy_observation(
                raw_observation, env.target_goal
            )
            episode_return = 0.0
            episode_length = 0
            for _ in range(env.spec.max_episode_steps):
                action = policy(
                    observation.reshape(1, -1), deterministic=True
                ).reshape(-1)
                raw_observation, reward, done, _ = env.step(action)
                observation = flatten_legacy_observation(
                    raw_observation, env.target_goal
                )
                episode_return += float(reward)
                episode_length += 1
                if done:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            normalized_returns.append(env.get_normalized_score(episode_return))
    finally:
        env.close()
    result = summarize_episodes(returns, lengths, normalized_returns)
    result["elapsed_seconds"] = time.time() - started
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(source):
    files = (
        "JaxCQL/conservative_sac.py",
        "JaxCQL/model.py",
        "JaxCQL/jax_utils.py",
        "JaxCQL/replay_buffer.py",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "path": str(source),
        "commit": commit,
        "sha256": {name: _sha256_file(source / name) for name in files},
    }


def _load_dataset(path):
    manifest_path = Path(path).with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("dataset manifest not found: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("observation_keys") != [
        "achieved_goal",
        "desired_goal",
        "observation",
    ]:
        raise ValueError("dataset manifest has an unexpected observation key order")

    with np.load(path) as archive:
        missing = [key for key in DATASET_KEYS if key not in archive]
        if missing:
            raise ValueError("dataset archive is missing keys: {}".format(missing))
        dataset = {
            key: np.asarray(archive[key], dtype=np.float32) for key in DATASET_KEYS
        }
    count = dataset["rewards"].shape[0]
    if any(value.shape[0] != count for value in dataset.values()):
        raise ValueError("dataset arrays have inconsistent transition counts")
    if dataset["observations"].shape != dataset["next_observations"].shape:
        raise ValueError("observation and next-observation shapes differ")
    for key, value in dataset.items():
        expected = manifest["arrays"][key]
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
            raise ValueError("dataset array metadata mismatch for {}".format(key))
        if _sha256_array(value) != expected["sha256"]:
            raise ValueError("dataset array hash mismatch for {}".format(key))
    if set(np.unique(dataset["rewards"]).tolist()) != {-5.0, 5.0}:
        raise ValueError("expected transformed sparse rewards {-5, 5}")
    return dataset, manifest


def _json_value(value):
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    return float(np.mean(array))


def _sha256_array(value):
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _save_checkpoint(path, sac, metadata):
    import cloudpickle

    with open(path, "wb") as target:
        cloudpickle.dump({"sac": sac, "metadata": metadata}, target)


def _load_checkpoint(path):
    import cloudpickle

    with open(path, "rb") as source:
        return cloudpickle.load(source)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path, value):
    with open(path, "a") as target:
        target.write(json.dumps(value, sort_keys=True) + "\n")


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calql-source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-python", required=True)
    parser.add_argument("--secondary-env-python")
    parser.add_argument("--env-server", required=True)
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

    source_metadata = _source_metadata(source)
    config = {
        "implementation": "official-jax-calql-with-minari-orchestration",
        "source": source_metadata,
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
    _write_json(output_dir / "config.json", config)

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

    observation_keys = ("achieved_goal", "desired_goal", "observation")
    primary_bridge = {
        "python_executable": args.env_python,
        "server_script": args.env_server,
        "dataset_id": args.dataset_id,
        "datasets_path": args.minari_datasets_path,
        "observation_keys": observation_keys,
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
        _append_jsonl(metrics_path, {**metrics, "evaluation": result})
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
                {"sac/" + key: _json_value(value) for key, value in last_metrics.items()}
            )
            run.log(train_metrics, step=step)
            _append_jsonl(metrics_path, train_metrics)

        if step % args.eval_every == 0 or step == args.offline_updates:
            evaluate_and_log(
                step, args.eval_episodes, "minari_v4_mujoco_3.11"
            )
            checkpoint = checkpoint_dir / "model_{:07d}.pkl".format(step)
            _save_checkpoint(checkpoint, sac, {**config, "grad_steps": step})

    final_checkpoint = output_dir / "model_final.pkl"
    _save_checkpoint(
        final_checkpoint, sac, {**config, "grad_steps": args.offline_updates}
    )
    loaded = _load_checkpoint(final_checkpoint)
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
    final_results["legacy_d4rl_mujoco_2_1"] = evaluate_legacy_policy(
        final_policy,
        args.legacy_env_id,
        args.final_eval_episodes,
        args.seed,
    )
    _write_json(output_dir / "final_evaluation.json", final_results)
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
