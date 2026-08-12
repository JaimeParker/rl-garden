"""Probe Q contrast for patched official JAX IQL checkpoints.

This mirrors the JSON shape of ``probe_q_contrast_local_noise.py`` for the
official IQL msgpack checkpoints saved under ``out/models/<step>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Args:
    env_name: str
    model_dir: str
    output_json: str | None = None
    iql_source_dir: str = "3rd_party/implicit_q_learning"

    max_transitions: int = 4096
    batch_size: int = 256
    num_noisy_actions: int = 64
    num_actor_candidates: int = 64
    num_replay_neighbors: int = 64
    radii: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
    boltzmann_temperatures: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    seed: int = 20260812

    actor_lr: float = 1e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.9
    temperature: float = 10.0
    dropout_rate: float | None = None
    opt_decay_schedule: str = "none"


@dataclass
class DemoBatch:
    observations: np.ndarray
    actions: np.ndarray
    num_available_transitions: int
    num_selected_transitions: int
    num_total_episodes: int


def _suffix(multiplier: float) -> str:
    return str(multiplier).replace(".", "p")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _summarize_samples(samples: dict[str, list[np.ndarray]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, parts in samples.items():
        values = np.concatenate([np.asarray(part).reshape(-1) for part in parts])
        stats = _summary(values)
        out[f"{key}_mean"] = stats["mean"]
        if key in {
            "abs_dq",
            "q_drop",
            "local_lipschitz",
            "dq_over_denominator",
            "action_grad_norm",
            "q_best_minus_actor",
            "actor_to_best_dist",
        }:
            out[f"{key}_p50"] = stats["p50"]
            out[f"{key}_p90"] = stats["p90"]
        if key == "action_grad_norm":
            out[f"{key}_p99"] = float(np.quantile(values, 0.99))
        if key == "local_ess" or key.startswith("ess_x"):
            out[f"{key}_p10"] = stats["p10"]
        if key == "max_weight" or key.startswith("max_weight_x"):
            out[f"{key}_p90"] = stats["p90"]
    return out


def _add_iql_source(path: str) -> None:
    source = str(Path(path).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def load_d4rl_batch(env_name: str, *, max_transitions: int, seed: int) -> DemoBatch:
    import d4rl  # noqa: F401
    import gym

    rng = np.random.default_rng(seed)
    env = gym.make(env_name)
    try:
        raw = env.get_dataset()
        observations = np.asarray(raw["observations"], dtype=np.float32)
        actions = np.asarray(raw["actions"], dtype=np.float32)
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "D4RL observations/actions length mismatch: "
                f"{observations.shape[0]} vs {actions.shape[0]}."
            )
        num_available = int(actions.shape[0])
        indices = np.arange(num_available)
        if max_transitions > 0 and num_available > max_transitions:
            indices = rng.choice(num_available, size=max_transitions, replace=False)
        terminals = np.asarray(raw.get("terminals", np.zeros(num_available)), dtype=bool)
        timeouts = np.asarray(raw.get("timeouts", np.zeros(num_available)), dtype=bool)
    finally:
        env.close()
    return DemoBatch(
        observations=observations[indices].astype(np.float32),
        actions=actions[indices].astype(np.float32),
        num_available_transitions=num_available,
        num_selected_transitions=int(indices.shape[0]),
        num_total_episodes=max(int(np.count_nonzero(terminals | timeouts)), 1),
    )


def load_learner(args: Args, obs_sample: np.ndarray, action_sample: np.ndarray) -> Any:
    _add_iql_source(args.iql_source_dir)
    from learner import Learner

    kwargs = {
        "actor_lr": args.actor_lr,
        "value_lr": args.value_lr,
        "critic_lr": args.critic_lr,
        "hidden_dims": args.hidden_dims,
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "temperature": args.temperature,
        "dropout_rate": args.dropout_rate,
        "opt_decay_schedule": args.opt_decay_schedule,
        "max_steps": None,
    }
    learner = Learner(args.seed, obs_sample[None], action_sample[None], **kwargs)
    model_dir = Path(args.model_dir)
    learner.actor = learner.actor.load(str(model_dir / "actor.msgpack"))
    learner.critic = learner.critic.load(str(model_dir / "critic.msgpack"))
    learner.target_critic = learner.target_critic.load(
        str(model_dir / "target_critic.msgpack")
    )
    learner.value = learner.value.load(str(model_dir / "value.msgpack"))
    return learner


def _q_values(learner: Any, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    q1, q2 = learner.critic(observations, actions)
    return np.asarray(np.minimum(q1, q2), dtype=np.float32)


def _actor_det_actions(learner: Any, observations: np.ndarray) -> np.ndarray:
    dist = learner.actor(observations)
    return np.asarray(dist.mean(), dtype=np.float32)


def _actor_sample_actions(
    learner: Any,
    observations: np.ndarray,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    import jax

    dist = learner.actor(observations)
    samples = dist.sample(seed=jax.random.PRNGKey(seed), sample_shape=(n,))
    return np.asarray(samples, dtype=np.float32).transpose(1, 0, 2)


def _action_grad_norm(learner: Any, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    def one_grad(obs, action):
        def q_action(a):
            q1, q2 = learner.critic(obs[None], a[None])
            return jnp.minimum(q1, q2).sum()

        return jnp.linalg.norm(jax.grad(q_action)(action))

    return np.asarray(jax.vmap(one_grad)(observations, actions), dtype=np.float32)


def fixed_radius_noise(
    actions: np.ndarray,
    *,
    radius: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = rng.normal(size=(actions.shape[0], n, actions.shape[1])).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-12)
    proposed = actions[:, None, :] + radius * directions
    clipped = np.clip(proposed, -1.0, 1.0)
    delta = clipped - actions[:, None, :]
    effective_radius = np.linalg.norm(delta, axis=-1)
    clip_fraction = np.mean(np.abs(proposed - clipped) > 1e-7, axis=-1)
    return clipped.astype(np.float32), effective_radius.astype(np.float32), clip_fraction.astype(np.float32)


def local_q_contrast_samples(
    learner: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    radius: float,
    num_noisy_actions: int,
    denominator: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    noisy, effective_radius, clip_fraction = fixed_radius_noise(
        actions, radius=radius, n=num_noisy_actions, rng=rng
    )
    flat_obs = np.repeat(observations, num_noisy_actions, axis=0)
    flat_actions = noisy.reshape(-1, actions.shape[-1])
    q_anchor = _q_values(learner, observations, actions)
    q_noisy = _q_values(learner, flat_obs, flat_actions).reshape(
        actions.shape[0], num_noisy_actions
    )
    dq = q_noisy - q_anchor[:, None]
    logits = np.concatenate([q_anchor[:, None], q_noisy], axis=1) / max(
        denominator, 1e-12
    )
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return {
        "q_anchor": q_anchor,
        "q_noisy_mean": q_noisy.mean(axis=1),
        "abs_dq": np.abs(dq).mean(axis=1),
        "q_drop": np.maximum(q_anchor[:, None] - q_noisy, 0.0).mean(axis=1),
        "local_lipschitz": (np.abs(dq) / np.maximum(effective_radius, 1e-12)).mean(axis=1),
        "dq_over_denominator": (np.abs(dq) / max(denominator, 1e-12)).mean(axis=1),
        "local_ess": 1.0 / np.maximum(np.square(weights).sum(axis=1), 1e-12),
        "local_entropy": -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1),
        "max_weight": weights.max(axis=1),
        "anchor_top1": (q_anchor >= q_noisy.max(axis=1)).astype(np.float32),
        "effective_radius": effective_radius.mean(axis=1),
        "clip_fraction": clip_fraction.mean(axis=1),
        "action_grad_norm": _action_grad_norm(learner, observations, actions),
    }


def replay_neighbor_actions(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    start: int,
    stop: int,
    n: int,
) -> np.ndarray:
    query = observations[start:stop]
    distances = np.sqrt(((query[:, None, :] - observations[None, :, :]) ** 2).sum(axis=-1))
    k = min(max(int(n), 1), observations.shape[0])
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    row = np.arange(indices.shape[0])[:, None]
    order = np.argsort(distances[row, indices], axis=1)
    return actions[indices[row, order]]


def candidate_q_contrast_samples(
    learner: Any,
    observations: np.ndarray,
    actor_actions: np.ndarray,
    candidate_actions: np.ndarray,
    *,
    denominator: float,
    temperature_multipliers: tuple[float, ...],
) -> dict[str, np.ndarray]:
    batch, num_candidates, action_dim = candidate_actions.shape
    flat_obs = np.repeat(observations, num_candidates, axis=0)
    q_candidates = _q_values(
        learner, flat_obs, candidate_actions.reshape(batch * num_candidates, action_dim)
    ).reshape(batch, num_candidates)
    q_actor = _q_values(learner, observations, actor_actions)
    best_index = q_candidates.argmax(axis=1)
    best_actions = candidate_actions[np.arange(batch), best_index]
    best_q = q_candidates[np.arange(batch), best_index]
    out = {
        "q_actor": q_actor,
        "q_candidate_mean": q_candidates.mean(axis=1),
        "q_candidate_max": best_q,
        "q_best_minus_actor": best_q - q_actor,
        "actor_to_best_dist": np.linalg.norm(actor_actions - best_actions, axis=-1),
        "actor_top1": (q_actor >= best_q).astype(np.float32),
    }
    for multiplier in temperature_multipliers:
        temperature = max(float(denominator) * float(multiplier), 1e-12)
        logits = q_candidates / temperature
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        out[f"ess_x{_suffix(multiplier)}"] = 1.0 / np.maximum(
            np.square(weights).sum(axis=1), 1e-12
        )
        out[f"max_weight_x{_suffix(multiplier)}"] = weights.max(axis=1)
    return out


def probe(learner: Any, batch: DemoBatch, args: Args) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + 17)
    results: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    denom = 1.0 / float(args.temperature)
    observations = batch.observations
    actions = batch.actions

    for anchor_group in ("dataset_action", "actor_det_action"):
        acc_by_radius: dict[float, dict[str, list[np.ndarray]]] = {
            radius: {} for radius in args.radii
        }
        for start in range(0, actions.shape[0], args.batch_size):
            stop = min(start + args.batch_size, actions.shape[0])
            obs_chunk = observations[start:stop]
            anchor = actions[start:stop] if anchor_group == "dataset_action" else _actor_det_actions(learner, obs_chunk)
            for radius in args.radii:
                samples = local_q_contrast_samples(
                    learner,
                    obs_chunk,
                    anchor,
                    radius=radius,
                    num_noisy_actions=args.num_noisy_actions,
                    denominator=denom,
                    rng=rng,
                )
                acc = acc_by_radius[radius]
                for key, value in samples.items():
                    acc.setdefault(key, []).append(value)
        for radius, acc in acc_by_radius.items():
            results.append(
                {
                    "anchor_group": anchor_group,
                    "radius": radius,
                    "num_states": batch.num_selected_transitions,
                    "num_noisy_actions": args.num_noisy_actions,
                    "action_dim": int(actions.shape[-1]),
                    "denominator_name": "inverse_iql_temperature",
                    "denominator_value": denom,
                    **_summarize_samples(acc),
                }
            )

    for candidate_group in ("actor_stochastic", "replay_neighbor"):
        acc: dict[str, list[np.ndarray]] = {}
        for start in range(0, actions.shape[0], args.batch_size):
            stop = min(start + args.batch_size, actions.shape[0])
            obs_chunk = observations[start:stop]
            actor_det = _actor_det_actions(learner, obs_chunk)
            if candidate_group == "actor_stochastic":
                candidate_actions = _actor_sample_actions(
                    learner,
                    obs_chunk,
                    n=args.num_actor_candidates,
                    seed=args.seed + start + 101,
                )
                num_candidates = args.num_actor_candidates
            else:
                candidate_actions = replay_neighbor_actions(
                    observations,
                    actions,
                    start=start,
                    stop=stop,
                    n=args.num_replay_neighbors,
                )
                num_candidates = min(args.num_replay_neighbors, actions.shape[0])
            samples = candidate_q_contrast_samples(
                learner,
                obs_chunk,
                actor_det,
                candidate_actions,
                denominator=denom,
                temperature_multipliers=args.boltzmann_temperatures,
            )
            for key, value in samples.items():
                acc.setdefault(key, []).append(value)
        candidate_results.append(
            {
                "candidate_group": candidate_group,
                "num_states": batch.num_selected_transitions,
                "num_candidate_actions": int(num_candidates),
                "action_dim": int(actions.shape[-1]),
                "denominator_name": "inverse_iql_temperature",
                "denominator_value": denom,
                "temperature_multipliers": list(args.boltzmann_temperatures),
                **_summarize_samples(acc),
            }
        )

    return {
        "algorithm": "iql-jax",
        "env_name": args.env_name,
        "model_dir": args.model_dir,
        "num_available_transitions": batch.num_available_transitions,
        "num_selected_transitions": batch.num_selected_transitions,
        "num_total_episodes": batch.num_total_episodes,
        "results": results,
        "candidate_results": candidate_results,
        "unsupported_metrics": [],
    }


def validate_args(args: Args) -> None:
    if args.max_transitions == 0 or args.max_transitions < -1:
        raise ValueError("--max-transitions must be positive or -1 for all transitions.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_noisy_actions <= 0:
        raise ValueError("--num-noisy-actions must be positive.")
    if args.num_actor_candidates <= 0:
        raise ValueError("--num-actor-candidates must be positive.")
    if args.num_replay_neighbors <= 0:
        raise ValueError("--num-replay-neighbors must be positive.")
    if not args.radii or any(radius <= 0 for radius in args.radii):
        raise ValueError("--radii must contain only positive values.")
    if not args.boltzmann_temperatures or any(
        temperature <= 0 for temperature in args.boltzmann_temperatures
    ):
        raise ValueError("--boltzmann-temperatures must contain only positive values.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not args.hidden_dims or any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError("--hidden-dims must contain only positive values.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--iql-source-dir", default=Args.iql_source_dir)
    parser.add_argument("--max-transitions", type=int, default=Args.max_transitions)
    parser.add_argument("--batch-size", type=int, default=Args.batch_size)
    parser.add_argument("--num-noisy-actions", type=int, default=Args.num_noisy_actions)
    parser.add_argument("--num-actor-candidates", type=int, default=Args.num_actor_candidates)
    parser.add_argument("--num-replay-neighbors", type=int, default=Args.num_replay_neighbors)
    parser.add_argument("--radii", type=float, nargs="+", default=Args.radii)
    parser.add_argument(
        "--boltzmann-temperatures",
        type=float,
        nargs="+",
        default=Args.boltzmann_temperatures,
    )
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--actor-lr", type=float, default=Args.actor_lr)
    parser.add_argument("--value-lr", type=float, default=Args.value_lr)
    parser.add_argument("--critic-lr", type=float, default=Args.critic_lr)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=Args.hidden_dims)
    parser.add_argument("--discount", type=float, default=Args.discount)
    parser.add_argument("--tau", type=float, default=Args.tau)
    parser.add_argument("--expectile", type=float, default=Args.expectile)
    parser.add_argument("--temperature", type=float, default=Args.temperature)
    parser.add_argument("--dropout-rate", type=float)
    parser.add_argument("--opt-decay-schedule", default=Args.opt_decay_schedule)
    ns = parser.parse_args()
    args = Args(
        env_name=ns.env_name,
        model_dir=ns.model_dir,
        output_json=ns.output_json,
        iql_source_dir=ns.iql_source_dir,
        max_transitions=ns.max_transitions,
        batch_size=ns.batch_size,
        num_noisy_actions=ns.num_noisy_actions,
        num_actor_candidates=ns.num_actor_candidates,
        num_replay_neighbors=ns.num_replay_neighbors,
        radii=tuple(ns.radii),
        boltzmann_temperatures=tuple(ns.boltzmann_temperatures),
        seed=ns.seed,
        actor_lr=ns.actor_lr,
        value_lr=ns.value_lr,
        critic_lr=ns.critic_lr,
        hidden_dims=tuple(ns.hidden_dims),
        discount=ns.discount,
        tau=ns.tau,
        expectile=ns.expectile,
        temperature=ns.temperature,
        dropout_rate=ns.dropout_rate,
        opt_decay_schedule=ns.opt_decay_schedule,
    )
    validate_args(args)
    batch = load_d4rl_batch(
        args.env_name,
        max_transitions=args.max_transitions,
        seed=args.seed,
    )
    learner = load_learner(args, batch.observations[0], batch.actions[0])
    payload = {"args": asdict(args), **probe(learner, batch, args)}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
