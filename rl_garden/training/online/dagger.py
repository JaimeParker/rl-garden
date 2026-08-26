"""DAgger run function.

State-only CLI entrypoint (matches ``DAggerTrainingArgs``'s scope). The
``--expert`` flag only selects among trivial built-in mock experts -- a real
scripted/oracle expert is task-specific and is expected to be supplied by
constructing ``DAgger`` directly rather than through this CLI path; see
``.agents/local/imitation-learning-expansion-dagger-notes.md``.
"""

from __future__ import annotations

import torch


class _ZeroExpert:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], self.action_dim, device=obs.device)


class _RandomUniformExpert:
    """Samples uniformly within the action space bounds every call --
    obviously not a real expert, but exercises the full DAgger rollout/label
    path end-to-end for smoke-testing the CLI entrypoint."""

    def __init__(self, low: torch.Tensor, high: torch.Tensor) -> None:
        self.low = low
        self.high = high

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        noise = torch.rand(batch, self.low.shape[0], device=obs.device)
        return self.low + noise * (self.high - self.low)


def _mock_expert_from_args(args, action_space):
    if args.expert == "zero":
        return _ZeroExpert(action_dim=action_space.shape[0])
    if args.expert == "random_uniform":
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        return _RandomUniformExpert(low, high)
    raise ValueError(f"Unknown --expert: {args.expert!r}")


def _dagger_env_request(args, run_name):
    del run_name
    from rl_garden.envs.backend_registry import EnvRequest

    backend_config = args.resolve_backend_config()
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=None,
        camera_height=None,
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=None,
        capture_video=False,
        create_eval_env=False,
        backend_config=backend_config,
    )


def build_dagger(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.dagger import DAgger
    from rl_garden.training.inspection import construct_agent

    expert = _mock_expert_from_args(args, env.single_action_space)
    return construct_agent(
        DAgger,
        env=env,
        expert=expert,
        eval_env=eval_env,
        demo_buffer_size=args.demo_buffer_size,
        beta_rounds=args.beta_rounds,
        rollout_steps_per_round=args.rollout_steps_per_round,
        gradient_steps_per_round=args.gradient_steps_per_round,
        buffer_device=args.buffer_device,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        weight_decay=args.weight_decay,
        tanh_squash=args.tanh_squash,
        net_arch=list(args.net_arch),
        seed=args.seed,
        device=args.device,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
    )


def run_dagger(args: "DAggerArgs") -> None:
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        make_env_request=_dagger_env_request,
        build_agent=build_dagger,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import DAggerTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class DAggerArgs(DAggerTrainingArgs, EnvBackendArgs):
    """DAgger with multi-env backend support (state-only)."""


registry.register("dagger", DAggerArgs, run_dagger)
