"""HIL-SERL run function.

Targets HIL-SERL's ``train_rlpd.py`` capability set (online RLPD + demo
mixing + human intervention + reward classifier), not BC or HG-DAgger's
growing-dataset problem -- see
docs/superpowers/specs/2026-07-09-hil-serl-migration-design.md.
"""
from __future__ import annotations

import dataclasses


def _hil_serl_env_request(args, run_name):
    from rl_garden.training.online.rlpd import _rlpd_env_request

    return _rlpd_env_request(args, run_name)


def _build_env(args, env_request):
    """Composes the env wrappers built in the SERL v1 round but never wired
    up until now: reward classifier (inner) -> human intervention (middle)
    -> optional reset-free forward/backward (outer, per
    ``FWBWActorLoop``/``FWBWResetFreeWrapper``'s documented requirement that
    it sit outermost)."""
    from rl_garden.envs.backend_registry import make_training_envs
    from rl_garden.envs.wrappers.fwbw_reset_free import FWBWResetFreeWrapper
    from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
    from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

    env, _ = make_training_envs(args.env_backend, env_request)

    if args.classifier_checkpoint is not None:
        from rl_garden.common.utils import get_device
        from rl_garden.models.reward.success.model import load_classifier_fn

        # FrankaRealEnv's observation space is {"state": ..., <camera_key>: image, ...}
        # (see rl_garden/envs/franka_real/env.py) -- camera keys are whatever
        # the user configured (cfg.camera_keys), not necessarily "rgb*"/"depth*"
        # prefixed, so every non-"state" key is an image key here.
        image_keys = args.classifier_image_keys or tuple(
            k for k in env.single_observation_space.spaces if k != "state"
        )
        classifier_fn = load_classifier_fn(
            args.classifier_checkpoint,
            env.single_observation_space,
            image_keys,
            device=get_device("auto"),
        )
        env = RewardClassifierWrapper(env, classifier_fn, threshold=args.classifier_threshold)

    env = TeleopInterventionWrapper(env, device=args.teleop_device)

    if args.fwbw:
        env = FWBWResetFreeWrapper(env)

    return env


def _run_actor(args) -> None:
    """Non-FWBW: one scratch agent, one ``HilSerlActorLoop`` -- same shape as
    ``serl.py``'s ``_run_actor``. FWBW: two scratch agents (forward/backward)
    and two sync clients, driving the existing ``FWBWActorLoop`` (a single
    actor process still only ever drives the one physical robot; it's the
    *policy/sync-client pair* that's doubled, switched per step based on
    ``info["fwbw_direction"]``)."""
    from rl_garden.real_world import FWBWActorLoop
    from rl_garden.real_world.hil_serl import HilSerlActorLoop, HilSerlActorSyncClient

    env_request = _hil_serl_env_request(args, run_name="real_world_actor")
    env = _build_env(args, env_request)

    if not args.fwbw:
        agent = _rebuild_scratch_agent_for_env(args, env)
        sync_client = HilSerlActorSyncClient(f"http://{args.sync_host}:{args.sync_port}")
        loop = HilSerlActorLoop(
            env,
            agent.policy,
            sync_client,
            control_hz=args.control_hz,
            device=agent.device,
            deterministic=args.deterministic_actor,
            seed=args.seed,
        )
        loop.run()
        return

    backward_port = args.sync_port_backward or (args.sync_port + 1)
    agents = {
        "forward": _rebuild_scratch_agent_for_env(args, env),
        "backward": _rebuild_scratch_agent_for_env(args, env),
    }
    sync_clients = {
        "forward": HilSerlActorSyncClient(f"http://{args.sync_host}:{args.sync_port}"),
        "backward": HilSerlActorSyncClient(f"http://{args.sync_host}:{backward_port}"),
    }
    loop = FWBWActorLoop(
        env,
        {k: a.policy for k, a in agents.items()},
        sync_clients,
        control_hz=args.control_hz,
        device=agents["forward"].device,
        deterministic=args.deterministic_actor,
        seed=args.seed,
    )
    loop.run()


def _rebuild_scratch_agent_for_env(args, env):
    from rl_garden.training.online.rlpd_hybrid import build_rlpd_hybrid

    scratch_args = dataclasses.replace(
        args, buffer_size=8, load_checkpoint=None, offline_dataset_path=None
    )
    return build_rlpd_hybrid(scratch_args, env, None, logger=None, checkpoint_dir=None)


def _run_learner(args) -> None:
    """Builds a full ``RLPDHybrid`` instance and drives
    ``rl_garden.real_world.hil_serl.HilSerlLearnerLoop``. In FWBW mode, a
    learner process is launched once per direction (``--fwbw_direction
    forward|backward``), each bound to its own sync port -- matching the
    SERL v1 design's "two full actor/learner pairs" for reset-free
    training. Non-FWBW: identical shape to ``serl.py``'s ``_run_learner``."""
    import os
    import time

    from rl_garden.common import Logger, seed_everything
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden.real_world.hil_serl import HilSerlLearnerLoop
    from rl_garden.training.online.rlpd_hybrid import build_rlpd_hybrid

    seed_everything(args.seed)

    direction_suffix = f"_{args.fwbw_direction}" if args.fwbw else ""
    port = (
        (args.sync_port_backward or (args.sync_port + 1))
        if args.fwbw and args.fwbw_direction == "backward"
        else args.sync_port
    )

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name or f"{args.env_id}__hil_serl_real_world{direction_suffix}__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.log_dir, run_name, "checkpoints")

    resolved_config = persist_resolved_config(
        args, training_phase="real_world", algorithm="hil_serl", run_name=run_name, log_dir=args.log_dir
    )
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=resolved_config,
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group or args.env_id,
    )

    env_request = _hil_serl_env_request(args, run_name=run_name)
    env = _build_env(args, env_request)

    agent = build_rlpd_hybrid(args, env, None, logger=logger, checkpoint_dir=checkpoint_dir)
    agent.init_demo_buffer(args.demo_buffer_size, args.demo_data_ratio)

    loop = HilSerlLearnerLoop(
        agent,
        host=args.sync_host,
        port=port,
        checkpoint_dir=checkpoint_dir,
        buffer_period=args.buffer_period,
        train_freq=args.train_freq,
        publish_freq=args.publish_freq,
    )
    try:
        loop.run()
    finally:
        logger.close()


def run_hil_serl(args: "HilSerlArgs") -> None:
    if args.role == "actor":
        _run_actor(args)
    else:
        _run_learner(args)


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402
from typing import Literal, Optional, Sequence  # noqa: E402

from rl_garden.training.online.rlpd_hybrid import RLPDHybridArgs  # noqa: E402
from rl_garden.training.real_world._args import (  # noqa: E402
    FrankaRealConfig,
    RealWorldFrankaArgs,
)
from rl_garden.training.real_world._registry import registry  # noqa: E402


@dataclass
class HilSerlArgs(RealWorldFrankaArgs, RLPDHybridArgs):
    franka_real: FrankaRealConfig = field(default_factory=FrankaRealConfig)

    classifier_checkpoint: Optional[str] = None
    classifier_threshold: float = 0.5
    classifier_image_keys: Sequence[str] = field(default_factory=tuple)

    teleop_device: Literal["pico", "spacemouse"] = "pico"

    # Growing human-intervention demo buffer (DemoInterventionMixin), mixed
    # into training via RLPD's existing offline_replay_buffer/
    # offline_data_ratio slot -- see rl_garden/buffers/demo_intervention.py.
    # Not supported together with --offline_dataset_path (RLPD's static prior
    # dataset) in this round; both use the same slot.
    demo_buffer_size: int = 100_000
    demo_data_ratio: float = 0.5
    # Crash-recovery pkl snapshot cadence (received transitions between
    # snapshots), mirroring HIL-SERL's own buffer_period.
    buffer_period: int = 1000

    fwbw: bool = False
    # Only used when fwbw=True. Actor: builds both directions' sync clients
    # (forward on sync_port, backward on sync_port_backward). Learner: which
    # direction this particular learner process owns.
    sync_port_backward: Optional[int] = None
    fwbw_direction: Literal["forward", "backward"] = "forward"


registry.register("hil_serl", HilSerlArgs, run_hil_serl)
