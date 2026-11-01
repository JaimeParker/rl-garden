"""RecurrentPPO run function."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rl_garden.training.online.entrypoints.ppo import (
    ppo_common_kwargs,
    ppo_observation_kwargs,
)


def build_recurrent_ppo(
    args,
    env,
    eval_env,
    logger,
    checkpoint_dir,
    *,
    algorithm_cls: type | None = None,
    **extra_kwargs: Any,
):
    """Build ``algorithm_cls`` (``RecurrentPPO`` when ``None``) from ``args``;
    ``extra_kwargs`` go to its constructor next to the PPO and rnn kwargs."""
    from rl_garden.training.inspection import construct_agent

    if algorithm_cls is None:
        from rl_garden.algorithms import RecurrentPPO

        algorithm_cls = RecurrentPPO
    image_kwargs = ppo_observation_kwargs(args)
    agent = construct_agent(
        algorithm_cls,
        **ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
        rnn_type=args.rnn_type,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        **extra_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_recurrent_ppo(
    args: RecurrentPPOArgs, *, build_agent: Callable | None = None
) -> None:
    """Train with ``build_agent`` (``build_recurrent_ppo`` when ``None``)."""
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_recurrent_ppo if build_agent is None else build_agent,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online.args import VisionRecurrentPPOTrainingArgs
from rl_garden.training.online.registry import registry


@dataclass
class RecurrentPPOArgs(VisionRecurrentPPOTrainingArgs, EnvBackendArgs):
    """RecurrentPPO — LSTM/GRU latent module between the encoder and actor/critic heads.

    Combine with any encoder via ``--encoder.backbone``, e.g.
    ``recurrent_ppo --obs.rgb base_camera --encoder.backbone resnet10``.
    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


def _recurrent_ppo_algorithm_cls() -> type:
    from rl_garden.algorithms import RecurrentPPO

    return RecurrentPPO


registry.register(
    "recurrent_ppo",
    RecurrentPPOArgs,
    run_recurrent_ppo,
    algorithm_cls=_recurrent_ppo_algorithm_cls,
    build_agent=build_recurrent_ppo,
)
