"""TransformerPPO run function."""

from __future__ import annotations

from rl_garden.training.online.ppo import _ppo_common_kwargs, _ppo_observation_kwargs


def build_transformer_ppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import TransformerPPO
    from rl_garden.training.inspection import construct_agent

    image_kwargs = _ppo_observation_kwargs(args)
    agent = construct_agent(
        TransformerPPO,
        **_ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
        embed_dim=args.embed_dim,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_transformer_layers=args.num_transformer_layers,
        mlp_num=args.mlp_num,
        memory_len=args.memory_len,
        dropout_rate=args.dropout_rate,
        gru_bias=args.gru_bias,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_transformer_ppo(args: TransformerPPOArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_transformer_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import (
    VisionTransformerPPOTrainingArgs,
)
from rl_garden.training.online._registry import registry


@dataclass
class TransformerPPOArgs(VisionTransformerPPOTrainingArgs, EnvBackendArgs):
    """TransformerPPO — GTrXL latent module between the encoder and actor/critic heads.

    Combine with any encoder via ``--encoder.backbone``, e.g.
    ``transformer_ppo --obs.rgb base_camera --encoder.backbone resnet10``.
    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


def _transformer_ppo_algorithm_cls() -> type:
    from rl_garden.algorithms import TransformerPPO

    return TransformerPPO


registry.register(
    "transformer_ppo",
    TransformerPPOArgs,
    run_transformer_ppo,
    algorithm_cls=_transformer_ppo_algorithm_cls,
)
