"""TransformerPPO run function."""
from __future__ import annotations

from rl_garden.training.online.ppo import _ppo_common_kwargs, _ppo_env_request, _ppo_image_kwargs

_transformer_ppo_env_request = _ppo_env_request


def build_transformer_ppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import TransformerPPO

    image_kwargs = _ppo_image_kwargs(args, env)
    agent = TransformerPPO(
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
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_transformer_ppo_env_request,
        build_agent=build_transformer_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionTransformerPPOTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class TransformerPPOArgs(VisionTransformerPPOTrainingArgs, EnvBackendArgs):
    """TransformerPPO — GTrXL latent module between the encoder and actor/critic heads.

    Combine with any encoder via ``--encoder``, e.g. ``transformer_ppo --encoder resnet10``.
    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


registry.register("transformer_ppo", TransformerPPOArgs, run_transformer_ppo)
