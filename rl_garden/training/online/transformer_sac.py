"""TransformerSAC run function."""

from __future__ import annotations

from rl_garden.training.online.sac import _sac_common_kwargs


def build_transformer_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import TransformerSAC
    from rl_garden.training.inspection import construct_agent

    # Deliberately no obs_groups/critic_encoder_config/encoder_sharing here --
    # ViT token_and_prop layouts are unsupported (SequenceSAC._build_policy
    # raises NotImplementedError for structured_feature_config()), and
    # asymmetric critic obs_groups were never wired for this entrypoint
    # either (encoder_sharing stays the SAC default "shared_critic_grad",
    # which SequenceSAC._build_policy's RecurrentSACPolicy requires -- it
    # rejects a non-None critic_features_extractor).
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "image_augmentation_seed": args.seed + 1_000_003,
    }

    agent = construct_agent(
        TransformerSAC,
        **_sac_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
        embed_dim=args.embed_dim,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_transformer_layers=args.num_transformer_layers,
        mlp_num=args.mlp_num,
        memory_len=args.memory_len,
        dropout_rate=args.dropout_rate,
        gru_bias=args.gru_bias,
        burn_in_len=args.burn_in_len,
        learning_len=args.learning_len,
        forward_len=args.forward_len,
        prio_exponent=args.prio_exponent,
        importance_sampling_exponent=args.importance_sampling_exponent,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_transformer_sac(args: TransformerSACArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_transformer_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import (
    VisionTransformerSACTrainingArgs,
)
from rl_garden.training.online._registry import registry


@dataclass
class TransformerSACArgs(VisionTransformerSACTrainingArgs, EnvBackendArgs):
    """TransformerSAC — GTrXL latent module + dense (not checkpoint-aligned),
    burn-in-from-zero priority replay buffer.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


registry.register(
    "transformer_sac",
    TransformerSACArgs,
    run_transformer_sac,
)
