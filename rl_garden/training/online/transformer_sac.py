"""TransformerSAC run function."""
from __future__ import annotations

from rl_garden.training.online.sac import _sac_common_kwargs, _sac_env_request

_transformer_sac_env_request = _sac_env_request


def build_transformer_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import TransformerSAC
    from rl_garden.common.cli_args import image_encoder_factory_from_args, image_keys_from_env

    is_visual = args.obs_mode != "state"
    image_kwargs: dict = {}
    if is_visual:
        image_kwargs = dict(
            image_keys=image_keys_from_env(env, args),
            image_encoder_factory=image_encoder_factory_from_args(args),
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=args.frame_stack > 1,
            image_augmentation=args.image_augmentation,
            random_shift_pad=args.image_random_shift_pad,
            image_augmentation_seed=args.seed + 1_000_003,
            # Deliberately omit vit_sac_kwargs_from_args(...) here -- ViT
            # token_and_prop layouts are unsupported (SequenceSAC._build_policy
            # raises NotImplementedError for structured_feature_config()).
        )

    agent = TransformerSAC(
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


def run_transformer_sac(args: "TransformerSACArgs") -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_transformer_sac_env_request,
        build_agent=build_transformer_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionTransformerSACTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class TransformerSACArgs(VisionTransformerSACTrainingArgs, EnvBackendArgs):
    """TransformerSAC — GTrXL latent module + dense (not checkpoint-aligned),
    burn-in-from-zero priority replay buffer.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


registry.register("transformer_sac", TransformerSACArgs, run_transformer_sac)
