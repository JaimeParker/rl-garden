"""RecurrentSAC run function."""

from __future__ import annotations

from rl_garden.training.online.sac import _sac_common_kwargs


def build_recurrent_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import RecurrentSAC
    from rl_garden.training.inspection import construct_agent

    # Deliberately no obs_groups/critic_encoder_config/encoder_sharing here:
    # ViT token_and_prop layouts are unsupported (RecurrentSAC._build_policy
    # raises NotImplementedError for structured_feature_config()), and
    # asymmetric critic obs_groups were never wired for this entrypoint
    # either (encoder_sharing stays the SAC default "shared_critic_grad",
    # which SequenceSAC._build_policy's RecurrentSACPolicy requires -- it
    # rejects a non-None critic_extractor).
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "image_augmentation_seed": args.seed + 1_000_003,
    }

    agent = construct_agent(
        RecurrentSAC,
        **_sac_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
        rnn_type=args.rnn_type,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        burn_in_len=args.burn_in_len,
        learning_len=args.learning_len,
        forward_len=args.forward_len,
        prio_exponent=args.prio_exponent,
        importance_sampling_exponent=args.importance_sampling_exponent,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_recurrent_sac(args: RecurrentSACArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_recurrent_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import VisionRecurrentSACTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class RecurrentSACArgs(VisionRecurrentSACTrainingArgs, EnvBackendArgs):
    """RecurrentSAC — LSTM/GRU latent module + R2D2-style (stored hidden state
    + burn-in + n-step + priority replay) recurrent buffer.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """




def _recurrent_sac_algorithm_cls() -> type:
    from rl_garden.algorithms import RecurrentSAC

    return RecurrentSAC

registry.register(
    "recurrent_sac",
    RecurrentSACArgs,
    run_recurrent_sac,
    algorithm_cls=_recurrent_sac_algorithm_cls)
