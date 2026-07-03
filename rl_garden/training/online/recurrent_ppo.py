"""RecurrentPPO run function."""
from __future__ import annotations

from rl_garden.training.online.ppo import _ppo_common_kwargs, _ppo_env_request, _ppo_image_kwargs

_recurrent_ppo_env_request = _ppo_env_request


def build_recurrent_ppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import RecurrentPPO

    image_kwargs = _ppo_image_kwargs(args, env)
    agent = RecurrentPPO(
        **_ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
        rnn_type=args.rnn_type,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_recurrent_ppo(args: RecurrentPPOArgs) -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_recurrent_ppo_env_request,
        build_agent=build_recurrent_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionRecurrentPPOTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class RecurrentPPOArgs(VisionRecurrentPPOTrainingArgs, EnvBackendArgs):
    """RecurrentPPO — LSTM/GRU latent module between the encoder and actor/critic heads.

    Combine with any encoder via ``--encoder``, e.g. ``recurrent_ppo --encoder resnet10``.
    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


registry.register("recurrent_ppo", RecurrentPPOArgs, run_recurrent_ppo)
