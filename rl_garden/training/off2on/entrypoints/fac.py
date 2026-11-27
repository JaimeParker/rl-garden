"""FAC (Flow Actor-Critic for Offline RL, ICLR 2026, arXiv 2602.18015)
offline-to-online run function.

Reuses the shared ``run_off2on`` runner (``rl_garden/training/off2on/runner.py``)
unmodified, matching ``fino.py``'s shape: only a ``build_fac`` callback is
needed here. Offline pretraining (BC-flow prelude -> dataset-logp cache ->
conservative actor-critic updates) is unchanged ``FACCore.train()``; the
online switch is handled entirely inside ``Off2OnFAC``/``FACCore`` (see
``rl_garden/algorithms/fac.py``'s module docstring).

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. Mirrors ``FINOOff2OnArgs``: fields inlined
against ``FQLTrainingArgs``, ``ObservationArgs``, ``EnvBackendArgs`` rather
than adding a new class to ``off2on/_args.py`` -- ``Off2OnFAC`` has no
critic-flow/rejection-sampling machinery, so FINO's fields (``noise_scale``,
``beta``, ``num_samples``) are dropped and FAC's BC-pretrain/conservative-
penalty fields (``fac_alpha``, ``fac_lambda``, ``fac_threshold``,
``logp_method``, ``logp_hutch_probes``, ``weight_type``, ``bc_lr``,
``bc_batch_size``, ``bc_pretrain_epochs``, ``bc_pretrain_steps``) are added
instead, matching ``rl_garden/training/offline/args.py``'s ``OfflineFACArgs``
defaults (which extend ``OfflineFQLArgs``' with ``q_agg="min"``,
``normalize_q_loss=True``).

Group-norm options reach the constructor; the flow actor does not consume
``actor_use_group_norm``. ``critic_use_group_norm`` and ``num_groups``
configure the critic.
``warmup_steps`` maps to ``initial_training_phase`` (default 0 means no
warmup). Inherited ``critic_subsample_size``, ``actor_dropout_rate`` and
``std_parameterization`` have no corresponding FQL constructor options and
remain unused.

The reference's ``balanced_sampling`` (``main.py:212-216``, half offline
dataset / half replay buffer, every online step) maps to this repo's existing
``--online_replay_mode mixed --offline_data_ratio 0.5`` combination rather
than a new mechanism, and is opt-in via those two flags. The default,
``online_replay_mode="append"``, is the reference's: online transitions join
the dataset in one buffer (``main.py:90-93,218``).
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.fql import FACParams


def build_fac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnFAC
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.off2on.args import initial_training_phase_from_args

    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    learning_kwargs = params_kwargs(args, Off2OnFAC.params_class)
    learning_kwargs["net_arch"] = list(args.net_arch) if args.net_arch is not None else [args.hidden_dim] * args.hidden_layers

    agent = construct_agent(
        Off2OnFAC,
        **learning_kwargs,
        env=env,
        eval_env=eval_env,
        **image_kwargs,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
        alpha=args.alpha,
        actor_use_group_norm=args.actor_use_group_norm,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.online_eval_freq or 0,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_fac(args: "FACOff2OnArgs") -> None:
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_fac, algorithm="fac")


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.off2on.args import FQLTrainingArgs  # noqa: E402
from rl_garden.training.off2on.registry import registry  # noqa: E402


@dataclass
class FACOff2OnArgs(FACParams, FQLTrainingArgs, ObservationArgs, EnvBackendArgs):
    """FAC -- offline-to-online flow actor-critic with a BC-pretrain ->
    logp-cache -> conservative-actor-critic offline schedule, then online
    fine-tuning matching the reference's ``online_finetuning=True`` branch
    (Chae et al., ICLR 2026, arXiv:2602.18015, ``3rd_party/FAC/agents/fac.py``).
    State-only observations by default; pass ``--obs.rgb <camera>`` for
    CNN-based Dict/RGBD observations.
    """

    normalize_q_loss: bool = True
    q_agg: Literal['mean', 'min'] = 'min'
    online_replay_mode: Literal["empty", "append", "mixed"] = "append"
    warmup_steps: int = 0
    gamma: float = 0.995
    alpha: float = 10.0  # unused by FAC; kept only for FQLCore-signature compatibility
    hidden_dim: int = 512
    hidden_layers: int = 4


def _off2_on_fac_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnFAC

    return Off2OnFAC


registry.register(
    "fac",
    FACOff2OnArgs,
    run_fac,
    algorithm_cls=_off2_on_fac_algorithm_cls,
    build_agent=build_fac,
)
