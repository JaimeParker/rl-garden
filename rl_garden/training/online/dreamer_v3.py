"""DreamerV3 run function."""

from __future__ import annotations


def build_dreamer_v3(args, env, eval_env, logger, checkpoint_dir):
    import dataclasses

    from rl_garden.algorithms.dreamer_v3 import DreamerV3
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent

    # Encoder backbone label is informational only -- DreamerV3 always
    # builds its own DreamerConvEncoder directly from the RSSMSize preset's
    # depth/units (rl_garden.encoders.dreamer_conv.DreamerConvEncoder's own
    # docstring; see DreamerV3's docstring), never through the generic
    # backbone-dispatch factory. Set here (not silently defaulted inside the
    # class) purely so --print-config's materialized encoder_config reflects
    # what is actually in use for a visual run.
    encoder_config = args.encoder
    if args.obs.is_visual and encoder_config.backbone != "dreamer_conv":
        encoder_config = dataclasses.replace(encoder_config, backbone="dreamer_conv")

    agent = construct_agent(
        DreamerV3,
        env=env,
        eval_env=eval_env,
        size=args.size,
        stoch=args.stoch,
        unimix=args.unimix,
        blocks=args.blocks,
        obs_layers=args.obs_layers,
        img_layers=args.img_layers,
        dyn_layers=args.dyn_layers,
        decoder_layers=args.decoder_layers,
        reward_bins=args.reward_bins,
        kl_free=args.kl_free,
        contdisc=args.contdisc,
        discount_horizon=args.horizon,
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        train_ratio=args.train_ratio,
        imag_horizon=args.imag_horizon,
        lam=args.lam,
        act_entropy=args.act_entropy,
        dyn_scale=args.dyn_scale,
        rep_scale=args.rep_scale,
        recon_scale=args.recon_scale,
        rew_scale=args.rew_scale,
        con_scale=args.con_scale,
        policy_scale=args.policy_scale,
        value_scale=args.value_scale,
        repval_scale=args.repval_scale,
        lr=args.lr,
        warmup=args.warmup,
        slow_target_fraction=args.slow_target_fraction,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        compute_dtype=args.compute_dtype,
        encoder_config=encoder_config if args.obs.is_visual else None,
        obs_groups=resolve_obs_groups_config(args),
        critic_encoder_config=resolve_critic_encoder_config(args),
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_dreamer_v3(args: "DreamerV3Args") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = "rgbd_dreamer_conv" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_dreamer_v3,
    )


# ---------------------------------------------------------------------------
# Args + registration (bottom of file, no # noqa: E402 -- ruff has it disabled)
# ---------------------------------------------------------------------------
from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import DreamerV3TrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class DreamerV3Args(DreamerV3TrainingArgs, EnvBackendArgs):
    """DreamerV3 -- RSSM world model + imagination-based actor-critic
    (Hafner et al., ``rl_garden.algorithms.dreamer_v3`` module docstring).

    Env backend: ``--env_backend maniskill`` (default), ``--env_backend
    mujoco``, or ``--env_backend robotwin``. State-only observations by
    default; pass ``--obs.rgb <camera>`` for Dict/RGB observations (the
    ``mujoco`` backend is state-only, see ``MujocoBackend.resolve_config``).
    """


def _dreamer_v3_algorithm_cls() -> type:
    from rl_garden.algorithms.dreamer_v3 import DreamerV3

    return DreamerV3


registry.register("dreamer_v3", DreamerV3Args, run_dreamer_v3, algorithm_cls=_dreamer_v3_algorithm_cls)
