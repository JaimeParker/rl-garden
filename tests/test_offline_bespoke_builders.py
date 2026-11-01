"""Builders of the bespoke offline entries: the eight offline entrypoints that
train on a dataset loaded by the agent itself (or on a dataset directory)
instead of the generic runner's replay buffer.

Each entry registers its algorithm class and its builder, so checkpoint
evaluation resolves its checkpoints (``--algorithm auto``) and rebuilds and
loads them. The dry run of every entry constructs the agent with exactly the
constructor call recorded before the builders were extracted.
"""

from __future__ import annotations

import io
import json
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest
from _ncrl_synthetic import write_benchmark
from test_a2a_bc import _write_vision_h5_dataset
from test_diffusion_bc_cli import _write_h5_dataset
from test_hilp import _write_h5 as write_goal_dataset
from test_pwm_smoke import write_converted_world_model, write_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from convert_ncrl_dataset import convert_dataset  # noqa: E402

ENTRIES = (
    "a2a_bc",
    "consistency_distill_bc",
    "diffusion_bc",
    "hilp",
    "opal",
    "pwm",
    "rssm_pretrain",
    "tdmpc2_multitask",
)
DIFFUSION_ARGS = ["--horizon_steps", "2", "--cond_steps", "1", "--denoising_steps", "5"]
MULTITASK_ARCH = [
    "--task_dim", "8", "--latent_dim", "16", "--enc_dim", "8", "--mlp_dim", "16",
    "--num_q", "2", "--num_bins", "11", "--horizon", "2", "--buffer_size", "500",
]


def run_cli(argv: list[str]) -> str:
    from rl_garden.training.offline import registry

    out = io.StringIO()
    with redirect_stdout(out):
        registry.run_cli(argv)
    return out.getvalue()


def common_argv(name: str, root: Path) -> list[str]:
    return [
        "--device", "cpu",
        "--log_type", "none",
        "--log_dir", str(root / "runs"),
        "--exp_name", f"{name}_run",
        "--num_offline_steps", "2",
    ]


@dataclass(frozen=True)
class Entries:
    """Tiny datasets for every entry under ``root`` and ``{entry: training argv}``."""

    root: Path
    argv: dict[str, list[str]]


def write_entries(root: Path) -> Entries:
    state_h5 = root / "state.h5"
    _write_h5_dataset(state_h5, num_traj=4, steps_per_traj=10, obs_dim=3, action_dim=2)
    vision_h5 = root / "vision.h5"
    _write_vision_h5_dataset(vision_h5, num_traj=4, steps_per_traj=10, state_dim=4, action_dim=2)
    goal_h5 = root / "goal.h5"
    write_goal_dataset(goal_h5)
    multitask_dir = write_dataset(root / "multitask")
    world_model_dir = write_converted_world_model(root / "world_model")
    write_benchmark(root / "sequence_src", {"cheetah": (6, (14, 16, 12))}, image_size=32)
    convert_dataset(root / "sequence_src", root / "sequence")

    teacher_dir = root / "teacher"
    run_cli(
        [
            "diffusion_bc", "--dataset_path", str(state_h5), *DIFFUSION_ARGS,
            "--batch_size", "8", *common_argv("teacher", root),
            "--checkpoint_dir", str(teacher_dir),
        ]
    )
    argv = {
        "diffusion_bc": [
            "--dataset_path", str(state_h5), *DIFFUSION_ARGS, "--batch_size", "8",
        ],
        "consistency_distill_bc": [
            "--dataset_path", str(state_h5),
            "--bc_checkpoint", str(teacher_dir / "diffusion_bc_offline_pretrained.pt"),
            *DIFFUSION_ARGS, "--batch_size", "8",
        ],
        "a2a_bc": [
            "--dataset_path", str(vision_h5),
            "--horizon_steps", "4", "--cond_steps", "2", "--latent_dim", "16",
            "--cnn_num_layers", "1", "--cnn_hidden_channels", "8",
            "--decoder_net_arch", "32", "32", "--flow_hidden_dims", "32", "32",
            "--batch_size", "16", "--obs.rgb", "cam",
            "--encoder.backbone", "plain_conv", "--encoder.features_dim", "16",
            "--encoder.plain_conv_pooling", "gap",
        ],
        "hilp": [
            "--dataset_path", str(goal_h5), "--batch_size", "32", "--skill_dim", "4",
            "--value_hidden_dims", "16", "16", "--actor_hidden_dims", "16", "16",
        ],
        "opal": [
            "--dataset_path", str(goal_h5), "--batch_size", "16", "--skill_dim", "4",
            "--chunk_size", "4", "--hidden_size", "8", "--vae_hidden_dims", "8", "8",
        ],
        "pwm": [
            "--dataset_dir", str(multitask_dir), "--mmap_dir", str(root / "pwm_mmap"),
            "--world_model_checkpoint", str(world_model_dir), "--task", "task_b",
            "--horizon", "3", "--batch_size", "8", "--actor_units", "16", "16",
            "--critic_units", "16", "--critic_iterations", "1", "--critic_batches", "2",
            "--eval_freq", "0",
        ],
        "rssm_pretrain": [
            "--dataset_dir", str(root / "sequence"), "--buffer_device", "cpu",
            "--compute_dtype", "float32", "--size", "size12M", "--deter", "32",
            "--hidden", "16", "--classes", "4", "--cnn_depth", "4", "--stoch", "4",
            "--batch_size", "4", "--batch_length", "8", "--backbone", "dreamer_v3",
        ],
        "tdmpc2_multitask": [
            "--dataset_dir", str(multitask_dir), "--mmap_dir", str(root / "tdmpc2_mmap"),
            *MULTITASK_ARCH, "--batch_size", "4",
        ],
    }
    return Entries(
        root, {name: [name, *entry, *common_argv(name, root)] for name, entry in argv.items()}
    )


@pytest.fixture(scope="module")
def entries(tmp_path_factory) -> Entries:
    return write_entries(tmp_path_factory.mktemp("bespoke"))


def normalized(payload, root: Path):
    """``payload`` with the fixture root and the dry run's temporary mmap
    directory replaced by stable placeholders."""
    text = json.dumps(payload).replace(str(root), "<root>")
    text = re.sub(r"[^\"]*rlg-dry-run-[^\"]*", "<dry-run-mmap>", text)
    return json.loads(text)


def dry_run_algorithm(name: str, entries: Entries) -> dict:
    output = run_cli([*entries.argv[name], "--dry-run"])
    return normalized(json.loads(output)["algorithm"], entries.root)


#: ``{entry: dry-run "algorithm" payload}``, recorded before the builders were
#: extracted from the run functions.
RECORDED_DRY_RUN: dict[str, dict] = {
    "a2a_bc": {
        "constructor_kwargs": {
            "actor_lr": 0.001,
            "batch_size": 16,
            "checkpoint_dir": "<root>/runs/a2a_bc_run/checkpoints",
            "checkpoint_freq": 0,
            "cnn_activation_fn": "relu",
            "cnn_hidden_channels": 8,
            "cnn_kernel_size": 5,
            "cnn_num_layers": 1,
            "cond_steps": 2,
            "consistency_weight": 1.0,
            "contrastive_temperature": 0.1,
            "dataset_path": "<root>/vision.h5",
            "decoder_net_arch": [32, 32],
            "device": "cpu",
            "enc_contrastive_weight": 0.0,
            "enc_recon_weight": 0.5,
            "encoder_config": {
                "backbone": "plain_conv",
                "dreamer_depth": 16,
                "features_dim": 16,
                "freeze_resnet_backbone": False,
                "freeze_resnet_encoder": False,
                "image_augmentation": "none",
                "image_fusion_mode": "stack_channels",
                "image_random_shift_pad": 4,
                "normalize_obs": False,
                "plain_conv_last_act": True,
                "plain_conv_pooling": "gap",
                "plain_conv_weight_init": "kaiming_uniform",
                "pooling_method": "spatial_softmax",
                "pretrained_weights": None,
                "proprio_latent_dim": 64,
                "vit_actor_feature_dim": 128,
                "vit_augmentation": "random_shift",
                "vit_critic_spatial_emb_dim": 1024,
                "vit_depth": 1,
                "vit_embed_dim": 128,
                "vit_embed_norm": False,
                "vit_fusion_mode": "per_key",
                "vit_num_heads": 4,
                "vit_random_shift_pad": 4,
            },
            "env": "<env>",
            "flow_contrastive_weight": 0.0,
            "flow_hidden_dims": [32, 32],
            "flow_recon_weight": 0.5,
            "grad_clip_norm": None,
            "horizon_steps": 4,
            "image_augmentation_seed": 1000004,
            "latent_dim": 16,
            "log_freq": 1000,
            "logger": "<logger>",
            "lr_decay_steps": 0,
            "lr_min_ratio": 0.0,
            "lr_schedule": "constant",
            "lr_warmup_steps": 0,
            "num_sampling_steps": 6,
            "num_traj": None,
            "obs_groups": None,
            "save_final_checkpoint": True,
            "seed": 1,
            "std_log": True,
            "weight_decay": 1e-06,
        },
        "target": "rl_garden.algorithms.a2a_bc.A2ABC",
    },
    "consistency_distill_bc": {
        "constructor_kwargs": {
            "activation_fn": "relu",
            "batch_size": 8,
            "bc_checkpoint": "<root>/teacher/diffusion_bc_offline_pretrained.pt",
            "checkpoint_dir": "<root>/runs/consistency_distill_bc_run/checkpoints",
            "checkpoint_freq": 0,
            "cm_ema_decay": 0.95,
            "cm_grad_clip_norm": 1.0,
            "cm_lr": 0.0001,
            "cm_sigma_data": 0.5,
            "cm_timestep_scaling": 0.1,
            "cond_steps": 1,
            "dataset_path": "<root>/state.h5",
            "denoised_clip_value": 1.0,
            "denoising_steps": 5,
            "device": "cpu",
            "env": "<env>",
            "final_action_clip_value": None,
            "horizon_steps": 2,
            "kernel_init": None,
            "log_freq": 1000,
            "logger": "<logger>",
            "min_sampling_denoising_std": 0.1,
            "net_cls": "rl_garden.networks.diffusion_mlp.DiffusionMLP",
            "net_kwargs": None,
            "num_traj": None,
            "randn_clip_value": 10.0,
            "residual_style": True,
            "save_final_checkpoint": True,
            "seed": 1,
            "std_log": True,
            "time_dim": 16,
            "weight_decay": 1e-06,
        },
        "target": "rl_garden.algorithms.consistency_distill_bc.ConsistencyDistillBC",
    },
    "diffusion_bc": {
        "constructor_kwargs": {
            "activation_fn": "relu",
            "actor_lr": 0.001,
            "batch_size": 8,
            "checkpoint_dir": "<root>/runs/diffusion_bc_run/checkpoints",
            "checkpoint_freq": 0,
            "cond_steps": 1,
            "dataset_path": "<root>/state.h5",
            "denoised_clip_value": 1.0,
            "denoising_steps": 5,
            "device": "cpu",
            "ema_decay": 0.995,
            "ema_start_step": 0,
            "ema_update_every": 10,
            "encoder_config": {
                "backbone": "plain_conv",
                "dreamer_depth": 16,
                "features_dim": 256,
                "freeze_resnet_backbone": False,
                "freeze_resnet_encoder": False,
                "image_augmentation": "none",
                "image_fusion_mode": "stack_channels",
                "image_random_shift_pad": 4,
                "normalize_obs": False,
                "plain_conv_last_act": True,
                "plain_conv_pooling": "flatten",
                "plain_conv_weight_init": "kaiming_uniform",
                "pooling_method": "spatial_softmax",
                "pretrained_weights": None,
                "proprio_latent_dim": 64,
                "vit_actor_feature_dim": 128,
                "vit_augmentation": "random_shift",
                "vit_critic_spatial_emb_dim": 1024,
                "vit_depth": 1,
                "vit_embed_dim": 128,
                "vit_embed_norm": False,
                "vit_fusion_mode": "per_key",
                "vit_num_heads": 4,
                "vit_random_shift_pad": 4,
            },
            "env": "<env>",
            "final_action_clip_value": None,
            "grad_clip_norm": None,
            "horizon_steps": 2,
            "kernel_init": None,
            "log_freq": 1000,
            "logger": "<logger>",
            "lr_decay_steps": 0,
            "lr_min_ratio": 0.0,
            "lr_schedule": "constant",
            "lr_warmup_steps": 0,
            "min_sampling_denoising_std": 0.1,
            "net_cls": "rl_garden.networks.diffusion_mlp.DiffusionMLP",
            "net_kwargs": None,
            "num_traj": None,
            "randn_clip_value": 10.0,
            "residual_style": True,
            "save_final_checkpoint": True,
            "seed": 1,
            "std_log": True,
            "time_dim": 16,
            "weight_decay": 1e-06,
        },
        "target": "rl_garden.algorithms.diffusion_bc.DiffusionBC",
    },
    "hilp": {
        "constructor_kwargs": {
            "actor_hidden_dims": [16, 16],
            "batch_size": 32,
            "checkpoint_dir": "<root>/runs/hilp_run/checkpoints",
            "checkpoint_freq": 0,
            "dataset_path": "<root>/goal.h5",
            "device": "cpu",
            "discount": 0.99,
            "encoder_config": None,
            "env": "<env>",
            "expectile": 0.95,
            "log_freq": 1000,
            "logger": "<logger>",
            "lr": 0.0003,
            "num_traj": None,
            "obs_groups": None,
            "p_currgoal": 0.0,
            "p_randomgoal": 0.375,
            "p_trajgoal": 0.625,
            "save_final_checkpoint": True,
            "seed": 1,
            "skill_dim": 4,
            "skill_discount": 0.99,
            "skill_expectile": 0.9,
            "skill_temperature": 10.0,
            "std_log": True,
            "tau": 0.005,
            "value_hidden_dims": [16, 16],
        },
        "target": "rl_garden.algorithms.hilp.HILP",
    },
    "opal": {
        "constructor_kwargs": {
            "batch_size": 16,
            "checkpoint_dir": "<root>/runs/opal_run/checkpoints",
            "checkpoint_freq": 0,
            "chunk_size": 4,
            "dataset_path": "<root>/goal.h5",
            "device": "cpu",
            "encoder_config": None,
            "env": "<env>",
            "hidden_size": 8,
            "kl_coef": 0.1,
            "log_freq": 1000,
            "logger": "<logger>",
            "lr": 0.0003,
            "num_traj": None,
            "obs_groups": None,
            "save_final_checkpoint": True,
            "seed": 1,
            "skill_dim": 4,
            "std_log": True,
            "vae_hidden_dims": [8, 8],
        },
        "target": "rl_garden.algorithms.opal.OPAL",
    },
    "pwm": {
        "constructor_kwargs": {
            "actor_grad_norm": 1.0,
            "actor_lr": 0.0005,
            "actor_units": [16, 16],
            "batch_size": 8,
            "buffer_size": 32,
            "checkpoint_dir": "<root>/runs/pwm_run/checkpoints",
            "checkpoint_freq": 0,
            "critic_batches": 2,
            "critic_grad_norm": 100.0,
            "critic_iterations": 1,
            "critic_lr": 0.0005,
            "critic_units": [16],
            "device": "cpu",
            "env": "<env>",
            "eval_env": "<eval_env>",
            "eval_freq": 0,
            "gamma": 0.99,
            "horizon": 3,
            "init_logstd": -1.0,
            "lam": 0.95,
            "log_freq": 1000,
            "logger": "<logger>",
            "lr_decay_steps": 2,
            "lr_final": 1e-05,
            "min_logstd": -1.427,
            "mmap_dir": "<dry-run-mmap>",
            "mmap_mode": "create",
            "num_critics": 3,
            "num_eval_episodes": 10,
            "num_eval_steps": 16,
            "ret_rms": True,
            "save_final_checkpoint": True,
            "save_replay_buffer": False,
            "seed": 1,
            "std_log": True,
            "task_idx": 1,
            "world_model_checkpoint": "<root>/world_model",
        },
        "target": "rl_garden.algorithms.pwm.PWM",
    },
    "rssm_pretrain": {
        "constructor_kwargs": {
            "adam_eps": 1e-05,
            "backbone": "dreamer_v3",
            "batch_length": 8,
            "batch_size": 4,
            "blocks": 8,
            "buffer_device": "cpu",
            "checkpoint_dir": "<root>/runs/rssm_pretrain_run/checkpoints",
            "checkpoint_freq": 0,
            "classes": 4,
            "cnn_depth": 4,
            "compute_dtype": "float32",
            "dataset_backend": "mmap",
            "dataset_dir": "<root>/sequence",
            "deter": 32,
            "device": "cpu",
            "dyn_scale": 1.0,
            "env": "<env>",
            "grad_clip": 100.0,
            "hidden": 16,
            "kl_free": 1.0,
            "kl_free_mode": "mean",
            "log_freq": 1000,
            "logger": "<logger>",
            "lr": 0.0001,
            "recon_scale": 1.0,
            "rep_scale": 0.1,
            "save_final_checkpoint": True,
            "seed": 1,
            "size": "size12M",
            "std_log": True,
            "stoch": 4,
            "weight_decay": 1e-06,
        },
        "target": "rl_garden.algorithms.rssm_pretrain.RSSMPretrain",
    },
    "tdmpc2_multitask": {
        "constructor_kwargs": {
            "action_dims": [2, 3, 2],
            "batch_size": 4,
            "buffer_size": 500,
            "checkpoint_dir": "<root>/runs/tdmpc2_multitask_run/checkpoints",
            "checkpoint_freq": 0,
            "consistency_coef": 20.0,
            "device": "cpu",
            "discount_denom": 5.0,
            "discount_max": 0.995,
            "discount_min": 0.95,
            "dropout": 0.01,
            "enc_dim": 8,
            "enc_lr_scale": 0.3,
            "entropy_coef": 0.0001,
            "env": "<env>",
            "episode_lengths": [10, 8, 12],
            "grad_clip_norm": 20.0,
            "horizon": 2,
            "latent_dim": 16,
            "log_freq": 1000,
            "log_std_max": 2.0,
            "log_std_min": -10.0,
            "logger": "<logger>",
            "lr": 0.0003,
            "mlp_dim": 16,
            "mmap_dir": "<dry-run-mmap>",
            "mmap_mode": "create",
            "num_bins": 11,
            "num_enc_layers": 2,
            "num_q": 2,
            "obs_dims": [4, 6, 5],
            "reward_coef": 0.1,
            "rho": 0.5,
            "save_final_checkpoint": True,
            "save_replay_buffer": False,
            "seed": 1,
            "simnorm_dim": 8,
            "std_log": True,
            "task_dim": 8,
            "tasks": ["task_a", "task_b", "task_c"],
            "tau": 0.01,
            "value_coef": 0.1,
            "vmax": 10.0,
            "vmin": -10.0,
        },
        "target": "rl_garden.algorithms.tdmpc2_multitask.TDMPC2Multitask",
    },
}


@pytest.mark.parametrize("name", ENTRIES)
def test_dry_run_constructs_the_recorded_agent(name, entries):
    assert dry_run_algorithm(name, entries) == RECORDED_DRY_RUN[name]


#: ``{entry: (algorithm class name, checkpoint file name the run saves)}``.
ENTRY_OUTPUTS = {
    "a2a_bc": ("A2ABC", "a2a_bc_offline_pretrained.pt"),
    "consistency_distill_bc": ("ConsistencyDistillBC", "consistency_distill_bc_offline.pt"),
    "diffusion_bc": ("DiffusionBC", "diffusion_bc_offline_pretrained.pt"),
    "hilp": ("HILP", "hilp_offline_pretrained.pt"),
    "opal": ("OPAL", "opal_offline_pretrained.pt"),
    "pwm": ("PWM", "pwm_offline.pt"),
    "rssm_pretrain": ("RSSMPretrain", "rssm_pretrain_offline_pretrained.pt"),
    "tdmpc2_multitask": ("TDMPC2Multitask", "tdmpc2_multitask_offline_pretrained.pt"),
}
#: Entries whose agent creates an mmap buffer directory: a rebuild needs a fresh one.
MMAP_ENTRIES = ("pwm", "tdmpc2_multitask")


def offline_entry(name: str):
    from rl_garden.training.offline import registry

    registry.discover()
    return registry.entries()[name]


def entry_env_spec(name: str, args):
    """The ``OfflineEnvSpec`` the entry's run function trains on."""
    from gymnasium import spaces

    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.algorithms.rssm_pretrain import sequence_dataset_env_spec
    from rl_garden.buffers.h5_dataset import (
        infer_box_specs_from_h5,
        infer_specs_from_h5,
    )
    from rl_garden.buffers.tdmpc2_multitask_dataset import infer_multitask_dataset_specs

    if name in ("a2a_bc", "diffusion_bc"):
        return OfflineEnvSpec(*infer_specs_from_h5(args.dataset_path), num_envs=1)
    if name in ("consistency_distill_bc", "hilp", "opal"):
        return OfflineEnvSpec(*infer_box_specs_from_h5(args.dataset_path), num_envs=1)
    if name == "rssm_pretrain":
        return sequence_dataset_env_spec(args.dataset_dir, args.padded_action_dim)
    _, obs_dims, action_dims, _ = infer_multitask_dataset_specs(args.dataset_dir)
    return OfflineEnvSpec(
        spaces.Box(low=-1e6, high=1e6, shape=(max(obs_dims),), dtype="float32"),
        spaces.Box(low=-1.0, high=1.0, shape=(max(action_dims),), dtype="float32"),
        num_envs=1,
    )


@pytest.fixture(scope="module")
def checkpoints(entries) -> dict[str, Path]:
    """``{entry: checkpoint}`` of a two-step training run of every entry."""
    paths = {}
    for name in ENTRIES:
        run_cli(entries.argv[name])
        _, filename = ENTRY_OUTPUTS[name]
        paths[name] = entries.root / "runs" / f"{name}_run" / "checkpoints" / filename
        assert paths[name].exists(), name
    return paths


@pytest.mark.parametrize("name", ENTRIES)
def test_entry_registers_its_algorithm_class_and_module_builder(name):
    import importlib

    from rl_garden.training import evaluation

    entry = offline_entry(name)
    class_name, _ = ENTRY_OUTPUTS[name]
    module = importlib.import_module(f"rl_garden.training.offline.entrypoints.{name}")
    assert entry.algorithm_cls().__name__ == class_name
    assert entry.build_agent is getattr(module, f"build_{name}")
    assert evaluation._registered_by_class_name(("offline",), class_name) == [("offline", name)]


@pytest.mark.parametrize("name", ENTRIES)
def test_builder_takes_a_subclass_and_replaces_or_extends_constructor_kwargs(
    name, entries, monkeypatch
):
    from rl_garden.training import inspection
    from rl_garden.training.offline import registry

    entry = offline_entry(name)
    args = registry.parse_args(entries.argv[name])
    env_spec = entry_env_spec(name, args)
    constructed = []
    monkeypatch.setattr(
        inspection,
        "construct_agent",
        lambda cls, **kwargs: constructed.append((cls, kwargs)),
    )

    class Subclass:
        pass

    entry.build_agent(args, env_spec, "logger")
    entry.build_agent(
        args, env_spec, "logger", algorithm_cls=Subclass, checkpoint_dir="ckpt", extra_flag=7
    )

    (default_cls, default_kwargs), (subclass, subclass_kwargs) = constructed
    assert default_cls is entry.algorithm_cls()
    assert subclass is Subclass
    assert default_kwargs["env"] is env_spec and default_kwargs["checkpoint_dir"] is None
    assert subclass_kwargs == {**default_kwargs, "checkpoint_dir": "ckpt", "extra_flag": 7}
    assert list(subclass_kwargs) == [*default_kwargs, "extra_flag"]


@pytest.mark.parametrize("name", ENTRIES)
def test_trained_checkpoint_rebuilds_and_loads_through_the_registered_builder(
    name, checkpoints, tmp_path
):
    import torch

    from rl_garden.common import Logger
    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.training import evaluation

    path = checkpoints[name]
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    eval_args = evaluation.EvalCheckpointArgs(
        checkpoint_path=str(path), device="cpu", buffer_device="cpu"
    )
    phase, algorithm = evaluation._resolve_phase_and_algorithm("auto", "auto", checkpoint)
    assert (phase, algorithm) == ("offline", name)
    args = evaluation._load_training_args(phase, algorithm, eval_args, checkpoint)
    builder = evaluation._registered_builder(phase, algorithm)
    extra = {"mmap_dir": str(tmp_path / "mmap")} if name in MMAP_ENTRIES else {}

    agent = builder(args, entry_env_spec(name, args), Logger(log_type="none"), None, **extra)
    agent.load(path, strict=True, load_replay_buffer=False, load_optimizers=False)

    expected = checkpoint["state"]["policy"]
    actual = agent.policy.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key].cpu(), value.cpu()), key


@pytest.mark.parametrize("name", ENTRIES)
def test_evaluation_refuses_the_checkpoint_with_the_algorithm_reason(
    name, checkpoints, monkeypatch
):
    from rl_garden.training import evaluation

    def no_environment(*args, **kwargs):
        raise AssertionError("an unevaluable checkpoint must not build an environment")

    monkeypatch.setattr(evaluation, "make_evaluation_env", no_environment)
    class_name, _ = ENTRY_OUTPUTS[name]
    reason = offline_entry(name).algorithm_cls().checkpoint_eval_unsupported_reason
    assert reason

    with pytest.raises(SystemExit) as exit_info:
        evaluation.evaluate_checkpoint(
            evaluation.EvalCheckpointArgs(checkpoint_path=str(checkpoints[name]), device="cpu")
        )

    message = str(exit_info.value)
    assert message.startswith(f"offline:{name} ({class_name}) checkpoints are not evaluable")
    assert message.endswith(reason)
