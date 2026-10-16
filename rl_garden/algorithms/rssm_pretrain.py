"""``RSSMPretrain``: reward-free RSSM world-model pretraining on offline image
sequences (NCRL stage 1).

The world model is an ``RSSM`` built with ``heads=()`` (no reward or
continue head). ``backbone="dreamer_v3"`` (constructor default) uses a
``DreamerConvEncoder`` and an ``ObservationDecoder``; ``backbone="ncrl"`` (the
CLI default) uses the official NCRL networks (``NCRLConvEncoder``,
``NCRLConvDecoder``, LayerNorm block-GRU; 64x64 images only), see
``rl_garden.networks.ncrl_nets``. The backbone is recorded in the checkpoint
metadata. Its loss is ``dyn_scale * dyn + rep_scale * rep +
recon_scale * recon`` (official scales 1.0 / 0.1 / 1.0). The KL
free-nats clip defaults to ``kl_free_mode="mean"``, which clips the batch-mean
KL as the official code does, after mixing ``unimix`` into both distributions.
The stochastic state uses NCRL's probs straight-through estimator
(``straight_through="probs"``) instead of DreamerV3's hard Gumbel-softmax.

Only ``_update_model`` trains. ``_update_actor_critic`` and
``_update_targets`` are no-ops. The model optimizer is NCRL's: Adam (lr 1e-4,
eps 1e-5), gradient-norm clip 100, then a multiplicative weight decay
``p <- (1 - wd) * p`` (wd 1e-6) applied to every parameter after clipping
and before the Adam step. This is not Adam's L2 term and not AdamW's
lr-scaled decay (``3rd_party/ncrl/src/dreamer/dreamer_utils.py:183-244``).
Autocast runs in ``compute_dtype`` (bf16 by default, as DreamerV3) instead of
the official fp16 GradScaler, so there is no loss scaling.

The policy is a ``DreamerPolicy`` so the checkpoint stores the RSSM under
``policy.actor_extractor.*``, the same namespace the stage-2 algorithm reads.
``DreamerPolicy`` always builds an actor and critic. They are left untrained
here (excluded from the optimizer) and stage 2 initializes its own.

Action dims are zero-padded to the dataset's ``padded_action_dim``, the
largest action dim in the dataset (deviation from the official code, which
pads every embodiment to a fixed 21), so the RSSM's action input is
``padded_action_dim`` wide and always zero in the padding dims.

Two data sources (``dataset_backend``):

- ``"mmap"`` (default): ``MmapSequenceDataset`` samples episode-strict
  windows straight from the memory-mapped files. Use this for the real NCRL
  data, which is tens of GB per embodiment.
- ``"buffer"``: ``load_sequence_dataset`` copies the dataset into a
  uniform-mode ``SequenceReplayBuffer`` (``load_dataset()``). Only for data
  that fits in ``buffer_device`` memory.

Either way, every sampled window starts from a zero carry with
``is_first[0] = True`` and the carry is never written back (no latent
replay, as in the official NCRL code).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.offline_model_based import OfflineModelBasedAlgorithm
from rl_garden.buffers.sequence_dataset import MmapSequenceDataset, infer_sequence_dataset_spec, load_sequence_dataset
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.observations import normalize_observation_space
from rl_garden.observations.schema import ObservationSchema
from rl_garden.policies.dreamer_policy import DreamerPolicy
from rl_garden.world_models.base import State
from rl_garden.world_models.rssm import RSSM, RSSM_SIZES, RSSMSize, build_rssm_encoder

_KL_LOSS_KEYS = ("dyn", "rep")


def resolve_rssm_size(
    size: str,
    *,
    deter: Optional[int] = None,
    hidden: Optional[int] = None,
    classes: Optional[int] = None,
    cnn_depth: Optional[int] = None,
) -> RSSMSize:
    """``RSSM_SIZES[size]`` with any non-``None`` override applied
    (``classes`` is ``RSSMSize.discrete``)."""
    if size not in RSSM_SIZES:
        raise ValueError(f"Unknown RSSM size {size!r}; choose from {sorted(RSSM_SIZES)}.")
    overrides = {"deter": deter, "hidden": hidden, "discrete": classes, "cnn_depth": cnn_depth}
    return dataclasses.replace(RSSM_SIZES[size], **{k: v for k, v in overrides.items() if v is not None})


def sequence_dataset_env_spec(dataset_dir: str | Path, padded_action_dim: Optional[int] = None) -> OfflineEnvSpec:
    """The ``OfflineEnvSpec`` (Dict image observation, ``[-1, 1]`` padded action
    Box) implied by a converted sequence dataset's manifests.
    ``padded_action_dim`` widens the action Box beyond the manifest's value."""
    spec = infer_sequence_dataset_spec(dataset_dir)
    action_dim = spec.padded_action_dim if padded_action_dim is None else padded_action_dim
    if action_dim < spec.padded_action_dim:
        raise ValueError(f"padded_action_dim {action_dim} < dataset's {spec.padded_action_dim}.")
    return OfflineEnvSpec(
        observation_space=spaces.Dict(
            {spec.obs_key: spaces.Box(low=0, high=255, shape=spec.image_shape, dtype=np.uint8)}
        ),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        num_envs=1,
    )


class RSSMPretrain(OfflineModelBasedAlgorithm):
    _compatible_checkpoint_algorithms = ("RSSMPretrain",)
    encoder_sharing = "shared"
    encoder_sharing_choices = ("shared",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        dataset_dir: str | Path,
        dataset_backend: Literal["mmap", "buffer"] = "mmap",
        backbone: Literal["dreamer_v3", "ncrl"] = "dreamer_v3",
        size: str = "ncrl_m",
        deter: Optional[int] = None,
        hidden: Optional[int] = None,
        classes: Optional[int] = None,
        cnn_depth: Optional[int] = None,
        stoch: int = 32,
        blocks: int = 8,
        unimix: float = 0.01,
        obs_layers: int = 1,
        img_layers: int = 2,
        dyn_layers: int = 1,
        decoder_layers: int = 3,
        kl_free: float = 1.0,
        kl_free_mode: Literal["per_element", "mean"] = "mean",
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        recon_scale: float = 1.0,
        batch_size: int = 32,
        batch_length: int = 64,
        lr: float = 1e-4,
        adam_eps: float = 1e-5,
        weight_decay: float = 1e-6,
        grad_clip: float = 100.0,
        compute_dtype: Literal["float32", "bfloat16"] = "bfloat16",
        buffer_device: str = "cuda",
        # Test-only escape hatch: replaces the resolved size row entirely.
        rssm_size: Optional[RSSMSize] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        if dataset_backend not in ("mmap", "buffer"):
            raise ValueError(f"dataset_backend must be 'mmap' or 'buffer', got {dataset_backend!r}.")
        if not isinstance(env.single_action_space, spaces.Box):
            raise NotImplementedError(f"{type(self).__name__} needs a Box action space.")
        super().__init__(
            env=env,
            buffer_size=0,
            buffer_device=buffer_device,
            batch_size=batch_size,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=0,
            eval_env=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=False,
            save_final_checkpoint=save_final_checkpoint,
        )
        self.dataset_dir = str(dataset_dir)
        self.dataset_backend = dataset_backend
        self.backbone = backbone
        self.size = size
        self.rssm_size = rssm_size or resolve_rssm_size(
            size, deter=deter, hidden=hidden, classes=classes, cnn_depth=cnn_depth
        )
        self.stoch = stoch
        self.blocks = blocks
        self.unimix = unimix
        self.obs_layers = obs_layers
        self.img_layers = img_layers
        self.dyn_layers = dyn_layers
        self.decoder_layers = decoder_layers
        self.kl_free = kl_free
        self.kl_free_mode = kl_free_mode
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.recon_scale = recon_scale
        self.batch_length = batch_length
        self.lr = lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.compute_dtype = compute_dtype
        self._autocast_kwargs = dict(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=compute_dtype == "bfloat16"
        )
        self.padded_action_dim = int(env.single_action_space.shape[0])
        self.straight_through = "probs"
        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        observation_space = self.env.single_observation_space
        schema = ObservationSchema.from_space(normalize_observation_space(observation_space))
        size = self.rssm_size
        encoder = build_rssm_encoder(self.backbone, observation_space, schema, size)
        world_model = RSSM(
            encoder,
            schema,
            self.padded_action_dim,
            size,
            stoch=self.stoch,
            unimix=self.unimix,
            blocks=self.blocks,
            obs_layers=self.obs_layers,
            img_layers=self.img_layers,
            dyn_layers=self.dyn_layers,
            decoder_layers=self.decoder_layers,
            kl_free=self.kl_free,
            heads=(),
            kl_free_mode=self.kl_free_mode,
            straight_through=self.straight_through,
            backbone=self.backbone,
        )
        self.policy = DreamerPolicy(
            observation_space, self.env.single_action_space, world_model, units=size.units
        ).to(self.device)
        self._model_params = list(self.policy.world_model.parameters())
        self.model_optimizer = torch.optim.Adam(self._model_params, lr=self.lr, eps=self.adam_eps)

        carry_spec = {"deter": (size.deter,), "stoch": (self.stoch * size.discrete,)}
        if self.dataset_backend == "mmap":
            self.replay_buffer = MmapSequenceDataset(
                self.dataset_dir, self.batch_length, carry_spec=carry_spec, sample_device=self.device
            )
            self.buffer_size = self.replay_buffer.num_steps
        else:
            spec = infer_sequence_dataset_spec(self.dataset_dir)
            self.buffer_size = max(spec.num_steps, self.batch_length + 2)
            self.replay_buffer = SequenceReplayBuffer(
                observation_space=observation_space,
                action_space=self.env.single_action_space,
                num_envs=1,
                buffer_size=self.buffer_size,
                cross_episode=True,
                priority=False,
                horizon=self.batch_length,
                carry_spec=carry_spec,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )

    def load_dataset(self) -> int:
        """Fills the ``"buffer"`` backend from ``dataset_dir`` (a no-op for
        ``"mmap"``, which reads the files lazily). Returns the step count."""
        if self.dataset_backend == "buffer":
            return load_sequence_dataset(self.dataset_dir, self.replay_buffer)
        return self.replay_buffer.num_steps

    # ------------------------------------------------------------------
    # Gradient step
    # ------------------------------------------------------------------

    def _sample_train_batch(self):
        batch = self.replay_buffer.sample(self.batch_size)
        batch.is_first[0] = True  # every window starts fresh from the zero carry (as in the official NCRL code)
        return batch

    def _update_model(self, batch) -> tuple[dict[str, float], State]:
        self.policy.world_model.train()
        with torch.autocast(**self._autocast_kwargs):
            losses, posterior = self.world_model.model_loss(batch)
        losses = {k: v.float() for k, v in losses.items()}
        recon_loss = sum(v for k, v in losses.items() if k not in _KL_LOSS_KEYS)
        total_loss = (
            self.dyn_scale * losses["dyn"] + self.rep_scale * losses["rep"] + self.recon_scale * recon_loss
        )

        self.model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model_params, self.grad_clip)
        if self.weight_decay:
            with torch.no_grad():
                for param in self._model_params:
                    param.mul_(1.0 - self.weight_decay)
        self.model_optimizer.step()

        metrics = {f"{k}_loss": float(v.detach()) for k, v in losses.items()}
        metrics["recon_loss"] = float(recon_loss.detach())
        metrics["total_loss"] = float(total_loss.detach())
        metrics["grad_norm"] = float(grad_norm)
        return metrics, posterior

    def _update_actor_critic(self, batch, posterior: State) -> dict[str, float]:
        return {}

    def _update_targets(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("model_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        observation_space = self.env.single_observation_space
        image_shapes = {key: list(space.shape) for key, space in observation_space.spaces.items()}
        return {
            **super()._checkpoint_metadata(),
            "dataset_dir": self.dataset_dir,
            "dataset_backend": self.dataset_backend,
            "backbone": self.backbone,
            "size": self.size,
            "rssm_size": dataclasses.asdict(self.rssm_size),
            "stoch": self.stoch,
            "blocks": self.blocks,
            "unimix": self.unimix,
            "obs_layers": self.obs_layers,
            "img_layers": self.img_layers,
            "dyn_layers": self.dyn_layers,
            "decoder_layers": self.decoder_layers,
            "heads": list(self.world_model.heads),
            "padded_action_dim": self.padded_action_dim,
            "image_shape": image_shapes,
            "kl_free": self.kl_free,
            "kl_free_mode": self.kl_free_mode,
            "straight_through": self.straight_through,
            "dyn_scale": self.dyn_scale,
            "rep_scale": self.rep_scale,
            "recon_scale": self.recon_scale,
            "batch_length": self.batch_length,
            "lr": self.lr,
            "adam_eps": self.adam_eps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "compute_dtype": self.compute_dtype,
        }
