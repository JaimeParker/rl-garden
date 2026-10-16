"""``NCRL(DreamerV3)``: NCRL stage 2, online fine-tuning of a reward-free
pretrained RSSM with experience rehearsal and execution guidance.

Differences from ``DreamerV3`` (all implemented here, none in the parent):

1. **Loading.** The RSSM's architecture comes from an ``RSSMPretrain``
   checkpoint (``--pretrained_world_model``). Its encoder/dynamics/posterior/
   prior/decoder weights are loaded (``policy.actor_extractor.*``); the reward
   head (``heads=("reward",)``, ``reward_detached=True``, as in the official NCRL code), actor,
   critic and slow critic start from scratch. Nothing is frozen. ``backbone``
   must equal the checkpoint's (``"dreamer_v3"`` or the official NCRL
   networks ``"ncrl"``, see ``RSSM``); with ``"ncrl"`` the reward head, actor
   and critic are ``NCRLHead``s and the actor a masked TruncatedNormal.
   Mechanisms follow the official code for both backbones: the reward head is
   2 x 512, reward and critic use symlog two-hot (``SymTwoHot``), the
   stochastic state uses the probs straight-through estimator and the KL is
   taken after unimix. The actor,
   RSSM action input and both buffers are ``padded_action_dim`` wide; the
   actor is masked to the task's real action dims (deviation from the official code, which leaves samples unmasked) and the env
   receives ``action[:, :env_action_dim]``.
2. **Experience rehearsal.** Before the first rollout, the first frame of
   ``env.reset()`` is embedded with the pretrained encoder and matched (L2,
   ``torch.cdist``) against a retrieval DB built by
   ``tools/conversion/build_ncrl_retrieval_db.py``. Only episodes with the
   task's action dim are eligible; nearest unseen episodes are added for at
   most ``max_retrieve_iterations`` resets until ``num_retrieve_episodes`` 
   (deviation from the official code: at most 10 retrieval iterations, no episode picked twice, action dims filtered before the top-k rather than after). They fill the fixed prior half of a
   ``RehearsalSequenceBuffer``; every batch is ``online_ratio`` online rows +
   the rest prior rows. The reward head trains on online rows only. Prior
   rows always start from the zero state (``is_first[0]``), as official
   chunks do; online rows keep the DreamerV3 carry write-back.
3. **Execution guidance.** A ``BCGuide`` is trained on the prior buffer once,
   then frozen. During warmup (``max(warmup_frames, 5 * episode_frames)``)
   every action is the BC action. Afterwards each episode is guided with
   probability ``p = linear(1 -> 0.1, 100k policy steps)`` inside a random
   window ``S < t < S + W``, ``W in {L/10 * i}``, ``L = episode_frames /
   action_repeat``. Whatever action is executed is the one stored in the
   replay buffer and fed to the RSSM carry. Guidance needs the prior data, so
   ``online_ratio == 1`` also disables it (as in the official code).
4. **Optimizers** (official ``finetune.yaml``): world model (without the reward
   head) Adam 1e-4; reward head its own Adam 1e-4 (deviation from the official code, which shares the world-model optimizer); actor and
   critic Adam 8e-5. All use eps 1e-5, grad-norm clip 100 and a multiplicative
   weight decay ``p <- (1 - 1e-6) * p`` before the step (as ``RSSMPretrain``).
5. **Actor-critic (DreamerV2 style, as in the official code).** Imagination from every
   posterior row (online + prior) for ``imag_horizon`` steps with gradients
   through the dynamics into the actor (RSSM parameters temporarily
   ``requires_grad=False``); constant discount (no continue head); the
   lambda-return uses the slow critic and the DV2 reward index
   (:func:`dv2_lambda_return`); no return normalisation and no repval loss.
6. **Update cadence.** ``utd = train_ratio / (batch_size * batch_length)``
   gradient steps per env step, starting after ``learning_starts`` (default 10
   episodes); deviation from the official code, which runs a burst after each episode.

Steps vs frames: ``self._global_step`` counts policy steps (env steps after
action repeat). ``episode_frames``, ``warmup_frames`` and eval cadence in the
docs are in frames; ``action_repeat`` converts them.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.algorithms.dreamer_v3 import DreamerV3
from rl_garden.buffers.rehearsal_sequence_buffer import RehearsalSequenceBuffer, read_dataset_episodes
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.dreamer_nets import MLPHead
from rl_garden.networks.ncrl_nets import NCRLHead
from rl_garden.networks.returns import lambda_return
from rl_garden.observations import ObsGroups, normalize_observation_space
from rl_garden.observations.schema import ObservationContractError, ObservationSchema
from rl_garden.policies.action_mask import (
    MaskedBoundedNormal,
    MaskedTruncatedNormal,
    action_mask,
    masked_bounded_normal,
)
from rl_garden.policies.bc_guide import BCGuide, train_bc_guide
from rl_garden.policies.dreamer_policy import DreamerPolicy
from rl_garden.world_models.imagine import imagine
from rl_garden.world_models.rssm import RSSM, RSSMSize, build_rssm_encoder

RETRIEVAL_DB_FORMAT = "ncrl_retrieval_db_v1"
PRETRAIN_ALGORITHM_CLASS = "RSSMPretrain"
_RSSM_PREFIX = "actor_extractor."
_KL_AND_REWARD_KEYS = ("dyn", "rep", "rew")

_ACTION_FROM_ACTOR = 0
_ACTION_FROM_BC = 1
_ACTION_FROM_RANDOM = 2


# ----------------------------------------------------------------------
# Pretrained RSSM loading (shared with tools/conversion/build_ncrl_retrieval_db.py)
# ----------------------------------------------------------------------


def load_rssm_pretrain_checkpoint(path: str | Path) -> dict[str, Any]:
    """Reads an ``RSSMPretrain`` checkpoint onto the CPU."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    algorithm_class = checkpoint.get("metadata", {}).get("algorithm_class")
    if algorithm_class != PRETRAIN_ALGORITHM_CLASS:
        raise ValueError(f"{path} is a {algorithm_class!r} checkpoint, expected {PRETRAIN_ALGORITHM_CLASS!r}.")
    return checkpoint


def pretrained_observation_space(checkpoint: dict[str, Any]) -> spaces.Dict:
    """The image-only Dict observation space the pretrained encoder expects."""
    image_shape = checkpoint["metadata"]["hyperparameters"]["image_shape"]
    return spaces.Dict(
        {key: spaces.Box(low=0, high=255, shape=tuple(shape), dtype=np.uint8) for key, shape in image_shape.items()}
    )


def pretrained_backbone(checkpoint: dict[str, Any]) -> str:
    """The checkpoint's RSSM backbone; checkpoints written before the option
    existed are ``"dreamer_v3"``."""
    return str(checkpoint["metadata"]["hyperparameters"].get("backbone", "dreamer_v3"))


def rssm_from_pretrain_checkpoint(
    checkpoint: dict[str, Any],
    *,
    backbone: Optional[str] = None,
    heads: tuple[str, ...] = (),
    reward_detached: bool = False,
    reward_bins: int = 255,
    kl_free: float = 1.0,
    kl_free_mode: Literal["per_element", "mean"] = "mean",
    reward_layers: int = 2,
    reward_units: int = 512,
    reward_twohot_space: Literal["raw", "symlog"] = "symlog",
    straight_through: Literal["gumbel_hard", "probs"] = "probs",
) -> RSSM:
    """Builds an ``RSSM`` with the checkpoint's architecture and loads its
    weights. Only the heads in ``heads`` may be missing from the checkpoint
    (they keep their fresh init); any other missing or unexpected key raises.
    The reward head and straight-through defaults are NCRL's (2 x 512,
    symlog two-hot, probs straight-through); they carry no pretrained weights.
    ``backbone=None`` takes the checkpoint's; a given ``backbone`` must match it."""
    hyper = checkpoint["metadata"]["hyperparameters"]
    checkpoint_backbone = pretrained_backbone(checkpoint)
    if backbone is not None and backbone != checkpoint_backbone:
        raise ValueError(
            f"The pretrained RSSM was trained with backbone {checkpoint_backbone!r}, but backbone "
            f"{backbone!r} was requested; pass --backbone {checkpoint_backbone} to fine-tune it."
        )
    observation_space = pretrained_observation_space(checkpoint)
    schema = ObservationSchema.from_space(normalize_observation_space(observation_space))
    size = RSSMSize(**hyper["rssm_size"])
    encoder = build_rssm_encoder(checkpoint_backbone, observation_space, schema, size)
    world_model = RSSM(
        encoder,
        schema,
        int(hyper["padded_action_dim"]),
        size,
        stoch=int(hyper["stoch"]),
        unimix=float(hyper["unimix"]),
        blocks=int(hyper["blocks"]),
        obs_layers=int(hyper["obs_layers"]),
        img_layers=int(hyper["img_layers"]),
        dyn_layers=int(hyper["dyn_layers"]),
        decoder_layers=int(hyper["decoder_layers"]),
        reward_bins=reward_bins,
        kl_free=kl_free,
        heads=heads,
        reward_detached=reward_detached,
        kl_free_mode=kl_free_mode,
        reward_layers=reward_layers,
        reward_units=reward_units,
        reward_twohot_space=reward_twohot_space,
        straight_through=straight_through,
        backbone=checkpoint_backbone,
    )
    state = {
        key[len(_RSSM_PREFIX):]: value
        for key, value in checkpoint["state"]["policy"].items()
        if key.startswith(_RSSM_PREFIX)
    }
    result = world_model.load_state_dict(state, strict=False)
    new_head_prefixes = tuple(f"{name}." for name in ("reward_head", "cont_head"))
    unexpected_missing = [key for key in result.missing_keys if not key.startswith(new_head_prefixes)]
    if unexpected_missing or result.unexpected_keys:
        raise ValueError(
            "Pretrained RSSM does not match the rebuilt architecture: "
            f"missing={unexpected_missing[:8]}, unexpected={result.unexpected_keys[:8]}."
        )
    return world_model


def encoder_fingerprint(encoder: nn.Module) -> float:
    """A float64 checksum of the encoder weights; ties a retrieval DB to the
    exact encoder that produced its embeddings."""
    with torch.no_grad():
        return float(sum(p.detach().double().pow(2).sum().item() for p in encoder.parameters()))


@torch.no_grad()
def retrieval_embedding(world_model: RSSM, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """The retrieval key of a batch of frames: the flattened encoder output in
    float32 (no autocast), as in the official ``prepare_dataset_kv.py``."""
    return world_model.encode(obs).float()


def load_retrieval_db(path: str | Path) -> dict[str, Any]:
    db = torch.load(Path(path), map_location="cpu", weights_only=False)
    if db.get("metadata", {}).get("format") != RETRIEVAL_DB_FORMAT:
        raise ValueError(f"{path} is not a {RETRIEVAL_DB_FORMAT} retrieval DB.")
    return db


def retrieve_nearest_episodes(
    query_fn: Callable[[], torch.Tensor],
    embedding: torch.Tensor,
    action_dims: torch.Tensor,
    action_dim: int,
    *,
    num_episodes: int,
    max_iterations: int = 10,
    per_iteration: int = 500,
) -> tuple[list[int], int]:
    """Accumulates the nearest (L2) DB entries whose action dim equals
    ``action_dim``. Each iteration calls ``query_fn()`` for a fresh ``(1, D)``
    or ``(D,)`` query and adds up to ``per_iteration`` nearest entries not
    already chosen. Stops at ``num_episodes``, after ``max_iterations``, or
    when every eligible entry is chosen. Returns ``(DB indices in selection
    order, iterations used)``."""
    candidates = (action_dims == action_dim).nonzero(as_tuple=True)[0]
    if candidates.numel() == 0:
        raise ValueError(f"The retrieval DB has no episode with action dim {action_dim}.")
    candidate_embedding = embedding[candidates].float()
    taken = torch.zeros(candidates.numel(), dtype=torch.bool, device=candidate_embedding.device)
    selected: list[int] = []
    iterations = 0
    while len(selected) < num_episodes and iterations < max_iterations and not bool(taken.all()):
        iterations += 1
        query = query_fn().to(candidate_embedding.device, torch.float32).reshape(1, -1)
        distance = torch.cdist(query, candidate_embedding).squeeze(0)
        distance[taken] = math.inf
        count = min(num_episodes - len(selected), per_iteration, int((~taken).sum()))
        nearest = torch.topk(distance, count, largest=False).indices
        taken[nearest] = True
        selected.extend(candidates[nearest.to(candidates.device)].tolist())
    return selected, iterations


# ----------------------------------------------------------------------
# Actor-critic pieces
# ----------------------------------------------------------------------


def dv2_lambda_return(reward: torch.Tensor, value: torch.Tensor, *, disc: float, lam: float) -> torch.Tensor:
    """DreamerV2 / NCRL lambda-return over ``H + 1`` imagined states
    (``reward``/``value``: ``(H + 1, N, 1)``), official
    ``lambda_return(reward[:-1], value[:-1], disc, bootstrap=value[-1], lam)``:

        target[t] = reward[t] + disc * ((1 - lam) * value[t+1] + lam * target[t+1]),
        target[H] = value[H],   t = 0..H-1   (reward[t] of the SAME state)

    ``rl_garden.networks.returns.lambda_return`` reads ``reward[t+1]`` for
    target ``t`` (the DreamerV3 index), so the rewards are shifted down by one
    row before the call. Returns ``(H, N, 1)``."""
    shifted_reward = torch.cat([torch.zeros_like(reward[:1]), reward[:-1]], dim=0)
    zeros = torch.zeros_like(reward)
    return lambda_return(shifted_reward, value, value, last=zeros, term=zeros, disc=disc, lam=lam)


@contextlib.contextmanager
def parameters_frozen(module: nn.Module) -> Iterator[None]:
    """Temporarily ``requires_grad_(False)`` on every parameter of ``module``
    (restored on exit). Graphs built inside still differentiate through the
    module's activations, but never accumulate into its parameters."""
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


class MaskedBoundedNormalHead(MLPHead):
    """``bounded_normal`` actor head (mean ``tanh``, std in ``[min_std,
    max_std]``) whose distribution is masked to the real action dims
    (``masked_bounded_normal``: mean, samples, entropy and log-prob)."""

    def __init__(self, inp_dim: int, mask: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(inp_dim, mask.numel(), output="bounded_normal", **kwargs)
        self.register_buffer("action_mask", mask.float().clone())

    def forward(self, x: torch.Tensor) -> MaskedBoundedNormal:
        normal = super().forward(x).base_dist
        return masked_bounded_normal(normal.loc, normal.scale, self.action_mask)


class MaskedTruncatedNormalHead(NCRLHead):
    """NCRL-backbone actor: official ``GaussianPolicy`` with
    ``policy_dist=truncated_normal`` (mean ``tanh``, std in ``[min_std,
    max_std]``), masked to the real action dims (``MaskedTruncatedNormal``)."""

    def __init__(self, deter_dim: int, stoch_dim: int, mask: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(deter_dim, stoch_dim, mask.numel(), output="bounded_normal", **kwargs)
        self.register_buffer("action_mask", mask.float().clone())

    def forward(self, x: torch.Tensor) -> MaskedTruncatedNormal:
        normal = super().forward(x).base_dist
        return MaskedTruncatedNormal(normal.loc, normal.scale, self.action_mask)


class NCRLPolicy(DreamerPolicy):
    """``DreamerPolicy`` with a masked actor over the padded action space and
    a symlog two-hot critic (NCRL's ``SymTwoHot``; the slow critic is rebuilt
    as its copy). With the ``"dreamer_v3"`` RSSM backbone the heads are
    ``MLPHead``s and the actor a masked Normal; with ``"ncrl"`` they are
    ``NCRLHead``s and the actor a masked TruncatedNormal (official
    ``finetune.yaml``)."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        world_model: RSSM,
        *,
        mask: torch.Tensor,
        actor_layers: int = 3,
        critic_layers: int = 3,
        units: int = 512,
        reward_bins: int = 255,
        min_std: float = 0.1,
        max_std: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            world_model,
            actor_layers=actor_layers,
            critic_layers=critic_layers,
            units=units,
            reward_bins=reward_bins,
            min_std=min_std,
            max_std=max_std,
            **kwargs,
        )
        if world_model.backbone == "ncrl":
            deter_dim = world_model.latent_dim - world_model.flat_stoch
            self.actor = MaskedTruncatedNormalHead(
                deter_dim, world_model.flat_stoch, mask, layers=actor_layers, units=units,
                blocks=world_model.blocks, min_std=min_std, max_std=max_std,
            )
            self.critic = NCRLHead(
                deter_dim, world_model.flat_stoch, reward_bins, output="symexp_twohot", layers=critic_layers,
                units=units, blocks=world_model.blocks, twohot_space="symlog",
            )
            self._rebuild_slow_critic()
            return
        self.actor = MaskedBoundedNormalHead(
            world_model.latent_dim,
            mask,
            layers=actor_layers,
            units=units,
            outscale=0.01,
            min_std=min_std,
            max_std=max_std,
        )
        self.critic = MLPHead(
            world_model.latent_dim,
            reward_bins,
            output="symexp_twohot",
            layers=critic_layers,
            units=units,
            outscale=0.0,
            twohot_space="symlog",
        )
        self._rebuild_slow_critic()

    def _rebuild_slow_critic(self) -> None:
        self.slow_critic = copy.deepcopy(self.critic)
        for parameter in self.slow_critic.parameters():
            parameter.requires_grad_(False)
        self.slow_critic.eval()


def _clip_decay_step(
    optimizer: torch.optim.Optimizer, params: list[nn.Parameter], grad_clip: float, weight_decay: float
) -> torch.Tensor:
    """NCRL's optimizer step (``dreamer_utils.py:183-244``): grad-norm clip,
    multiplicative weight decay, then the Adam step."""
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    if weight_decay:
        with torch.no_grad():
            for param in params:
                param.mul_(1.0 - weight_decay)
    optimizer.step()
    return grad_norm


class NCRL(DreamerV3):
    _compatible_checkpoint_algorithms = ("NCRL",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        pretrained_world_model: str,
        backbone: Literal["dreamer_v3", "ncrl"] = "dreamer_v3",
        retrieval_db: Optional[str] = None,
        prior_dataset_dir: Optional[str] = None,
        online_ratio: float = 0.75,
        num_retrieve_episodes: int = 500,
        max_retrieve_iterations: int = 10,
        use_guidance: bool = True,
        bc_train_steps: int = 150_000,
        bc_batch_size: int = 256,
        bc_lr: float = 1e-4,
        guidance_schedule: tuple[str, float, float, int] = ("linear", 1.0, 0.1, 100_000),
        episode_frames: int = 1000,
        action_repeat: int = 2,
        warmup_frames: int = 15_000,
        learning_starts: Optional[int] = None,
        imag_horizon: int = 16,
        discount: float = 0.99,
        lam: float = 0.95,
        actor_ent: float = 1e-4,
        model_lr: float = 1e-4,
        reward_lr: float = 1e-4,
        actor_lr: float = 8e-5,
        critic_lr: float = 8e-5,
        adam_eps: float = 1e-5,
        weight_decay: float = 1e-6,
        grad_clip: float = 100.0,
        slow_target_fraction: float = 0.02,
        ac_units: int = 512,
        actor_layers: int = 3,
        critic_layers: int = 3,
        reward_layers: int = 2,
        reward_units: int = 512,
        padded_action_dim: Optional[int] = None,
        reward_bins: int = 255,
        kl_free: float = 1.0,
        kl_free_mode: Literal["per_element", "mean"] = "mean",
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        recon_scale: float = 1.0,
        batch_size: int = 16,
        batch_length: int = 64,
        train_ratio: float = 256.0,
        buffer_size: int = 250_000,
        buffer_device: str = "cuda",
        compute_dtype: Optional[Literal["float32", "bfloat16"]] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 5_000,
        num_eval_steps: int = 500,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        if not pretrained_world_model:
            raise ValueError("NCRL needs pretrained_world_model (an RSSMPretrain checkpoint); train from scratch with DreamerV3.")
        if save_replay_buffer:
            raise NotImplementedError(
                "NCRL does not checkpoint its replay buffer: the prior half is rebuilt from the "
                "saved retrieved-episode ids, the online half is not saved."
            )
        if not 0.0 < online_ratio <= 1.0:
            raise ValueError(f"online_ratio must be in (0, 1], got {online_ratio}.")
        if online_ratio < 1.0 and not retrieval_db:
            raise ValueError("online_ratio < 1 (experience rehearsal) needs retrieval_db.")
        kind, start_prob, end_prob, decay_steps = guidance_schedule
        if kind != "linear" or decay_steps <= 0:
            raise ValueError(f"guidance_schedule must be ('linear', start, end, steps>0), got {guidance_schedule}.")

        checkpoint = load_rssm_pretrain_checkpoint(pretrained_world_model)
        hyper = checkpoint["metadata"]["hyperparameters"]
        if backbone != pretrained_backbone(checkpoint):
            raise ValueError(
                f"{pretrained_world_model} was pretrained with backbone {pretrained_backbone(checkpoint)!r}, "
                f"but NCRL got backbone={backbone!r}; pass --backbone {pretrained_backbone(checkpoint)}."
            )
        if padded_action_dim is not None and padded_action_dim != int(hyper["padded_action_dim"]):
            raise ValueError(
                f"padded_action_dim={padded_action_dim} differs from the pretrained RSSM's "
                f"{hyper['padded_action_dim']} (its action input width is fixed)."
            )

        # Set before super().__init__(), which calls _setup_model().
        self.pretrained_world_model = str(pretrained_world_model)
        self.backbone = backbone
        self.retrieval_db = retrieval_db
        self.prior_dataset_dir = prior_dataset_dir
        self.online_ratio = online_ratio
        self.num_retrieve_episodes = num_retrieve_episodes
        self.max_retrieve_iterations = max_retrieve_iterations
        self.use_guidance = use_guidance
        self.guidance_enabled = use_guidance and online_ratio < 1.0
        self.bc_train_steps = bc_train_steps
        self.bc_batch_size = bc_batch_size
        self.bc_lr = bc_lr
        self.guidance_schedule = (str(kind), float(start_prob), float(end_prob), int(decay_steps))
        self.episode_frames = episode_frames
        self.action_repeat = action_repeat
        self.episode_steps = episode_frames // action_repeat
        self.warmup_frames = warmup_frames
        self.warmup_steps = max(warmup_frames // action_repeat, 5 * self.episode_steps)
        self.discount = discount
        self.model_lr = model_lr
        self.reward_lr = reward_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.ac_units = ac_units
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.reward_layers = reward_layers
        self.reward_units = reward_units
        self.kl_free_mode = kl_free_mode
        self.padded_action_dim = int(hyper["padded_action_dim"])
        self._pretrained_checkpoint = checkpoint
        if learning_starts is None:
            learning_starts = 10 * self.episode_steps

        super().__init__(
            env=env,
            eval_env=eval_env,
            stoch=int(hyper["stoch"]),
            unimix=float(hyper["unimix"]),
            blocks=int(hyper["blocks"]),
            obs_layers=int(hyper["obs_layers"]),
            img_layers=int(hyper["img_layers"]),
            dyn_layers=int(hyper["dyn_layers"]),
            decoder_layers=int(hyper["decoder_layers"]),
            reward_bins=reward_bins,
            kl_free=kl_free,
            discount_horizon=1.0 / (1.0 - discount),
            batch_size=batch_size,
            batch_length=batch_length,
            train_ratio=train_ratio,
            imag_horizon=imag_horizon,
            lam=lam,
            act_entropy=actor_ent,
            dyn_scale=dyn_scale,
            rep_scale=rep_scale,
            recon_scale=recon_scale,
            slow_target_fraction=slow_target_fraction,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            compute_dtype=compute_dtype,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            rssm_size=RSSMSize(**hyper["rssm_size"]),
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=False,
            save_final_checkpoint=save_final_checkpoint,
        )
        self.size = str(hyper["size"])
        del self._pretrained_checkpoint

        # DreamerV3 sets training_freq = num_envs, so the base loop's
        # max(1, int(training_freq * utd)) rounds utd < 1/num_envs up to one
        # update per env step (4x too many for 256 / (16 * 64) with one env).
        # Collect ~1 / (num_envs * utd) env steps per iteration instead; the
        # ratio is exact when num_envs * utd or its inverse is an integer.
        steps_per_env = max(1, round(1.0 / (self.num_envs * self.utd)))
        self.steps_per_env = steps_per_env
        self.training_freq = self.num_envs * steps_per_env
        self.grad_steps_per_iteration = max(1, round(self.training_freq * self.utd))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        self._resolve_encoder_sharing(self.env.single_observation_space)
        checkpoint = self._pretrained_checkpoint
        observation_space = self.env.single_observation_space
        expected_space = pretrained_observation_space(checkpoint)
        actual_shapes = {key: tuple(space.shape) for key, space in observation_space.spaces.items()}
        expected_shapes = {key: tuple(space.shape) for key, space in expected_space.spaces.items()}
        if actual_shapes != expected_shapes:
            raise ObservationContractError(
                f"NCRL's env observations {actual_shapes} must match the pretrained encoder's "
                f"{expected_shapes} exactly (image only: drop the state key, e.g. "
                "--obs.no-state --obs.rgb default --obs.image_size 64 64)."
            )
        (self._image_key,) = expected_shapes
        self._image_shape = expected_shapes[self._image_key]

        world_model = rssm_from_pretrain_checkpoint(
            checkpoint,
            backbone=self.backbone,
            heads=("reward",),
            reward_detached=True,
            reward_bins=self.reward_bins,
            kl_free=self.kl_free,
            kl_free_mode=self.kl_free_mode,
            reward_layers=self.reward_layers,
            reward_units=self.reward_units,
        )
        self._env_action_dim = int(np.prod(self.env.single_action_space.shape))
        self._action_mask = action_mask(self._env_action_dim, self.padded_action_dim)
        self._padded_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.padded_action_dim,), dtype=np.float32)
        self.policy = NCRLPolicy(
            observation_space,
            self._padded_action_space,
            world_model,
            mask=self._action_mask,
            actor_layers=self.actor_layers,
            critic_layers=self.critic_layers,
            units=self.ac_units,
            reward_bins=self.reward_bins,
            slow_target_fraction=self.slow_target_fraction,
        ).to(self.device)
        world_model = self.policy.world_model

        self._reward_params = list(world_model.reward_head.parameters())
        reward_ids = {id(p) for p in self._reward_params}
        self._model_params = [p for p in world_model.parameters() if id(p) not in reward_ids]
        self._actor_params = list(self.policy.actor.parameters())
        self._critic_params = list(self.policy.critic.parameters())
        self.model_optimizer = torch.optim.Adam(self._model_params, lr=self.model_lr, eps=self.adam_eps)
        self.reward_optimizer = torch.optim.Adam(self._reward_params, lr=self.reward_lr, eps=self.adam_eps)
        self.actor_optimizer = torch.optim.Adam(self._actor_params, lr=self.actor_lr, eps=self.adam_eps)
        self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=self.critic_lr, eps=self.adam_eps)

        self._carry_spec = {"deter": (self._rssm_size.deter,), "stoch": (self.stoch * self._rssm_size.discrete,)}
        online = SequenceReplayBuffer(
            observation_space=observation_space,
            action_space=self._padded_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            cross_episode=True,
            priority=False,
            horizon=self.batch_length,
            carry_spec=self._carry_spec,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )
        self.replay_buffer = RehearsalSequenceBuffer(online, prior=None, online_ratio=self.online_ratio)

        self._retrieval: Optional[dict[str, Any]] = None
        if self.online_ratio < 1.0:
            db = load_retrieval_db(self.retrieval_db)
            fingerprint = encoder_fingerprint(world_model.encoder)
            if not math.isclose(db["metadata"]["encoder_fingerprint"], fingerprint, rel_tol=1e-9):
                raise ValueError(
                    f"Retrieval DB {self.retrieval_db} was built with a different encoder "
                    f"(DB checkpoint {db['metadata']['checkpoint']!r}); rebuild it from "
                    f"{self.pretrained_world_model}."
                )
            self._retrieval = db
            if self.prior_dataset_dir is None:
                self.prior_dataset_dir = db["metadata"]["dataset_dir"]

        self.bc_guide: Optional[BCGuide] = (
            BCGuide(self._image_shape, self._action_mask).to(self.device) if self.guidance_enabled else None
        )
        self._bc_trained = False
        self._retrieved_episode_keys: list[tuple[str, int]] = []
        self._num_guided_steps = torch.zeros((), dtype=torch.long, device=self.device)

    def _make_prior_buffer(self, num_steps: int) -> SequenceReplayBuffer:
        return SequenceReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self._padded_action_space,
            num_envs=1,
            buffer_size=max(num_steps, self.batch_length + 2),
            cross_episode=True,
            priority=False,
            horizon=self.batch_length,
            carry_spec=self._carry_spec,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    # ------------------------------------------------------------------
    # Retrieval + BC (once, before the first rollout)
    # ------------------------------------------------------------------

    def _on_training_start(self, total_timesteps: int) -> None:
        super()._on_training_start(total_timesteps)
        self._prepare_prior_and_guidance()

    def _prepare_prior_and_guidance(self) -> None:
        """Runs after the pretrained weights are loaded and before any update,
        so the retrieval encoder equals the DB's. A resumed run reuses the
        checkpointed episode ids and BC weights instead of recomputing them."""
        if self.online_ratio < 1.0 and self.replay_buffer.prior is None:
            keys = self._retrieved_episode_keys or self._retrieve_prior_episodes()
            episodes = read_dataset_episodes(self.prior_dataset_dir, keys)
            self.replay_buffer.prior = self._make_prior_buffer(sum(ep.length for ep in episodes))
            written = self.replay_buffer.load_prior_episodes(episodes)
            self._retrieved_episode_keys = keys
            if self.std_log:
                print(f"[ncrl] prior buffer: {len(keys)} episodes, {written} steps", flush=True)
        if self.guidance_enabled and not self._bc_trained:
            losses = train_bc_guide(
                self.bc_guide,
                self.replay_buffer.prior,
                steps=self.bc_train_steps,
                batch_size=self.bc_batch_size,
                lr=self.bc_lr,
                log_every=10_000 if self.std_log else None,
            )
            self._bc_trained = True
            if self.logger is not None and losses:
                self.logger.add_summary("ncrl/bc_final_loss", losses[-1])

    def _retrieve_prior_episodes(self) -> list[tuple[str, int]]:
        """``(embodiment, episode_id)`` of the retrieved prior episodes, in
        selection order. Each iteration resets the env for a new query frame."""
        db = self._retrieval
        world_model = self.world_model

        def query() -> torch.Tensor:
            obs, _ = self.env.reset()
            obs = self._obs_to_policy_device(obs)
            return retrieval_embedding(world_model, {self._image_key: obs[self._image_key][:1]})

        indices, iterations = retrieve_nearest_episodes(
            query,
            db["embedding"].to(self.device),
            torch.as_tensor(db["action_dim"]).to(self.device),
            self._env_action_dim,
            num_episodes=self.num_retrieve_episodes,
            max_iterations=self.max_retrieve_iterations,
        )
        if self.std_log:
            print(
                f"[ncrl] retrieved {len(indices)}/{self.num_retrieve_episodes} episodes "
                f"in {iterations} iteration(s)",
                flush=True,
            )
        return [(str(db["embodiment"][i]), int(db["episode_id"][i])) for i in indices]

    # ------------------------------------------------------------------
    # Rollout: action source per env (warmup / guidance window / actor)
    # ------------------------------------------------------------------

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        n = self.num_envs
        self._episode_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self._guidance_start = torch.zeros(n, device=self.device)
        self._guidance_width = torch.zeros(n, device=self.device)
        self._bc_frames: Optional[torch.Tensor] = None

    def _explore_action(self, obs) -> torch.Tensor:
        action = torch.zeros(self.num_envs, self.padded_action_dim, device=self.device)
        action[:, : self._env_action_dim] = super()._explore_action(obs)
        return action

    def guidance_probability(self) -> float:
        _, start, end, steps = self.guidance_schedule
        return start + (end - start) * min(self._global_step / steps, 1.0)

    def _update_bc_frames(self, frame: torch.Tensor, is_first: torch.Tensor) -> None:
        """3-frame queue per env; refilled with the first frame of an episode."""
        filled = frame.unsqueeze(1).expand(-1, self.bc_guide.frame_stack, *frame.shape[1:])
        if self._bc_frames is None:
            self._bc_frames = filled.clone()
            return
        shifted = torch.cat([self._bc_frames[:, 1:], frame.unsqueeze(1)], dim=1)
        self._bc_frames = torch.where(is_first.reshape(-1, 1, 1, 1, 1), filled, shifted)

    def _resample_guidance_windows(self, is_first: torch.Tensor) -> None:
        """At each episode start: with probability ``p`` a window of width
        ``W = L/10 * i`` (``i`` uniform in 1..10) starting at ``S`` uniform in
        ``[0, L - W]``; otherwise no window (``W = 0``)."""
        n, length = self.num_envs, float(self.episode_steps)
        guided = torch.rand(n, device=self.device) < self.guidance_probability()
        width = (length / 10.0) * torch.randint(1, 11, (n,), device=self.device).float()
        start = torch.floor(torch.rand(n, device=self.device) * (length - width + 1.0))
        width = torch.where(guided, width, torch.zeros_like(width))
        start = torch.where(guided, start, torch.zeros_like(start))
        self._guidance_width = torch.where(is_first, width, self._guidance_width)
        self._guidance_start = torch.where(is_first, start, self._guidance_start)

    def _action_source(self) -> torch.Tensor:
        n = self.num_envs
        if self._global_step <= self.warmup_steps:
            source = _ACTION_FROM_BC if self.guidance_enabled else _ACTION_FROM_RANDOM
            return torch.full((n,), source, dtype=torch.long, device=self.device)
        if not self.guidance_enabled:
            return torch.full((n,), _ACTION_FROM_ACTOR, dtype=torch.long, device=self.device)
        step = self._episode_step.float()
        in_window = (self._guidance_start < step) & (step < self._guidance_start + self._guidance_width)
        return torch.where(in_window, _ACTION_FROM_BC, _ACTION_FROM_ACTOR)

    def _rollout_action(self, obs, learning_has_started: bool):
        del learning_has_started  # warmup, not learning_starts, decides random/BC actions
        obs_device = self._obs_to_policy_device(obs)
        is_first = self._rollout_is_first
        self._episode_step = torch.where(is_first, torch.zeros_like(self._episode_step), self._episode_step)
        if self.guidance_enabled:
            self._update_bc_frames(obs_device[self._image_key], is_first)
            self._resample_guidance_windows(is_first)

        with torch.no_grad():
            with torch.autocast(**self._autocast_kwargs):
                actor_action, new_state = self.policy.predict(
                    obs_device, self._rollout_state, is_first, deterministic=False
                )
            source = self._action_source()
            action = actor_action.float()
            if self.guidance_enabled:
                bc_action = self.bc_guide.act(self._bc_frames).clamp(-1.0, 1.0)
                action = torch.where((source == _ACTION_FROM_BC).unsqueeze(1), bc_action, action)
                self._num_guided_steps += (source == _ACTION_FROM_BC).sum()
            else:
                action = torch.where(
                    (source == _ACTION_FROM_RANDOM).unsqueeze(1), self._explore_action(obs), action
                )

        new_state = {**new_state, "prev_action": action}
        self._rollout_state = new_state
        self._episode_step = self._episode_step + 1
        action_context = {"new_state": new_state, "action_source": source}
        return action, action[:, : self._env_action_dim], action_context

    def _eval_action_and_critic_action(self, obs):
        action, critic_action = super()._eval_action_and_critic_action(obs)
        return action[:, : self._env_action_dim], critic_action

    # ------------------------------------------------------------------
    # Gradient step
    # ------------------------------------------------------------------

    def _update_model(self, batch) -> tuple[dict[str, float], Any]:
        world_model = self.world_model
        with torch.autocast(**self._autocast_kwargs):
            losses, posterior = world_model.model_loss(batch)
        losses = {k: v.float() for k, v in losses.items()}
        recon_loss = sum(v for k, v in losses.items() if k not in _KL_AND_REWARD_KEYS)
        model_loss = self.dyn_scale * losses["dyn"] + self.rep_scale * losses["rep"] + self.recon_scale * recon_loss

        # The reward loss reads detached features (reward_detached=True), so
        # one backward fills disjoint gradient sets: model params from
        # model_loss, reward-head params from losses["rew"].
        self.model_optimizer.zero_grad(set_to_none=True)
        self.reward_optimizer.zero_grad(set_to_none=True)
        (model_loss + losses["rew"]).backward()
        model_grad_norm = _clip_decay_step(self.model_optimizer, self._model_params, self.grad_clip, self.weight_decay)
        reward_grad_norm = _clip_decay_step(
            self.reward_optimizer, self._reward_params, self.grad_clip, self.weight_decay
        )

        carry = {
            "deter": posterior["deter"].detach(),
            "stoch": posterior["stoch"].detach().reshape(*posterior["stoch"].shape[:2], -1),
        }
        self.replay_buffer.write_back_carry(batch.indices, carry)
        metrics = {f"{k}_loss": float(v.detach()) for k, v in losses.items()}
        metrics.update(
            recon_loss=float(recon_loss.detach()),
            model_loss=float(model_loss.detach()),
            model_grad_norm=float(model_grad_norm),
            reward_grad_norm=float(reward_grad_norm),
        )
        return metrics, posterior

    def _imagination_losses(self, batch, posterior) -> dict[str, Any]:
        """Pure (no backward/step): DreamerV2 dynamics-backprop actor loss and
        twohot critic loss from an imagined rollout of every posterior row.

        Gradient paths: actor -> rsample action -> RSSM.step -> reward head and
        slow critic -> lambda-return -> actor loss. RSSM parameters are frozen
        during the rollout and the slow critic never requires grad, so only
        actor parameters receive actor-loss gradients. The actor reads
        detached features (official ``policy(sg(deter), sg(stoch))``). The
        critic loss reads detached features and a detached target.
        """
        del batch
        world_model = self.world_model
        actor = self.policy.actor
        horizon, batch_size = posterior["deter"].shape[:2]
        start = {
            key: posterior[key].detach().reshape(horizon * batch_size, *posterior[key].shape[2:])
            for key in ("deter", "stoch", "logits")
        }

        def policy_fn(state):
            return actor(world_model.features(state).detach()).rsample()

        with parameters_frozen(world_model), torch.autocast(**self._autocast_kwargs):
            traj = imagine(world_model, policy_fn, start, horizon=self.imag_horizon, grad=True)
            feat = world_model.features(traj.states)  # (H + 1, N, latent)
            reward = world_model.reward_value(traj.states)  # (H + 1, N, 1)
            slow_value = self.policy.slow_critic(feat).mode  # (H + 1, N, 1)
            target = dv2_lambda_return(reward, slow_value, disc=self.discount, lam=self.lam)  # (H, N, 1)
            weight = (self.discount ** torch.arange(self.imag_horizon + 1, device=feat.device, dtype=torch.float32))
            weight = weight.reshape(-1, 1, 1)  # cumprod([1, disc, disc, ...]), constant (no continue head)

            entropy = actor(feat[:-2].detach()).entropy().unsqueeze(-1)  # (H - 1, N, 1)
            objective = target[1:] + self.act_entropy * entropy
            actor_loss = -(weight[:-2] * objective).float().mean()

            critic_dist = self.policy.critic(feat[:-1].detach())
            critic_loss = -(critic_dist.log_prob(target.detach()) * weight[:-1].squeeze(-1)).float().mean()

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "target": target,
            "reward": reward,
            "slow_value": slow_value,
            "entropy": entropy,
        }

    def _update_actor_critic(self, batch, posterior) -> dict[str, float]:
        imag = self._imagination_losses(batch, posterior)
        self.actor_optimizer.zero_grad(set_to_none=True)
        imag["actor_loss"].backward()
        actor_grad_norm = _clip_decay_step(self.actor_optimizer, self._actor_params, self.grad_clip, self.weight_decay)
        self.critic_optimizer.zero_grad(set_to_none=True)
        imag["critic_loss"].backward()
        critic_grad_norm = _clip_decay_step(
            self.critic_optimizer, self._critic_params, self.grad_clip, self.weight_decay
        )
        return {
            "actor_loss": float(imag["actor_loss"].detach()),
            "critic_loss": float(imag["critic_loss"].detach()),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "imag_target_mean": float(imag["target"].detach().float().mean()),
            "imag_reward_mean": float(imag["reward"].detach().float().mean()),
            "actor_entropy": float(imag["entropy"].detach().float().mean()),
            "guided_steps": float(self._num_guided_steps),
        }

    # ------------------------------------------------------------------
    # Checkpointing (DreamerV3's versions handle its LaProp scheduler, which
    # NCRL does not have -- go to ModelBasedAlgorithm's directly)
    # ------------------------------------------------------------------

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("model_optimizer", "reward_optimizer", "actor_optimizer", "critic_optimizer")

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super(DreamerV3, self)._extra_checkpoint_state(),
            "bc_guide": self.bc_guide.state_dict() if self.bc_guide is not None and self._bc_trained else None,
            "retrieved_episodes": [list(key) for key in self._retrieved_episode_keys],
            "guided_steps": int(self._num_guided_steps),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super(DreamerV3, self)._load_extra_checkpoint_state(state)
        bc_state = state.get("bc_guide")
        if bc_state is not None and self.bc_guide is not None:
            self.bc_guide.load_state_dict(bc_state)
            self.bc_guide.eval()
            for parameter in self.bc_guide.parameters():
                parameter.requires_grad_(False)
            self._bc_trained = True
        self._retrieved_episode_keys = [(str(emb), int(ep)) for emb, ep in state.get("retrieved_episodes", [])]
        self._num_guided_steps.fill_(int(state.get("guided_steps", 0)))

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super(DreamerV3, self)._checkpoint_metadata(),
            "pretrained_world_model": self.pretrained_world_model,
            "backbone": self.backbone,
            "retrieval_db": self.retrieval_db,
            "prior_dataset_dir": self.prior_dataset_dir,
            "size": self.size,
            "rssm_size": dataclasses.asdict(self._rssm_size),
            "stoch": self.stoch,
            "unimix": self.unimix,
            "blocks": self.blocks,
            "obs_layers": self.obs_layers,
            "img_layers": self.img_layers,
            "dyn_layers": self.dyn_layers,
            "decoder_layers": self.decoder_layers,
            "padded_action_dim": self.padded_action_dim,
            "env_action_dim": self._env_action_dim,
            "reward_bins": self.reward_bins,
            "kl_free": self.kl_free,
            "kl_free_mode": self.kl_free_mode,
            "online_ratio": self.online_ratio,
            "num_retrieve_episodes": self.num_retrieve_episodes,
            "max_retrieve_iterations": self.max_retrieve_iterations,
            "use_guidance": self.use_guidance,
            "guidance_enabled": self.guidance_enabled,
            "bc_train_steps": self.bc_train_steps,
            "bc_batch_size": self.bc_batch_size,
            "bc_lr": self.bc_lr,
            "guidance_schedule": list(self.guidance_schedule),
            "episode_frames": self.episode_frames,
            "action_repeat": self.action_repeat,
            "warmup_frames": self.warmup_frames,
            "warmup_steps": self.warmup_steps,
            "batch_length": self.batch_length,
            "train_ratio": self.train_ratio,
            "imag_horizon": self.imag_horizon,
            "discount": self.discount,
            "lam": self.lam,
            "actor_ent": self.act_entropy,
            "dyn_scale": self.dyn_scale,
            "rep_scale": self.rep_scale,
            "recon_scale": self.recon_scale,
            "model_lr": self.model_lr,
            "reward_lr": self.reward_lr,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "adam_eps": self.adam_eps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "slow_target_fraction": self.slow_target_fraction,
            "ac_units": self.ac_units,
            "actor_layers": self.actor_layers,
            "critic_layers": self.critic_layers,
            "reward_layers": self.reward_layers,
            "reward_units": self.reward_units,
            "compute_dtype": self.compute_dtype,
            "encoder_sharing": self.encoder_sharing,
            "encoder_sharing_origin": self.encoder_sharing_origin,
        }
