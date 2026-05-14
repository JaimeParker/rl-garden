"""Base-action provider interface and ACT checkpoint wrapper."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.models.act.config import (
    ACTCheckpointSpec,
    ACTConfig,
    infer_act_config,
    resolve_act_checkpoint_path,
    select_act_state_dict,
)

StateObsGetter = Callable[[], torch.Tensor]


class BaseActionProvider(Protocol):
    """ResidualSAC base-policy contract.

    Providers return raw env-space actions with shape ``(num_envs, action_dim)``.
    ``ResidualSAC`` handles action scaling into its normalized residual
    coordinates before writing replay or evaluating the critic.
    """

    def __call__(self, obs) -> torch.Tensor:
        ...

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        ...


def _install_vendored_act_alias() -> None:
    """Expose the copied ACT package as top-level ``act`` for vendored imports."""

    vendored = importlib.import_module("rl_garden.models.act.act")
    vendored_detr = importlib.import_module("rl_garden.models.act.act.detr")
    sys.modules.setdefault("act", vendored)
    sys.modules.setdefault("act.detr", vendored_detr)


_install_vendored_act_alias()

from rl_garden.models.act.act.detr.backbone import (  # noqa: E402
    BackboneBase,
    FrozenBatchNorm2d,
    Joiner,
)
from rl_garden.models.act.act.detr.detr_vae import (  # noqa: E402
    DETRVAE,
    build_encoder,
)
from rl_garden.models.act.act.detr.position_encoding import (  # noqa: E402
    build_position_encoding,
)
from rl_garden.models.act.act.detr.transformer import (  # noqa: E402
    build_transformer,
)


def _torchvision_resnet(name: str, *, dilation: bool):
    import torchvision

    kwargs = dict(
        replace_stride_with_dilation=[False, False, dilation],
        norm_layer=FrozenBatchNorm2d,
    )
    try:
        return getattr(torchvision.models, name)(weights=None, **kwargs)
    except TypeError:
        return getattr(torchvision.models, name)(pretrained=False, **kwargs)


def _build_backbone_no_download(config: ACTConfig) -> Joiner:
    backbone = _torchvision_resnet(config.backbone, dilation=config.dilation)
    if config.include_depth:
        weight = backbone.conv1.weight
        depth_weight = torch.zeros(
            weight.shape[0],
            1,
            weight.shape[2],
            weight.shape[3],
            dtype=weight.dtype,
            device=weight.device,
        )
        backbone.conv1.weight = nn.Parameter(torch.cat([weight, depth_weight], dim=1))

    num_channels = 512 if config.backbone in ("resnet18", "resnet34") else 2048
    body = BackboneBase(
        backbone,
        train_backbone=config.lr_backbone > 0,
        num_channels=num_channels,
        return_interm_layers=config.masks,
    )
    model = Joiner(body, build_position_encoding(config))
    model.num_channels = body.num_channels
    return model


def make_act_state_obs_getter(
    env: Any,
    *,
    expected_dim: int,
) -> Optional[StateObsGetter]:
    """Build an env-backed state getter for state-only ACT checkpoints.

    RGB ManiSkill observations can expose a smaller policy ``state`` than the
    full privileged ``obs_mode=state`` vector used to train ACT.  The vendored
    peg-only ACT checkpoint expects that full vector, so for the known peg env
    we reconstruct it directly from env attributes.
    """

    base_env = getattr(env, "base_env", None) or getattr(env, "unwrapped", env)
    required_attrs = (
        "agent",
        "peg",
        "peg_half_sizes",
        "box_hole_pose",
        "box_hole_radii",
    )
    if expected_dim != 43 or not all(hasattr(base_env, attr) for attr in required_attrs):
        return None

    def _getter() -> torch.Tensor:
        from mani_skill.utils import common

        state_dict = {
            "agent": {
                "qpos": base_env.agent.robot.get_qpos(),
                "qvel": base_env.agent.robot.get_qvel(),
            },
            "extra": {
                "tcp_pose": base_env.agent.tcp.pose.raw_pose,
                "peg_pose": base_env.peg.pose.raw_pose,
                "peg_half_size": base_env.peg_half_sizes,
                "box_hole_pose": base_env.box_hole_pose.raw_pose,
                "box_hole_radius": base_env.box_hole_radii,
            },
        }
        return common.flatten_state_dict(state_dict, use_torch=True)

    return _getter


class ACTPolicyModel(nn.Module):
    """Inference-only ACT model compatible with copied ACT checkpoints."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        visual: bool,
        config: ACTConfig,
    ) -> None:
        super().__init__()
        backbones = [_build_backbone_no_download(config)] if visual else None
        transformer = build_transformer(config)
        encoder = build_encoder(config)
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            action_dim=action_dim,
            num_queries=config.num_queries,
        )

    def forward(self, obs) -> torch.Tensor:
        action_seq, _ = self.model(obs)
        return action_seq


class ACTBaseActionProvider(nn.Module):
    """ACT checkpoint adapter for ``ResidualSAC`` base actions."""

    def __init__(
        self,
        policy: ACTPolicyModel,
        *,
        config: ACTConfig,
        spec: ACTCheckpointSpec,
        checkpoint_path: Path,
        norm_stats: Optional[dict[str, Any]] = None,
        state_obs_getter: Optional[StateObsGetter] = None,
        auto_state_obs_getter: bool = False,
        temporal_agg: bool = True,
        temporal_agg_k: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.policy = policy.eval()
        self.config = config
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.temporal_agg = temporal_agg
        self.temporal_agg_k = float(temporal_agg_k)
        self.device = torch.device(device)
        self._norm_stats = self._tensorize_norm_stats(norm_stats)
        self._state_obs_getter = state_obs_getter
        self._auto_state_obs_getter = auto_state_obs_getter
        self.register_buffer(
            "_rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "_rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 1, 3, 1, 1
            ),
        )
        self._step = 0
        self._history: list[tuple[int, torch.Tensor]] = []
        self._cached_action_seq: Optional[torch.Tensor] = None
        self.to(self.device)

    def bind_env(self, env: Any) -> None:
        """Rebind the env-backed full-state getter used by state-only ACT."""

        if not self._auto_state_obs_getter:
            return
        self._state_obs_getter = make_act_state_obs_getter(
            env, expected_dim=self.spec.state_dim
        )

    @staticmethod
    def _tensorize_norm_stats(
        norm_stats: Optional[dict[str, Any]]
    ) -> Optional[dict[str, torch.Tensor]]:
        if norm_stats is None:
            return None
        return {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in norm_stats.items()
            if isinstance(value, (torch.Tensor, float, int)) or hasattr(value, "__array__")
        }

    @classmethod
    def from_checkpoint(
        cls,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        ckpt_path: str | Path | None = None,
        state_dict_key: str = "ema_agent",
        env: Any | None = None,
        state_obs_getter: Optional[StateObsGetter] = None,
        temporal_agg: bool = True,
        temporal_agg_k: float = 0.01,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> "ACTBaseActionProvider":
        path = resolve_act_checkpoint_path(ckpt_path)
        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"ACT checkpoint at {path} is not a dict.")
        state_dict = select_act_state_dict(checkpoint, state_dict_key=state_dict_key)
        config, spec = infer_act_config(state_dict)
        state_obs_dim = None
        auto_state_obs_getter = False
        if state_obs_getter is None and env is not None:
            env_state_dim = cls._state_dim_from_space(observation_space)
            if env_state_dim != spec.state_dim:
                state_obs_getter = make_act_state_obs_getter(
                    env, expected_dim=spec.state_dim
                )
                if state_obs_getter is not None:
                    state_obs_dim = spec.state_dim
                    auto_state_obs_getter = True
        elif state_obs_getter is not None:
            state_obs_dim = spec.state_dim

        cls._validate_spaces(
            spec,
            observation_space,
            action_space,
            state_obs_dim=state_obs_dim,
        )

        policy = ACTPolicyModel(
            state_dim=spec.state_dim,
            action_dim=spec.action_dim,
            visual=spec.visual,
            config=config,
        )
        policy.load_state_dict(state_dict, strict=strict)
        return cls(
            policy,
            config=config,
            spec=spec,
            checkpoint_path=path,
            norm_stats=checkpoint.get("norm_stats"),
            state_obs_getter=state_obs_getter,
            auto_state_obs_getter=auto_state_obs_getter,
            temporal_agg=temporal_agg,
            temporal_agg_k=temporal_agg_k,
            device=device,
        )

    @staticmethod
    def _state_dim_from_space(observation_space: spaces.Space) -> int:
        if isinstance(observation_space, spaces.Dict):
            if "state" not in observation_space.spaces:
                raise ValueError("ACT base policy requires a 'state' observation key.")
            state_space = observation_space.spaces["state"]
        else:
            state_space = observation_space
        if not isinstance(state_space, spaces.Box) or len(state_space.shape) != 1:
            raise ValueError(f"ACT state observation must be 1D Box, got {state_space}.")
        return int(state_space.shape[0])

    @classmethod
    def _validate_spaces(
        cls,
        spec: ACTCheckpointSpec,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        *,
        state_obs_dim: Optional[int] = None,
    ) -> None:
        state_dim = state_obs_dim or cls._state_dim_from_space(observation_space)
        action_dim = int(action_space.shape[0])
        if state_dim != spec.state_dim:
            raise ValueError(
                "ACT checkpoint state dimension does not match env observation: "
                f"checkpoint={spec.state_dim}, env={state_dim}."
            )
        if action_dim != spec.action_dim:
            raise ValueError(
                "ACT checkpoint action dimension does not match env action space: "
                f"checkpoint={spec.action_dim}, env={action_dim}."
            )
        if spec.visual and not isinstance(observation_space, spaces.Dict):
            raise ValueError("Visual ACT checkpoint requires Dict observations.")
        if spec.visual and "rgb" not in observation_space.spaces:
            raise ValueError("Visual ACT checkpoint requires an 'rgb' observation key.")

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        device = next(self.policy.parameters()).device
        self.device = device
        if self._norm_stats is not None:
            self._norm_stats = {
                key: value.to(device) for key, value in self._norm_stats.items()
            }
        return module

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        del env_ids
        self._step = 0
        self._history.clear()
        self._cached_action_seq = None

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device, dtype=torch.float32)
        if self._norm_stats is None:
            return state
        if "state_mean" not in self._norm_stats or "state_std" not in self._norm_stats:
            return state
        return (state - self._norm_stats["state_mean"]) / self._norm_stats["state_std"]

    def _postprocess_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        if self._norm_stats is None:
            return action_seq
        if "action_mean" not in self._norm_stats or "action_std" not in self._norm_stats:
            return action_seq
        return action_seq * self._norm_stats["action_std"] + self._norm_stats["action_mean"]

    @staticmethod
    def _image_to_bnc_hw(image: torch.Tensor, channels_per_camera: int) -> torch.Tensor:
        if image.ndim == 5 and image.shape[2] == channels_per_camera:
            return image.contiguous()
        if image.ndim != 4:
            raise ValueError(
                "ACT visual observations must be either (B,H,W,Ctotal) or "
                f"(B,N,{channels_per_camera},H,W); got shape={tuple(image.shape)}."
            )
        batch, height, width, channels = image.shape
        if channels % channels_per_camera != 0:
            raise ValueError(
                f"Image channels {channels} are not divisible by "
                f"{channels_per_camera}."
            )
        num_cams = channels // channels_per_camera
        return (
            image.reshape(batch, height, width, num_cams, channels_per_camera)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    def _resize_camera_tensor(self, image: torch.Tensor) -> torch.Tensor:
        batch, num_cams, channels, height, width = image.shape
        if height == self.config.image_size and width == self.config.image_size:
            return image
        flat = image.reshape(batch * num_cams, channels, height, width)
        flat = F.interpolate(
            flat,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return flat.reshape(
            batch, num_cams, channels, self.config.image_size, self.config.image_size
        )

    def _prepare_visual_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rgb = self._image_to_bnc_hw(obs["rgb"].to(self.device), 3)
        rgb = self._resize_camera_tensor(rgb.float() / 255.0)
        rgb = (rgb - self._rgb_mean) / self._rgb_std

        state = self._state_from_obs(obs)
        prepared: dict[str, torch.Tensor] = {
            "state": self._normalize_state(state),
            "rgb": rgb,
        }
        if self.config.include_depth:
            if "depth" not in obs:
                raise ValueError("ACT checkpoint expects depth observations.")
            depth = self._image_to_bnc_hw(obs["depth"].to(self.device), 1)
            prepared["depth"] = self._resize_camera_tensor(depth.float())
        return prepared

    def _state_from_obs(self, obs) -> torch.Tensor:
        if self._state_obs_getter is not None:
            state = self._state_obs_getter()
        elif isinstance(obs, dict):
            state = obs["state"]
        else:
            state = obs
        if int(state.shape[-1]) != self.spec.state_dim:
            raise ValueError(
                "ACT state observation dimension mismatch at runtime: "
                f"expected={self.spec.state_dim}, got={int(state.shape[-1])}."
            )
        return state

    def _prepare_obs(self, obs):
        if isinstance(obs, dict):
            if self.spec.visual:
                return self._prepare_visual_obs(obs)
            return self._normalize_state(self._state_from_obs(obs))
        if self.spec.visual:
            raise ValueError("Visual ACT checkpoint requires dict observations.")
        return self._normalize_state(self._state_from_obs(obs))

    def _query_policy(self, obs) -> torch.Tensor:
        prepared_obs = self._prepare_obs(obs)
        action_seq = self.policy(prepared_obs)
        if action_seq.ndim != 3:
            raise RuntimeError(
                f"ACT policy must return (B, num_queries, action_dim), got {action_seq.shape}."
            )
        return self._postprocess_actions(action_seq)

    def _temporal_agg_action(self) -> torch.Tensor:
        actions: list[torch.Tensor] = []
        kept: list[tuple[int, torch.Tensor]] = []
        for query_step, action_seq in self._history:
            offset = self._step - query_step
            if 0 <= offset < action_seq.shape[1]:
                actions.append(action_seq[:, offset])
                kept.append((query_step, action_seq))
        self._history = kept
        if not actions:
            raise RuntimeError("ACT temporal aggregation has no populated actions.")
        stacked = torch.stack(actions, dim=1)
        weights = torch.exp(
            -self.temporal_agg_k
            * torch.arange(stacked.shape[1], device=stacked.device, dtype=stacked.dtype)
        )
        weights = weights / weights.sum()
        return (stacked * weights.view(1, -1, 1)).sum(dim=1)

    @torch.no_grad()
    def select_action(self, obs) -> torch.Tensor:
        if self.temporal_agg:
            action_seq = self._query_policy(obs).detach()
            self._history.append((self._step, action_seq))
            action = self._temporal_agg_action()
        else:
            if self._cached_action_seq is None or self._step % self.config.num_queries == 0:
                self._cached_action_seq = self._query_policy(obs).detach()
            action = self._cached_action_seq[:, self._step % self.config.num_queries]
        self._step += 1
        return action

    def predict(self, obs) -> torch.Tensor:
        return self.select_action(obs)

    def forward(self, obs) -> torch.Tensor:
        return self.select_action(obs)
