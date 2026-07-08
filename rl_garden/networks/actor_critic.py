from __future__ import annotations

import re
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.mlp import KernelInit, MLPResNet, create_mlp

BackboneType = Literal["mlp", "mlp_resnet"]
CriticImpl = Literal["vmap", "legacy"]


def get_actor_critic_arch(
    net_arch: Sequence[int] | dict[str, Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Resolve actor/critic hidden dims from a shared net_arch spec.

    Mirrors SB3 semantics:
      - list[int] -> same architecture for actor and critic
      - dict(pi=[...], qf=[...]) -> separate architectures
    """
    if isinstance(net_arch, dict):
        if "pi" not in net_arch or "qf" not in net_arch:
            raise ValueError("net_arch dict must contain both 'pi' and 'qf' keys.")
        return list(net_arch["pi"]), list(net_arch["qf"])

    return list(net_arch), list(net_arch)


def _build_trunk(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    backbone_type: BackboneType,
    use_layer_norm: bool,
    use_group_norm: bool,
    num_groups: int,
    dropout_rate: Optional[float],
    kernel_init: Optional[KernelInit],
    use_pnorm: bool = False,
) -> tuple[nn.Module, int]:
    """Build a feature trunk and return (module, output_dim).

    For ``backbone_type='mlp'``, returns standard create_mlp() with no output head.
    For ``backbone_type='mlp_resnet'``, ``hidden_dims`` must be a list of identical
    widths; ``hidden_dim`` is taken from the first entry and ``num_blocks`` from len.
    ``use_pnorm`` L2-normalizes the trunk's final hidden features (RLPD/WSRL ablation).
    """
    if backbone_type == "mlp":
        trunk = create_mlp(
            input_dim=input_dim,
            output_dim=-1,
            net_arch=hidden_dims,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            use_pnorm=use_pnorm,
        )
        out_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        return trunk, out_dim

    if backbone_type == "mlp_resnet":
        if len(hidden_dims) < 1:
            raise ValueError("mlp_resnet requires at least one hidden dim.")
        if any(h != hidden_dims[0] for h in hidden_dims):
            raise ValueError(
                "mlp_resnet requires identical widths across hidden_dims; "
                f"got {list(hidden_dims)!r}."
            )
        trunk = MLPResNet(
            input_dim=input_dim,
            output_dim=-1,
            hidden_dim=hidden_dims[0],
            num_blocks=len(hidden_dims),
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            use_pnorm=use_pnorm,
        )
        return trunk, hidden_dims[0]

    raise ValueError(f"Unknown backbone_type: {backbone_type!r}")


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for SAC/WSRL families."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        use_pnorm: bool = False,
        std_parameterization: Literal["exp", "uniform"] = "exp",
        log_std_mode: Literal["clamp", "tanh"] = "clamp",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if std_parameterization not in ("exp", "uniform"):
            raise ValueError(
                "std_parameterization must be 'exp' or 'uniform', "
                f"got {std_parameterization!r}"
            )
        if log_std_mode not in ("clamp", "tanh"):
            raise ValueError(
                f"log_std_mode must be 'clamp' or 'tanh', got {log_std_mode!r}"
            )

        self.std_parameterization = std_parameterization
        self.log_std_mode = log_std_mode
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        act_dim = int(np.prod(action_space.shape))
        self.trunk, trunk_dim = _build_trunk(
            features_dim,
            hidden_dims,
            backbone_type=backbone_type,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            use_pnorm=use_pnorm,
        )

        self.fc_mean = nn.Linear(trunk_dim, act_dim)
        if std_parameterization == "exp":
            self.fc_logstd = nn.Linear(trunk_dim, act_dim)
            self.log_stds = None
        else:
            self.fc_logstd = None
            self.log_stds = nn.Parameter(torch.zeros(act_dim))

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(features)
        mean = self.fc_mean(x)

        if self.std_parameterization == "exp":
            assert self.fc_logstd is not None
            raw_log_std = self.fc_logstd(x)
        else:
            assert self.log_stds is not None
            raw_log_std = self.log_stds.expand_as(mean)

        if self.log_std_mode == "tanh":
            log_std = torch.tanh(raw_log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                log_std + 1
            )
        else:
            log_std = torch.clamp(raw_log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action_log_prob(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = self._squashed_log_prob(normal, x_t, y_t)
        return action, log_prob

    def evaluate_action_log_prob(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate log-probability for externally supplied env-space actions.

        This complements ``action_log_prob()``: that method samples a new action
        from the policy and returns its log-prob, while this method scores an
        action that came from outside the policy, e.g. an offline dataset action
        used by IQL/BC-style actor losses. ``actions`` are expected to already
        be tanh-squashed and scaled into ``action_space`` bounds, so we map them
        back through inverse tanh before applying the same squashing correction
        used for sampled actions.
        """
        mean, log_std = self(features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        eps = 1e-6
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = y_t.clamp(-1.0 + eps, 1.0 - eps)
        x_t = 0.5 * (torch.log1p(y_t) - torch.log1p(-y_t))
        return self._squashed_log_prob(normal, x_t, y_t)

    def _squashed_log_prob(
        self,
        normal: torch.distributions.Normal,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return log_prob.sum(-1, keepdim=True)

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self(features)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class DiagGaussianActor(nn.Module):
    """Unsquashed diagonal Gaussian actor used by PPO-style algorithms.

    Actions are sampled in env action coordinates and may be clamped by the
    training loop before stepping the environment. Log-probabilities are always
    evaluated on the unclamped action tensor, matching SB3 and ManiSkill PPO.
    """

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        log_std_init: float = -0.5,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.trunk, trunk_dim = _build_trunk(
            features_dim,
            hidden_dims,
            backbone_type=backbone_type,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
        )
        self.mean = nn.Linear(trunk_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * log_std_init)

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_high", high)
        self.register_buffer("action_low", low)

    def forward(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(self.trunk(features))
        log_std = self.log_std.expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp())

    def action_log_prob(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self(features)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate_action_log_prob(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self(features)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).mean

    def clamp_action(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(actions, self.action_high), self.action_low)


class _QHead(nn.Module):
    """Single Q-network: trunk over (features, actions) -> scalar Q."""

    def __init__(
        self,
        features_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        *,
        backbone_type: BackboneType,
        use_layer_norm: bool,
        use_group_norm: bool,
        num_groups: int,
        dropout_rate: Optional[float],
        kernel_init: Optional[KernelInit],
        use_pnorm: bool = False,
    ) -> None:
        super().__init__()
        self.trunk, trunk_dim = _build_trunk(
            features_dim + act_dim,
            hidden_dims,
            backbone_type=backbone_type,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            use_pnorm=use_pnorm,
        )
        self.head = nn.Linear(trunk_dim, 1)

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, actions], dim=-1)
        return self.head(self.trunk(x))

    def trunk_features(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([features, actions], dim=-1)
        return self.trunk(x)


# Internal: separator used to encode dotted parameter names into nn.Module-safe
# attribute names. Dots are illegal in attribute keys, so ``trunk.0.weight``
# becomes ``trunk__0__weight``. ``_PARAM_PREFIX`` distinguishes ensemble param
# tensors from buffers in the state_dict — needed for migration.
_PARAM_SEP = "__"
_PARAM_PREFIX = "ens_p_"
_BUFFER_PREFIX = "ens_b_"


def _safe_name(dotted: str, prefix: str) -> str:
    return prefix + dotted.replace(".", _PARAM_SEP)


def _dotted_from_safe(safe: str, prefix: str) -> str:
    return safe[len(prefix) :].replace(_PARAM_SEP, ".")


class EnsembleQCritic(nn.Module):
    """Ensemble of Q(s, a) networks.

    ``critic_impl="vmap"`` uses the fused ``torch.func.vmap`` implementation.
    ``critic_impl="legacy"`` keeps the pre-vmap ``nn.ModuleList`` layout for
    training-regression ablations while preserving the same public API.

    Each critic is initialized independently before stacking — the diverse
    random init across the ensemble is preserved in both modes.

    Vmap checkpoint state_dict keys:
        ``ens_p_<dotted_path>``  (e.g. ``ens_p_trunk__0__weight``, shape
        ``(n_critics, *original_shape)``).
    Legacy checkpoints with ``q_nets.<i>.<dotted_path>`` keys are migrated
    transparently when loading into vmap mode.
    """

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        n_critics: int = 2,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        use_pnorm: bool = False,
        critic_impl: CriticImpl = "vmap",
    ) -> None:
        super().__init__()
        if n_critics < 1:
            raise ValueError(f"n_critics must be >= 1, got {n_critics}")
        if critic_impl not in ("vmap", "legacy"):
            raise ValueError(
                f"critic_impl must be 'vmap' or 'legacy', got {critic_impl!r}"
            )

        self.n_critics = n_critics
        self.critic_impl = critic_impl
        act_dim = int(np.prod(action_space.shape))
        self._head_kwargs = dict(
            features_dim=features_dim,
            act_dim=act_dim,
            hidden_dims=tuple(hidden_dims),
            backbone_type=backbone_type,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            use_pnorm=use_pnorm,
        )

        if critic_impl == "legacy":
            self.q_nets = nn.ModuleList(
                [_QHead(**self._head_kwargs) for _ in range(n_critics)]
            )
            self._dotted_param_names = []
            self._dotted_buffer_names = []
            return

        # 1) Initialize N independent critics so each has diverse random init.
        critics = [_QHead(**self._head_kwargs) for _ in range(n_critics)]
        # 2) Stack their state_dicts along axis 0.
        stacked_params, stacked_buffers = torch.func.stack_module_state(critics)

        # 3) Register stacked params as actual nn.Parameters (so optimizers,
        #    ``parameters()`` and ``state_dict`` see them). Remember the
        #    dotted-name mapping for ``functional_call`` reconstruction.
        self._dotted_param_names: list[str] = list(stacked_params.keys())
        for dotted in self._dotted_param_names:
            self.register_parameter(
                _safe_name(dotted, _PARAM_PREFIX),
                nn.Parameter(stacked_params[dotted].detach().clone()),
            )

        # 4) Same for buffers (MLP/LayerNorm have none, but be defensive).
        self._dotted_buffer_names: list[str] = list(stacked_buffers.keys())
        for dotted in self._dotted_buffer_names:
            self.register_buffer(
                _safe_name(dotted, _BUFFER_PREFIX),
                stacked_buffers[dotted].detach().clone(),
            )

        # 5) Prototype carries the forward() structure for functional_call.
        #    Stored via object.__setattr__ to bypass nn.Module's submodule
        #    tracking — we don't want prototype params in ``self.parameters()``.
        #    Save/restore CPU RNG state so prototype init doesn't shift
        #    downstream random ops such as replay-buffer sampling.
        _cpu_rng = torch.get_rng_state()
        prototype = _QHead(**self._head_kwargs)
        torch.set_rng_state(_cpu_rng)
        # Move prototype to meta device so its parameters take no memory and
        # cannot be accidentally trained. functional_call replaces all params.
        prototype.to("meta")
        object.__setattr__(self, "_prototype", prototype)

    def _gather_params(self) -> dict[str, torch.Tensor]:
        if self.critic_impl != "vmap":
            raise RuntimeError("_gather_params is only valid for critic_impl='vmap'.")
        return {
            dotted: getattr(self, _safe_name(dotted, _PARAM_PREFIX))
            for dotted in self._dotted_param_names
        }

    def _gather_buffers(self) -> dict[str, torch.Tensor]:
        if self.critic_impl != "vmap":
            raise RuntimeError("_gather_buffers is only valid for critic_impl='vmap'.")
        return {
            dotted: getattr(self, _safe_name(dotted, _BUFFER_PREFIX))
            for dotted in self._dotted_buffer_names
        }

    def _vmapped_forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Run all N critics in one fused pass; returns ``(n_critics, batch, 1)``."""
        if self.critic_impl != "vmap":
            raise RuntimeError("_vmapped_forward is only valid for critic_impl='vmap'.")
        prototype = self._prototype

        def single(params, buffers, f, a):
            return torch.func.functional_call(prototype, (params, buffers), (f, a))

        return torch.func.vmap(single, in_dims=(0, 0, None, None))(
            self._gather_params(), self._gather_buffers(), features, actions
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if self.critic_impl == "legacy":
            return tuple(q(features, actions) for q in self.q_nets)
        return tuple(self._vmapped_forward(features, actions).unbind(0))

    def forward_all(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if self.critic_impl == "legacy":
            return torch.stack(self.forward(features, actions), dim=0)
        return self._vmapped_forward(features, actions)

    def trunk_features_first(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return trunk activations for critic 0 for diagnostics only."""
        if self.critic_impl == "legacy":
            return self.q_nets[0].trunk_features(features, actions)

        prototype = self._prototype
        params = {
            dotted[len("trunk.") :]: tensor[0]
            for dotted, tensor in self._gather_params().items()
            if dotted.startswith("trunk.")
        }
        buffers = {
            dotted[len("trunk.") :]: tensor[0]
            for dotted, tensor in self._gather_buffers().items()
            if dotted.startswith("trunk.")
        }
        x = torch.cat([features, actions], dim=-1)
        return torch.func.functional_call(prototype.trunk, (params, buffers), (x,))

    # ------------------------------------------------------------------
    # Legacy state_dict migration (q_nets.{i}.* → ens_p_*).
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if getattr(self, "critic_impl", "vmap") == "legacy":
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        # Detect legacy ``q_nets.<i>.<...>`` layout in the keys that target
        # this module and convert them in-place to the new ``ens_p_<...>``
        # stacked layout before delegating to the base implementation.
        legacy_prefix = prefix + "q_nets."
        legacy_keys = [k for k in state_dict if k.startswith(legacy_prefix)]
        if legacy_keys:
            grouped: dict[str, dict[int, torch.Tensor]] = {}
            for key in legacy_keys:
                # key looks like "<prefix>q_nets.3.trunk.0.weight"
                tail = key[len(legacy_prefix) :]
                idx_str, _, dotted = tail.partition(".")
                idx = int(idx_str)
                grouped.setdefault(dotted, {})[idx] = state_dict.pop(key)
            for dotted, by_idx in grouped.items():
                stacked = torch.stack([by_idx[i] for i in range(self.n_critics)], dim=0)
                state_dict[prefix + _safe_name(dotted, _PARAM_PREFIX)] = stacked

        # Migrate intermediate flat-sequential format: ``ens_p_{N}__weight``
        # (no trunk/head submodule) → current ``ens_p_trunk__{N}__weight`` /
        # ``ens_p_head__weight`` layout.  The last (highest-indexed) linear
        # layer becomes ``head``; all preceding ones stay in ``trunk``.
        flat_pat = re.compile(
            r"^" + re.escape(prefix + _PARAM_PREFIX) + r"(\d+)__(.+)$"
        )
        flat_keys = [k for k in list(state_dict) if flat_pat.match(k)]
        if flat_keys:
            max_idx = max(int(flat_pat.match(k).group(1)) for k in flat_keys)
            for key in flat_keys:
                m = flat_pat.match(key)
                flat_idx, param_suffix = int(m.group(1)), m.group(2)
                if flat_idx == max_idx:
                    new_dotted = "head." + param_suffix.replace(_PARAM_SEP, ".")
                else:
                    new_dotted = f"trunk.{flat_idx}." + param_suffix.replace(
                        _PARAM_SEP, "."
                    )
                state_dict[prefix + _safe_name(new_dotted, _PARAM_PREFIX)] = (
                    state_dict.pop(key)
                )

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
