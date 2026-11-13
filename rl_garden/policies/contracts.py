"""The actor contracts the PPO and SAC families train, and their validators.

An actor head reads the features the policy's extractor produced, a
``(B, features_dim)`` tensor, and acts in a ``Box`` action space of
``A = prod(action_space.shape)`` dimensions; actions are ``(B, A)``.

- ``PPOActor``: sample an action with its log-probability, score a STORED
  action differentiably, optionally report an entropy. With ``sum_dims=True``
  log-probabilities and entropies are ``(B, 1)`` (summed over the action
  dimensions with ``keepdim=True``, as ``DiagGaussianActorBase`` returns
  them); with ``sum_dims=False`` they are ``(B, A)``, one entry per action
  dimension. ``entropy`` is ``None`` for an actor without a tractable
  entropy (a flow head); ``sum_dims=False`` may raise
  ``NotImplementedError`` when the log-probability does not factor per
  dimension.
- ``GaussianParamsActor``: an opt-in for a diagonal Gaussian PPO actor whose
  ``forward`` returns the ``Normal`` and whose ``log_std`` parameter
  broadcasts to its mean (``adaptive_kl``, decision-level ``log_std``
  logging).
- ``SACActor``: ``action_log_prob`` draws a REPARAMETERIZED sample, so the
  actor loss backpropagates through the action into the actor, and returns
  its log-probability ``(B, 1)``. ``sample_actions_with_log_prob(f, n)``
  returns ``(B, n, A)`` actions and ``(B, n)`` log-probabilities. Three
  class attributes declare what the actor supports: ``stochastic`` (False for
  a deterministic, TD3-shaped actor whose log-probability is a zero
  placeholder), ``log_prob_is_density`` (False when the log-probability is a
  path likelihood, as in a flow actor, not the density of the action) and
  ``supports_evaluate_log_prob`` (True when ``evaluate_action_log_prob``
  scores external actions, ``(B, 1)``; otherwise it raises
  ``NotImplementedError``).

``validate_ppo_actor`` and ``validate_sac_actor`` check an actor once, at
construction: a presence check (``TypeError`` listing every missing
attribute or method), then one dry pass on random features under
``torch.no_grad()`` and ``torch.random.fork_rng`` (``ValueError`` naming the
method and the shapes on a mismatch, ``TypeError`` on a wrong return type).
Both check that ``evaluate_action_log_prob`` reproduces, for the actor's
own sample, the log-probability ``action_log_prob`` returned with it
(``ValueError`` otherwise; for a SAC actor only when it declares
``supports_evaluate_log_prob``): PPO's importance ratio and the SAC
family's scoring of stored actions rely on it. The comparison (``atol`` and
``rtol`` 1e-3) skips rows whose sampled action lies within 1e-3 of the
action range from a bound: scoring a saturated action goes back through
the squash's inverse (``atanh``), which float32 cannot resolve there. When
no row of the batch is left, one more batch of random features is drawn;
when none is left again, the check is skipped (logged at DEBUG).
The dry pass leaves the global torch RNG state (CPU and the given CUDA
device) byte-identical and runs with every submodule in eval mode, restoring
each module's mode afterwards, so it neither consumes random numbers nor
updates running statistics. It does not check gradients (it runs without
autograd) and does not isolate NumPy or Python ``random`` state.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable

import torch
from gymnasium import spaces
from torch import Tensor, nn

from rl_garden.common.effective_config import import_path_of, unrecorded_import_path

#: Rows of the random features the validators' dry pass feeds the actor.
_DRY_PASS_BATCH = 2
#: ``n`` of the dry pass's ``sample_actions_with_log_prob`` call.
_DRY_PASS_SAMPLES = 2
#: ``atol`` and ``rtol`` of the dry pass's score comparisons: the check that
#: ``evaluate_action_log_prob`` scores the actor's own sample with the
#: log-prob ``action_log_prob`` returned, and the ``is_gaussian`` probe
#: (``evaluate_action_log_prob`` of a sample of ``forward``'s ``Normal``
#: against that ``Normal``'s log-density).
_SCORE_TOLERANCE = 1e-3
#: A sampled action closer than this fraction of a dimension's range to
#: one of its (finite) bounds is saturated; its row is left out of the
#: score comparisons.
_SATURATION_MARGIN = 1e-3

_logger = logging.getLogger(__name__)

#: ``draw(features) -> (actions, evaluated log-prob, reference log-prob)``,
#: one row per feature row.
_ScoreDraw = Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]


@runtime_checkable
class PPOActor(Protocol):
    """The actor PPO trains: see the module docstring for the shapes."""

    def action_log_prob(
        self, features: Tensor, deterministic: bool = False, *, sum_dims: bool = True
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """``(action, log_prob, entropy or None)`` for a sampled action (the
        mean action when ``deterministic``)."""
        ...

    def evaluate_action_log_prob(
        self, features: Tensor, actions: Tensor, *, sum_dims: bool = True
    ) -> tuple[Tensor, Optional[Tensor]]:
        """``(log_prob, entropy or None)`` of stored ``actions``,
        differentiable in the actor's parameters."""
        ...

    def deterministic_action(self, features: Tensor) -> Tensor: ...

    def clamp_action(self, actions: Tensor) -> Tensor:
        """``actions`` clipped to the action bounds before an env step; the
        log-probability is always evaluated on the unclamped action."""
        ...


@runtime_checkable
class GaussianParamsActor(Protocol):
    """Opt-in: a diagonal Gaussian PPO actor (``adaptive_kl``,
    decision-level ``log_std`` logging).

    Read ``PPOActorCapabilities.is_gaussian`` rather than
    ``isinstance(actor, GaussianParamsActor)``: from Python 3.12 a
    runtime-checkable protocol looks its members up with
    ``inspect.getattr_static``, which does not see a ``log_std`` registered
    as an ``nn.Parameter`` or buffer, so ``isinstance`` is ``False`` for
    ``DiagGaussianActor`` there."""

    log_std: Tensor

    def forward(self, features: Tensor) -> torch.distributions.Normal: ...


@runtime_checkable
class SACActor(Protocol):
    """The actor the SAC family trains: see the module docstring."""

    stochastic: bool
    log_prob_is_density: bool
    supports_evaluate_log_prob: bool

    def action_log_prob(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """A reparameterized ``(action (B, A), log_prob (B, 1))``."""
        ...

    def deterministic_action(self, features: Tensor) -> Tensor: ...

    def sample_actions_with_log_prob(self, features: Tensor, n: int) -> tuple[Tensor, Tensor]:
        """``n`` reparameterized samples per row: ``(B, n, A)``, ``(B, n)``."""
        ...

    def evaluate_action_log_prob(self, features: Tensor, actions: Tensor) -> Tensor:
        """``(B, 1)`` log-probability of external ``actions``; raises
        ``NotImplementedError`` unless ``supports_evaluate_log_prob``."""
        ...

    def diagnostics(self, features: Tensor) -> dict[str, Tensor]:
        """Detached scalar diagnostics, logged by name."""
        ...


@dataclass(frozen=True)
class PPOActorCapabilities:
    #: Both log-prob methods return an entropy (not ``None``).
    has_entropy: bool
    #: Both log-prob methods accept ``sum_dims=False`` (per-dimension terms).
    supports_per_dim_log_prob: bool
    #: A ``log_std`` tensor and a ``forward`` returning a ``Normal`` over
    #: ``(B, A)`` that ``log_std`` broadcasts to (``GaussianParamsActor``),
    #: whose log-density ``evaluate_action_log_prob`` reproduces for a
    #: sample of it (so the ``Normal`` is the action's distribution, not,
    #: say, the pre-squash one of a tanh-squashed actor).
    is_gaussian: bool


@dataclass(frozen=True)
class SACActorCapabilities:
    stochastic: bool
    log_prob_is_density: bool
    supports_evaluate_log_prob: bool


_PPO_METHODS = (
    "action_log_prob",
    "evaluate_action_log_prob",
    "deterministic_action",
    "clamp_action",
)
_SAC_METHODS = (
    "action_log_prob",
    "deterministic_action",
    "sample_actions_with_log_prob",
    "evaluate_action_log_prob",
    "diagnostics",
)
_SAC_FLAGS = ("stochastic", "log_prob_is_density", "supports_evaluate_log_prob")


def validate_ppo_actor(
    actor: nn.Module,
    features_dim: int,
    action_space: spaces.Box,
    device: Union[torch.device, str, None] = None,
) -> PPOActorCapabilities:
    """Check ``actor`` against ``PPOActor`` and return what it supports.

    ``device`` is where the dry pass runs and must hold the actor; ``None``
    takes the device of the actor's first parameter or buffer (CPU when it
    has none)."""
    _check_presence(actor, "PPO", _PPO_METHODS, ())
    act_dim = _action_dim(action_space)
    device = _resolve_device(actor, device)
    name = type(actor).__name__
    with _dry_pass(actor, device):
        features = torch.randn(_DRY_PASS_BATCH, features_dim, device=device)
        summed = (_DRY_PASS_BATCH, 1)
        per_dim = (_DRY_PASS_BATCH, act_dim)

        action, log_prob, entropy = _unpack(
            actor.action_log_prob(features, sum_dims=True), 3, f"{name}.action_log_prob"
        )
        _expect_shape(action, per_dim, f"{name}.action_log_prob", "action")
        _expect_shape(log_prob, summed, f"{name}.action_log_prob(sum_dims=True)", "log_prob")
        _expect_optional_shape(
            entropy, summed, f"{name}.action_log_prob(sum_dims=True)", "entropy"
        )
        mean_action, _, _ = _unpack(
            actor.action_log_prob(features, deterministic=True),
            3,
            f"{name}.action_log_prob(deterministic=True)",
        )
        _expect_shape(mean_action, per_dim, f"{name}.action_log_prob(deterministic=True)", "action")

        eval_log_prob, eval_entropy = _unpack(
            actor.evaluate_action_log_prob(features, action, sum_dims=True),
            2,
            f"{name}.evaluate_action_log_prob",
        )
        _expect_shape(
            eval_log_prob, summed, f"{name}.evaluate_action_log_prob(sum_dims=True)", "log_prob"
        )
        _expect_optional_shape(
            eval_entropy, summed, f"{name}.evaluate_action_log_prob(sum_dims=True)", "entropy"
        )
        bounds = _action_bounds(action_space, device)

        def draw_ppo_self_score(rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            sample, sample_log_prob, _ = actor.action_log_prob(rows, sum_dims=True)
            evaluated, _ = actor.evaluate_action_log_prob(rows, sample, sum_dims=True)
            return sample, evaluated, sample_log_prob

        _expect_self_score(
            _unsaturated_scores(
                (action, eval_log_prob, log_prob), draw_ppo_self_score, features, bounds
            ),
            name,
        )
        if (entropy is None) != (eval_entropy is None):
            raise ValueError(
                f"{name}: action_log_prob returned entropy "
                f"{_describe(entropy)} but evaluate_action_log_prob returned "
                f"{_describe(eval_entropy)}; both must return an entropy or both None."
            )

        _expect_shape(
            actor.deterministic_action(features), per_dim, f"{name}.deterministic_action", "action"
        )
        _expect_shape(actor.clamp_action(action), per_dim, f"{name}.clamp_action", "action")

        supports_per_dim = _probe_per_dim_log_prob(actor, features, action, per_dim, name)
        is_gaussian = _probe_gaussian(actor, features, per_dim, bounds, name)
    return PPOActorCapabilities(
        has_entropy=entropy is not None,
        supports_per_dim_log_prob=supports_per_dim,
        is_gaussian=is_gaussian,
    )


def validate_sac_actor(
    actor: nn.Module,
    features_dim: int,
    action_space: spaces.Box,
    device: Union[torch.device, str, None] = None,
) -> SACActorCapabilities:
    """Check ``actor`` against ``SACActor`` and return its declared flags.

    ``device`` as in ``validate_ppo_actor``."""
    _check_presence(actor, "SAC", _SAC_METHODS, _SAC_FLAGS)
    capabilities = SACActorCapabilities(**{flag: getattr(actor, flag) for flag in _SAC_FLAGS})
    act_dim = _action_dim(action_space)
    device = _resolve_device(actor, device)
    name = type(actor).__name__
    batch, n = _DRY_PASS_BATCH, _DRY_PASS_SAMPLES
    with _dry_pass(actor, device):
        features = torch.randn(batch, features_dim, device=device)

        action, log_prob = _unpack(actor.action_log_prob(features), 2, f"{name}.action_log_prob")
        _expect_shape(action, (batch, act_dim), f"{name}.action_log_prob", "action")
        _expect_shape(log_prob, (batch, 1), f"{name}.action_log_prob", "log_prob")

        _expect_shape(
            actor.deterministic_action(features),
            (batch, act_dim),
            f"{name}.deterministic_action",
            "action",
        )

        method = f"{name}.sample_actions_with_log_prob(n={n})"
        actions, log_probs = _unpack(actor.sample_actions_with_log_prob(features, n), 2, method)
        _expect_shape(actions, (batch, n, act_dim), method, "actions")
        _expect_shape(log_probs, (batch, n), method, "log_prob")

        if capabilities.supports_evaluate_log_prob:
            eval_log_prob = actor.evaluate_action_log_prob(features, action)
            _expect_shape(
                eval_log_prob, (batch, 1), f"{name}.evaluate_action_log_prob", "log_prob"
            )

            def draw_sac_self_score(rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                sample, sample_log_prob = actor.action_log_prob(rows)
                return sample, actor.evaluate_action_log_prob(rows, sample), sample_log_prob

            _expect_self_score(
                _unsaturated_scores(
                    (action, eval_log_prob, log_prob),
                    draw_sac_self_score,
                    features,
                    _action_bounds(action_space, device),
                ),
                name,
            )

        diagnostics = actor.diagnostics(features)
        if not isinstance(diagnostics, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in diagnostics.items()
        ):
            raise TypeError(
                f"{name}.diagnostics must return a dict[str, Tensor], got {_describe(diagnostics)}."
            )
    return capabilities


# --- helpers -------------------------------------------------------------------------------


def _check_presence(
    actor: Any, family: str, methods: tuple[str, ...], flags: tuple[str, ...]
) -> None:
    name = type(actor).__name__
    if not isinstance(actor, nn.Module):
        raise TypeError(f"A {family} actor must be a torch.nn.Module, got {name}.")
    missing = [method for method in methods if not callable(getattr(actor, method, None))]
    missing += [flag for flag in flags if not hasattr(actor, flag)]
    if missing:
        raise TypeError(
            f"{name} does not implement the {family} actor contract "
            f"(rl_garden.policies.contracts.{family}Actor); missing: {', '.join(missing)}."
        )
    not_bool = [
        f"{flag}={getattr(actor, flag)!r}"
        for flag in flags
        if not isinstance(getattr(actor, flag), bool)
    ]
    if not_bool:
        raise TypeError(f"{name}: these {family} actor flags must be bool: {', '.join(not_bool)}.")


def _action_dim(action_space: spaces.Space) -> int:
    if not isinstance(action_space, spaces.Box):
        raise TypeError(
            f"The actor contracts act in a Box action space, got {type(action_space).__name__}."
        )
    return int(math.prod(action_space.shape))


def _resolve_device(actor: nn.Module, device: Union[torch.device, str, None]) -> torch.device:
    if device is not None:
        return torch.device(device)
    for tensor in _module_tensors(actor):
        return tensor.device
    return torch.device("cpu")


def _module_tensors(module: nn.Module) -> Iterator[Tensor]:
    yield from module.parameters()
    yield from module.buffers()


@contextmanager
def _dry_pass(actor: nn.Module, device: torch.device) -> Iterator[None]:
    """No autograd, a forked global RNG (CPU and ``device``) and eval mode,
    each module's training flag restored afterwards."""
    fork_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        fork_devices = [torch.cuda.current_device() if device.index is None else device.index]
    modes = [(module, module.training) for module in actor.modules()]
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
            actor.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def _probe_per_dim_log_prob(
    actor: Any, features: Tensor, action: Tensor, per_dim: tuple[int, ...], name: str
) -> bool:
    """Whether both log-prob methods accept ``sum_dims=False``; one accepting
    it while the other raises ``NotImplementedError`` is a contract error."""
    probes: dict[str, Callable[[], tuple[Any, ...]]] = {
        "action_log_prob": lambda: _unpack(
            actor.action_log_prob(features, sum_dims=False),
            3,
            f"{name}.action_log_prob(sum_dims=False)",
        )[1:],
        "evaluate_action_log_prob": lambda: _unpack(
            actor.evaluate_action_log_prob(features, action, sum_dims=False),
            2,
            f"{name}.evaluate_action_log_prob(sum_dims=False)",
        ),
    }
    supported: dict[str, bool] = {}
    for method, probe in probes.items():
        try:
            log_prob, entropy = probe()
        except NotImplementedError:
            supported[method] = False
            continue
        supported[method] = True
        _expect_shape(log_prob, per_dim, f"{name}.{method}(sum_dims=False)", "log_prob")
        _expect_optional_shape(entropy, per_dim, f"{name}.{method}(sum_dims=False)", "entropy")
    if len(set(supported.values())) > 1:
        raise ValueError(
            f"{name}: sum_dims=False is supported by "
            f"{[m for m, ok in supported.items() if ok]} but raises NotImplementedError in "
            f"{[m for m, ok in supported.items() if not ok]}; support it in both or neither."
        )
    return all(supported.values())


def _probe_gaussian(
    actor: Any,
    features: Tensor,
    per_dim: tuple[int, ...],
    bounds: tuple[Tensor, Tensor],
    name: str,
) -> bool:
    # Not ``isinstance(actor, GaussianParamsActor)``: see that class.
    log_std = getattr(actor, "log_std", None)
    if not isinstance(log_std, Tensor):
        return False
    dist = actor(features)
    if not isinstance(dist, torch.distributions.Normal):
        return False
    _expect_shape(dist.mean, per_dim, f"{name}.forward", "Normal mean")
    try:
        log_std.expand_as(dist.mean)
    except RuntimeError as error:
        raise ValueError(
            f"{name}.log_std of shape {tuple(log_std.shape)} does not broadcast to "
            f"its Normal's mean {tuple(dist.mean.shape)}."
        ) from error
    # A tanh-squashed actor may hold a ``log_std`` and return its pre-squash
    # Normal from ``forward``: only a Normal whose density scores the actions
    # is the action distribution the Gaussian-only features read.
    method = f"{name}.evaluate_action_log_prob(sum_dims=True)"

    def draw_gaussian_score(rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normal = actor(rows)
        sample = normal.sample()
        evaluated, _ = _unpack(
            actor.evaluate_action_log_prob(rows, sample, sum_dims=True), 2, method
        )
        _expect_shape(evaluated, (rows.shape[0], 1), method, "log_prob")
        return sample, evaluated, normal.log_prob(sample).sum(-1, keepdim=True)

    scores = _unsaturated_scores(
        draw_gaussian_score(features), draw_gaussian_score, features, bounds
    )
    if scores is None:
        _logger.debug(
            "%s: every sample of forward's Normal is saturated; is_gaussian rests on "
            "log_std and forward alone.",
            name,
        )
        return True
    return _scores_match(*scores)


def _action_bounds(action_space: spaces.Box, device: torch.device) -> tuple[Tensor, Tensor]:
    """``(low, high)`` of ``action_space`` flattened to ``(A,)`` on ``device``."""
    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device).reshape(-1)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device).reshape(-1)
    return low, high


def _unsaturated_rows(actions: Tensor, bounds: tuple[Tensor, Tensor]) -> Tensor:
    """``(B,)``: the rows of ``actions`` with no dimension within
    ``_SATURATION_MARGIN`` of its range from a finite bound."""
    low, high = bounds
    finite = torch.isfinite(low) & torch.isfinite(high)
    margin = torch.where(finite, _SATURATION_MARGIN * (high - low), torch.zeros_like(low))
    flat = actions.reshape(actions.shape[0], -1).to(low.dtype)
    near_bound = finite & (((flat - low).abs() < margin) | ((high - flat).abs() < margin))
    return ~near_bound.any(dim=-1)


def _unsaturated_scores(
    first: tuple[Tensor, Tensor, Tensor],
    draw: _ScoreDraw,
    features: Tensor,
    bounds: tuple[Tensor, Tensor],
) -> Optional[tuple[Tensor, Tensor]]:
    """``(evaluated, reference)`` on the rows of ``first`` (one ``draw`` on
    ``features``) whose action is not saturated; when there are none, the
    same for one ``draw`` on a fresh batch of random features; ``None``
    when that batch has none either."""
    actions, evaluated, reference = first
    keep = _unsaturated_rows(actions, bounds)
    if not keep.any():
        actions, evaluated, reference = draw(torch.randn_like(features))
        keep = _unsaturated_rows(actions, bounds)
        if not keep.any():
            return None
    return evaluated[keep], reference[keep]


def _scores_match(evaluated: Tensor, reference: Tensor) -> bool:
    return torch.allclose(
        evaluated.to(reference.dtype), reference, atol=_SCORE_TOLERANCE, rtol=_SCORE_TOLERANCE
    )


def _expect_self_score(scores: Optional[tuple[Tensor, Tensor]], name: str) -> None:
    """``evaluate_action_log_prob`` of the samples ``action_log_prob`` drew
    must reproduce the log-probs ``action_log_prob`` returned with them, on
    the unsaturated rows (``_unsaturated_scores``)."""
    if scores is None:
        _logger.debug(
            "%s: every sampled action is saturated; the evaluate_action_log_prob "
            "self-score check is skipped.",
            name,
        )
        return
    eval_log_prob, log_prob = scores
    if _scores_match(eval_log_prob, log_prob):
        return
    gap = (eval_log_prob.to(log_prob.dtype) - log_prob).abs().max().item()
    raise ValueError(
        f"{name}: evaluate_action_log_prob does not reproduce action_log_prob's log-prob "
        f"for its own sample (largest gap {gap:.3g}, atol/rtol {_SCORE_TOLERANCE:g}); the "
        "training loop scores stored actions with evaluate_action_log_prob, so both "
        "methods must compute the same log-probability."
    )


def _unpack(result: Any, length: int, method: str) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or len(result) != length:
        raise TypeError(f"{method} must return a tuple of {length}, got {_describe(result)}.")
    return result


def _expect_shape(value: Any, expected: tuple[int, ...], method: str, what: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{method}: {what} must be a Tensor, got {_describe(value)}.")
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{method}: {what} has shape {tuple(value.shape)}, expected {expected}."
        )


def _expect_optional_shape(
    value: Any, expected: tuple[int, ...], method: str, what: str
) -> None:
    if value is not None:
        _expect_shape(value, expected, method, what)


def _describe(value: Any) -> str:
    if isinstance(value, Tensor):
        return f"a Tensor of shape {tuple(value.shape)}"
    if isinstance(value, tuple):
        return f"a tuple of {len(value)}"
    return "None" if value is None else type(value).__name__


#: ``policy_kwargs`` key -> checkpoint-metadata key (= the training CLI
#: field) of the user-supplied extractor classes.
_EXTRACTOR_METADATA_KEYS = {
    "actor_extractor_class": "actor_extractor",
    "critic_extractor_class": "critic_extractor",
}


def policy_module_checkpoint_metadata(
    actor_factory: Optional[Callable[..., Any]],
    policy_kwargs: Optional[Mapping[str, Any]],
) -> dict[str, str]:
    """The import paths (``"pkg.mod:attr"``) of the user-supplied policy
    modules an algorithm was built with, keyed like the training CLI fields
    ``actor_factory``, ``actor_extractor`` and ``critic_extractor``, so that
    evaluation restores them onto the rebuilt run's Args. Only the modules
    that were given appear (a default build adds no key). One that has no
    import path (a lambda, a local or ``__main__`` definition) is recorded as
    ``unrecorded_import_path(module)`` (``"<unrecorded: {module!r}>"``), with
    a warning: rebuilding from the checkpoint then raises ``ConfigError``
    (``reject_unrecorded_import_path``), and the checkpoint loads only into
    an agent built with the same module by hand. ``*_extractor_kwargs`` are
    not recorded."""
    modules = {"actor_factory": actor_factory}
    for kwargs_key, metadata_key in _EXTRACTOR_METADATA_KEYS.items():
        modules[metadata_key] = (policy_kwargs or {}).get(kwargs_key)
    metadata: dict[str, str] = {}
    for key, module in modules.items():
        if module is None:
            continue
        try:
            metadata[key] = import_path_of(module)
        except ValueError as error:
            metadata[key] = unrecorded_import_path(module)
            warnings.warn(
                f"{key} has no import path ({error}); the checkpoint metadata records "
                f"{metadata[key]!r} instead, so evaluation cannot rebuild this agent from "
                f"the checkpoint alone. Define the {key} at module level.",
                stacklevel=2,
            )
    return metadata


__all__ = [
    "GaussianParamsActor",
    "PPOActor",
    "PPOActorCapabilities",
    "SACActor",
    "SACActorCapabilities",
    "policy_module_checkpoint_metadata",
    "validate_ppo_actor",
    "validate_sac_actor",
]
