"""Shared ``policy_kwargs`` validation/normalization for algorithms that
accept a ``policy_kwargs`` constructor dict (features-extractor overrides).

Every algorithm family below independently duplicated the same "unknown key"
rejection and the same per-key ``*_kwargs`` dict-type check; only ``SAC``
additionally guarded against a ``*_kwargs`` dict being given without its
matching ``*_class`` (which the default extractor would otherwise silently
ignore). ``normalize_policy_kwargs`` is the one implementation, used by every
family so that guard applies uniformly rather than only for SAC.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def normalize_policy_kwargs(
    policy_kwargs: Optional[dict[str, Any]],
    *,
    supported_keys: frozenset,
    pairs: Sequence[tuple[str, str]] = (
        ("features_extractor_kwargs", "features_extractor_class"),
    ),
) -> dict[str, Any]:
    """Validate ``policy_kwargs`` against ``supported_keys`` and normalize
    each ``(kwargs_key, class_key)`` pair in ``pairs``.

    For every pair: ``kwargs_key``'s value must be a ``dict`` (or ``None``,
    left absent); a non-empty one given without ``class_key`` set raises
    (it would be silently ignored -- the default extractor takes no such
    kwargs). ``"features_extractor_kwargs"`` is always guaranteed present in
    the returned dict (defaulting to ``{}``), matching every existing
    algorithm's convention of reading it unconditionally.
    """
    normalized = dict(policy_kwargs or {})
    unknown_keys = sorted(set(normalized) - supported_keys)
    if unknown_keys:
        raise ValueError(
            "Unsupported policy_kwargs keys: "
            + ", ".join(unknown_keys)
            + ". Supported keys are: "
            + ", ".join(sorted(supported_keys))
            + "."
        )

    for kwargs_key, class_key in pairs:
        kwargs = normalized.get(kwargs_key)
        if kwargs is None:
            continue
        if not isinstance(kwargs, dict):
            raise TypeError(f"policy_kwargs[{kwargs_key!r}] must be a dict.")
        if kwargs and normalized.get(class_key) is None:
            raise ValueError(
                f"policy_kwargs[{kwargs_key!r}] was given without "
                f"policy_kwargs[{class_key!r}]; it would be silently "
                "ignored (the default extractor takes no such kwargs)."
            )
        normalized[kwargs_key] = dict(kwargs)

    normalized.setdefault("features_extractor_kwargs", {})
    return normalized
