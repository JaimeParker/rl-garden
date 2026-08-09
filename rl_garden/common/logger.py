"""Unified logger backend for TensorBoard / W&B / stdout-only modes."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Literal

from torch.utils.tensorboard import SummaryWriter

LogType = Literal["tensorboard", "wandb", "none"]


def _flatten_wandb_config(config: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Flatten a resolved-config dict for wandb's config table.

    ``resolved_run_config()`` nests every training arg under one ``"args"``
    key (so ``config.json`` reads as a structured record), but wandb's table
    view doesn't expand nested dicts into columns. Promote ``args``' own keys
    to the top level unprefixed (matching the pre-refactor flat schema every
    existing wandb report/table filter was built against), and dot-flatten
    any further nesting (e.g. per-backend ``EnvBackendArgs`` groups).
    """
    flat: dict[str, Any] = {}

    def _add(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, sub_value in value.items():
                _add(f"{prefix}{sep}{key}" if prefix else str(key), sub_value)
        else:
            flat[prefix] = value

    is_effective_config = config.get("schema_version") == 3 and isinstance(
        config.get("inputs"), dict
    )
    for key, value in config.items():
        if key in {"args", "inputs"} and isinstance(value, dict):
            _add("", value)
        elif is_effective_config and key == "selection":
            _add("selection", value)
        elif is_effective_config and config.get("status") == "preflight":
            continue
        elif is_effective_config:
            _add(f"effective{sep}{key}", value)
        else:
            _add(str(key), value)
    return flat


class Logger:
    def __init__(
        self,
        tensorboard: SummaryWriter | None = None,
        wandb_run: Any | None = None,
        log_type: LogType = "none",
    ) -> None:
        self.writer = tensorboard
        self.wandb_run = wandb_run
        self.log_type = log_type

    @staticmethod
    def _parse_keywords(log_keywords: str | None) -> list[str]:
        if log_keywords is None:
            return []
        tags: list[str] = []
        for raw in log_keywords.split(","):
            tag = raw.strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    @classmethod
    def create(
        cls,
        *,
        log_type: str,
        log_dir: str,
        run_name: str,
        config: dict[str, Any] | None = None,
        start_time: str | None = None,
        log_keywords: str | None = None,
        wandb_project: str = "rl-garden",
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
    ) -> Logger:
        normalized = log_type.lower()
        if normalized == "tensorboard":
            writer = SummaryWriter(os.path.join(log_dir, run_name))
            return cls(tensorboard=writer, log_type="tensorboard")
        if normalized == "none":
            return cls(log_type="none")
        if normalized != "wandb":
            raise ValueError(
                f"Unsupported log_type={log_type!r}. Choose from: tensorboard, wandb, none."
            )

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "log_type=wandb requires `wandb` to be installed. "
                "Install with: pip install -e '.[wandb]'"
            ) from exc

        api_key = os.getenv("WANDB_API_KEY") or getattr(
            getattr(wandb, "api", None), "api_key", None
        )
        if not api_key:
            raise RuntimeError(
                "log_type=wandb requires wandb authentication. "
                "Run `wandb login` or set WANDB_API_KEY."
            )

        tags = cls._parse_keywords(log_keywords)
        wandb_name = run_name
        if start_time:
            wandb_name = f"{wandb_name}__{start_time}"
        if tags:
            wandb_name = f"{wandb_name}__{'-'.join(tags)}"

        init_kwargs: dict[str, Any] = {
            "project": wandb_project,
            "name": wandb_name,
            "config": _flatten_wandb_config(config or {}),
        }
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity
        if wandb_group:
            init_kwargs["group"] = wandb_group
        if tags:
            init_kwargs["tags"] = tags

        run = wandb.init(**init_kwargs)
        if run is None:
            raise RuntimeError(
                "wandb.init() returned None; failed to initialize wandb run."
            )
        return cls(wandb_run=run, log_type="wandb")

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log({tag: value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str) -> None:
        if self.wandb_run is not None:
            self.wandb_run.summary[tag] = text
        if self.writer is not None:
            self.writer.add_text(tag, text)

    def add_summary(self, tag: str, value: Any) -> None:
        """Record run-level metadata without creating sparse scalar curves."""
        if self.wandb_run is not None:
            self.wandb_run.summary[tag] = value
        elif self.writer is not None:
            self.writer.add_text(tag, str(value))

    def update_config(self, config: dict[str, Any]) -> None:
        """Replace run metadata after runtime configuration is materialized."""
        flat = _flatten_wandb_config(config)
        if self.wandb_run is not None:
            self.wandb_run.config.update(flat, allow_val_change=True)
        if self.writer is not None:
            import json

            self.writer.add_text("effective_config", json.dumps(config, sort_keys=True))

    # --- RL metric logging utilities ---

    # Explicit namespace mapping for RL training metrics (shared by online and offline)
    METRIC_NAMESPACES: ClassVar[dict[str, str]] = {
        # Q-value metrics
        "predicted_q": "q/predicted",
        "target_q": "q/target",
        "cql_ood_values": "q/cql_ood",
        "cql_q_diff": "q/cql_diff",
        # Q landscape diagnostics
        "q_uniform_var": "qdiag/q_uniform_var",
        "q_action_grad_norm": "qdiag/q_action_grad_norm",
        "feature_norm": "qdiag/feature_norm",
        "feature_dormant_ratio": "qdiag/feature_dormant_ratio",
        "critic_hidden_dormant_ratio": "qdiag/critic_hidden_dormant_ratio",
        # Loss metrics
        "actor_loss": "losses/actor_loss",
        "critic_loss": "losses/critic_loss",
        "td_loss": "losses/td_loss",
        "cql_loss": "losses/cql_loss",
        "alpha_loss": "losses/alpha_loss",
        "cql_alpha_loss": "losses/cql_alpha_loss",
        # CQL-specific metrics
        "cql_alpha": "cql/alpha",
        "calql_bound_rate": "cql/bound_rate",
        # Entropy metrics
        "alpha": "entropy/alpha",
        "entropy": "entropy/entropy",
        "target_entropy": "entropy/target_entropy",
        "policy_std": "entropy/policy_std",
        "action_saturation": "entropy/action_saturation",
        "entropy_gaussian": "entropy/entropy_gaussian",
        "tanh_correction": "entropy/tanh_correction",
        # Training metrics
        "utd_ratio": "train/utd_ratio",
    }

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        *,
        metric_namespaces: dict[str, str] | None = None,
        default_namespace: str = "losses",
    ) -> None:
        """Log RL training metrics with explicit namespace mapping.

        Each metric is logged to its explicitly defined namespace path.
        Unknown metrics are logged to the default namespace.

        Args:
            metrics: Raw metric dict from training.
            step: Global step for logging.
            metric_namespaces: Mapping from metric keys to full namespace paths.
                Example: {"predicted_q": "q/predicted", "actor_loss": "losses/actor_loss"}
                If None, uses Logger.METRIC_NAMESPACES.
            default_namespace: Namespace prefix for unmapped metrics (default: "losses").
        """
        if metric_namespaces is None:
            metric_namespaces = self.METRIC_NAMESPACES

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            # Use explicit mapping or default namespace
            full_path = metric_namespaces.get(key, f"{default_namespace}/{key}")
            self.add_scalar(full_path, value, step)

    @staticmethod
    def format_metrics(
        metrics: dict[str, float],
        *,
        metric_namespaces: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Format training metrics for console output.

        Separates metrics into loss and Q-value groups based on their namespace.

        Args:
            metrics: Raw metric dict from training.
            metric_namespaces: Mapping from metric keys to full namespace paths.
                If None, uses Logger.METRIC_NAMESPACES.

        Returns:
            (loss_summary, q_summary) as formatted strings.
        """
        if metric_namespaces is None:
            metric_namespaces = Logger.METRIC_NAMESPACES

        loss_parts: list[str] = []
        q_parts: list[str] = []

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            # Get the full namespace path
            full_path = metric_namespaces.get(key, f"losses/{key}")
            namespace, _, display_key = full_path.rpartition("/")

            # Group by namespace for console output
            if namespace == "q":
                q_parts.append(f"{display_key}={value:.4f}")
            elif namespace == "losses":
                # For losses and other namespaces, show the key as-is
                loss_parts.append(f"{key}={value:.4f}")

        return " ".join(loss_parts), " ".join(q_parts)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
