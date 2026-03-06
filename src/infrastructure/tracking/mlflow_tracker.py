"""MLflow-backed tracker adapter implementation."""

# Відповідно до:
# - ARCHITECTURE_RULES.md -> розділ 2-4 (Adapters реалізують порти Application)
# - EVF_MVP1.MD -> розділ 5 (Experiment Tracking / MLflow)
# - AGENT_GUIDE.MD -> розділ 2 (Dependency inversion)

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import mlflow

from src.application.ports.tracker_port import ITracker


class MLflowTracker(ITracker):
    """Tracker adapter that logs params/metrics into MLflow run."""

    def __init__(self, experiment_name: str, run_name: str, tracking_uri: Optional[str] = None) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric value with optional training step."""
        if step is None:
            mlflow.log_metric(key, float(value))
            return
        mlflow.log_metric(key, float(value), step=step)

    def log_param(self, key: str, value: Any) -> None:
        """Log param; nested dictionaries are flattened into dotted keys."""
        for flat_key, flat_value in self._flatten_params(prefix=key, value=value):
            mlflow.log_param(flat_key, flat_value)

    def log_tag(self, key: str, value: Any) -> None:
        """Log run tag into MLflow."""
        mlflow.set_tag(key, str(value))

    def log_artifact(self, path: str) -> None:
        """Log local file artifact into MLflow."""
        mlflow.log_artifact(path)

    def close(self) -> None:
        """Close active MLflow run."""
        mlflow.end_run()

    def _flatten_params(self, prefix: str, value: Any) -> Iterable[Tuple[str, Any]]:
        """Safely flatten nested dictionaries and keep primitives as-is."""
        if value is None:
            yield (prefix, "None")
            return

        if isinstance(value, dict):
            if not value:
                yield (prefix, "{}")
                return
            for nested_key, nested_value in value.items():
                child_prefix = f"{prefix}.{nested_key}" if prefix else str(nested_key)
                yield from self._flatten_params(child_prefix, nested_value)
            return

        if isinstance(value, (list, tuple, set)):
            yield (prefix, ",".join(str(item) for item in value))
            return

        yield (prefix, value)
