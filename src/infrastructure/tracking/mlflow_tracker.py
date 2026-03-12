"""MLflow-backed tracker adapter implementation."""

# Р’С–РґРїРѕРІС–РґРЅРѕ РґРѕ:
# - ARCHITECTURE_RULES.MD -> СЂРѕР·РґС–Р» 2-4 (Adapters СЂРµР°Р»С–Р·СѓСЋС‚СЊ РїРѕСЂС‚Рё Application)
# - EVF_MVP1.MD -> СЂРѕР·РґС–Р» 5 (Experiment Tracking / MLflow)
# - AGENT_GUIDE.MD -> СЂРѕР·РґС–Р» 2 (Dependency inversion)

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import logging
import re

import mlflow
import mlflow.exceptions

from src.application.ports.tracker_port import ITracker


logger = logging.getLogger(__name__)


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
            safe_key = self._sanitize_key(flat_key)
            try:
                mlflow.log_param(safe_key, flat_value)
            except mlflow.exceptions.MlflowException as exc:
                logger.warning("MLflow param collision for %s: %s", safe_key, exc)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log already-flattened params in MLflow-friendly batches."""
        if not params:
            return
        sanitized: Dict[str, Any] = {
            self._sanitize_key(key): value
            for key, value in params.items()
        }
        for safe_key, safe_value in sanitized.items():
            try:
                mlflow.log_param(safe_key, safe_value)
            except mlflow.exceptions.MlflowException as exc:
                logger.warning("MLflow param collision for %s: %s", safe_key, exc)

    def log_tag(self, key: str, value: Any) -> None:
        """Log run tag into MLflow."""
        mlflow.set_tag(self._sanitize_key(key), str(value))

    def log_artifact(self, local_path: str) -> None:
        """Log local file artifact into MLflow."""
        mlflow.log_artifact(local_path)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log PyTorch model into MLflow artifacts."""
        mlflow.pytorch.log_model(model, artifact_path)


    def _sanitize_key(self, key: str) -> str:
        """Sanitize MLflow key by replacing unsupported characters with underscores."""
        sanitized = re.sub(r"[^A-Za-z0-9_./-]", "_", str(key))
        if not sanitized:
            return "unnamed"
        return sanitized

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

