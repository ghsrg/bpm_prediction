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

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ) -> None:
        self._known_params: Dict[str, str] = {}
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        resume_id = str(resume_run_id or "").strip()
        if resume_id:
            mlflow.start_run(run_id=resume_id)
            mlflow.set_tag("resume_run_id", resume_id)
            try:
                existing_run = mlflow.get_run(resume_id)
                params = getattr(existing_run.data, "params", {}) if existing_run is not None else {}
                if isinstance(params, dict):
                    self._known_params.update({str(k): str(v) for k, v in params.items()})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not preload existing MLflow params for resumed run %s: %s", resume_id, exc)
            logger.info("Resumed existing MLflow run: run_id=%s", resume_id)
        else:
            mlflow.start_run(run_name=run_name)

    def get_run_id(self) -> Optional[str]:
        """Return active MLflow run id if available."""
        active = mlflow.active_run()
        if active is None or active.info is None:
            return None
        return str(active.info.run_id or "").strip() or None

    @staticmethod
    def find_latest_run_id(
        *,
        experiment_name: str,
        run_name: str,
        tracking_uri: Optional[str] = None,
    ) -> Optional[str]:
        """Find latest MLflow run id by experiment name and run name tag."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        exp_name = str(experiment_name or "").strip()
        rn = str(run_name or "").strip()
        if not exp_name or not rn:
            return None
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[str(experiment.experiment_id)],
            filter_string=f"tags.`mlflow.runName` = '{rn}'",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if runs is None or runs.empty:
            return None
        run_id = str(runs.iloc[0].get("run_id", "")).strip()
        return run_id or None

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
            normalized_value = str(flat_value)
            existing_value = self._known_params.get(safe_key)
            if existing_value is not None:
                if existing_value != normalized_value:
                    logger.info(
                        "Skip MLflow param update for %s (existing=%s, attempted=%s).",
                        safe_key,
                        existing_value,
                        normalized_value,
                    )
                continue
            try:
                mlflow.log_param(safe_key, flat_value)
                self._known_params[safe_key] = normalized_value
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
            normalized_value = str(safe_value)
            existing_value = self._known_params.get(safe_key)
            if existing_value is not None:
                if existing_value != normalized_value:
                    logger.info(
                        "Skip MLflow param update for %s (existing=%s, attempted=%s).",
                        safe_key,
                        existing_value,
                        normalized_value,
                    )
                continue
            try:
                mlflow.log_param(safe_key, safe_value)
                self._known_params[safe_key] = normalized_value
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

