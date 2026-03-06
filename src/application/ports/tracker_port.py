"""Application port contract for metric tracking (e.g., MLflow)."""

from __future__ import annotations

from typing import Any, Optional, Protocol


class ITracker(Protocol):
    """Port for logging experiment metrics and parameters."""

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a numeric metric."""
        ...

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter (e.g., hyperparameters, model name)."""
        ...

    def log_tag(self, key: str, value: Any) -> None:
        """Log a run tag for grouping/filtering experiments."""
        ...

    def log_artifact(self, path: str) -> None:
        """Log local file artifact for reproducibility."""
        ...
