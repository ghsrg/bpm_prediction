from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.infrastructure.tracking import mlflow_tracker
from src.infrastructure.tracking.mlflow_tracker import MLflowTracker


def test_log_model_uses_name_for_mlflow3_signature(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_log_model(pytorch_model: Any, artifact_path: str | None = None, *, name: str | None = None) -> None:
        calls.append({"model": pytorch_model, "artifact_path": artifact_path, "name": name})

    monkeypatch.setattr(mlflow_tracker.mlflow, "pytorch", SimpleNamespace(log_model=fake_log_model))
    tracker = MLflowTracker.__new__(MLflowTracker)
    model = object()

    tracker.log_model(model, "best_model")

    assert calls == [{"model": model, "artifact_path": None, "name": "best_model"}]


def test_log_model_keeps_artifact_path_for_mlflow2_signature(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_log_model(pytorch_model: Any, artifact_path: str) -> None:
        calls.append({"model": pytorch_model, "artifact_path": artifact_path})

    monkeypatch.setattr(mlflow_tracker.mlflow, "pytorch", SimpleNamespace(log_model=fake_log_model))
    tracker = MLflowTracker.__new__(MLflowTracker)
    model = object()

    tracker.log_model(model, "best_model")

    assert calls == [{"model": model, "artifact_path": "best_model"}]
