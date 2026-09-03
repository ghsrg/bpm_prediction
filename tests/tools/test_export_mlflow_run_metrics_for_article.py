from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools import export_mlflow_run_metrics_for_article as exporter


@dataclass
class FakeRunInfo:
    run_id: str
    experiment_id: str = "355"
    status: str = "FINISHED"
    lifecycle_stage: str = "active"
    start_time: int = 100
    end_time: int = 200


@dataclass
class FakeRunData:
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeRun:
    run_id: str
    status: str = "FINISHED"
    lifecycle_stage: str = "active"
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    experiment_id: str = "355"

    def __post_init__(self) -> None:
        self.info = FakeRunInfo(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            status=self.status,
            lifecycle_stage=self.lifecycle_stage,
        )
        self.data = FakeRunData(params=self.params, tags=self.tags, metrics=self.metrics)


@dataclass
class FakeMetricPoint:
    value: float
    step: int = 0
    timestamp: int = 123


class FakeMlflowClient:
    def __init__(self, runs: list[FakeRun]) -> None:
        self.runs = {run.info.run_id: run for run in runs}
        self.search_calls: list[dict[str, object]] = []

    def search_runs(self, *, experiment_ids: list[str]) -> list[FakeRun]:
        self.search_calls.append({"experiment_ids": experiment_ids})
        return list(self.runs.values())

    def get_run(self, run_id: str) -> FakeRun:
        return self.runs[run_id]

    def get_metric_history(self, run_id: str, metric_name: str) -> list[FakeMetricPoint]:
        run = self.runs[run_id]
        return [FakeMetricPoint(value=float(run.data.metrics[metric_name]))]


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch, client: FakeMlflowClient) -> None:
    mlflow_module = types.ModuleType("mlflow")
    mlflow_module.set_tracking_uri = lambda _uri: None  # type: ignore[attr-defined]
    tracking_module = types.ModuleType("mlflow.tracking")
    tracking_module.MlflowClient = lambda tracking_uri=None: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_module)


def test_normalize_paper_model_recognizes_mou_case_insensitively():
    assert exporter._normalize_paper_model(
        {"model.type": "mou"}, {"mlflow.runName": "MOU drift"}
    ) == "MOU"


def test_dataset_complexity_is_empty_when_parameter_is_absent():
    assert exporter._dataset_complexity({}, {}) == ""


def test_dataset_complexity_reads_explicit_parameter():
    assert exporter._dataset_complexity({"dataset_complexity": "complex"}, {}) == "complex"


def test_discover_runs_keeps_finished_active_runs_and_excludes_failed_killed_deleted():
    client = FakeMlflowClient(
        [
            FakeRun("finished", status="FINISHED", lifecycle_stage="active"),
            FakeRun("failed", status="FAILED", lifecycle_stage="active"),
            FakeRun("killed", status="KILLED", lifecycle_stage="active"),
            FakeRun("running", status="RUNNING", lifecycle_stage="active"),
            FakeRun("deleted", status="FINISHED", lifecycle_stage="deleted"),
        ]
    )

    assert exporter._discover_run_ids(client, ["355548395761513983"]) == ["finished"]
    assert client.search_calls == [{"experiment_ids": ["355548395761513983"]}]


def test_cli_rejects_multiple_run_selectors():
    assert exporter.main(["--runs-id", "abc", "--experiment-id", "355"]) == 2


def test_experiment_selection_writes_existing_learn_and_drift_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeMlflowClient(
        [
            FakeRun(
                "train-run",
                params={"experiment.mode": "train", "model.type": "UnknownArticleModel"},
                tags={"mlflow.runName": "Unknown train"},
                metrics={"strict_val_macro_f1": 0.71},
            ),
            FakeRun(
                "drift-run",
                params={"experiment.mode": "eval_drift", "model.type": "mou", "dataset_complexity": "middle"},
                tags={"mlflow.runName": "MOU drift"},
                metrics={"drift_window_strict_macro_f1": 0.42},
            ),
        ]
    )
    _install_fake_mlflow(monkeypatch, client)

    rc = exporter.main(
        [
            "--tracking-uri",
            "file:./mlruns",
            "--experiment-id",
            "355",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert (tmp_path / "learn" / "run_manifest.csv").exists()
    assert (tmp_path / "drift" / "run_manifest.csv").exists()
    assert (tmp_path / "learn" / "strict_val_macro_f1.csv").exists()
    assert (tmp_path / "drift" / "drift_window_strict_macro_f1.csv").exists()
    assert not (tmp_path / "all").exists()

    learn_manifest = (tmp_path / "learn" / "run_manifest.csv").read_text(encoding="utf-8")
    drift_metric = (tmp_path / "drift" / "drift_window_strict_macro_f1.csv").read_text(encoding="utf-8")
    assert "UnknownArticleModel" in learn_manifest
    assert "dataset_complexity" in drift_metric
    assert "middle" in drift_metric
    assert "MOU" in drift_metric


def test_experiment_selection_rejects_unrecognized_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeMlflowClient(
        [
            FakeRun(
                "unknown-mode",
                params={"experiment.mode": "predict", "model.type": "BaselineGATv2"},
                tags={"mlflow.runName": "Prediction only"},
                metrics={"accuracy": 0.1},
            )
        ]
    )
    _install_fake_mlflow(monkeypatch, client)

    assert exporter.main(["--experiment-id", "355", "--output-dir", str(tmp_path)]) == 1
    assert not (tmp_path / "all").exists()


def test_experiment_export_preserves_mou_metadata_and_metric_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeMlflowClient(
        [
            FakeRun(
                "mou-drift",
                params={
                    "experiment.mode": "eval_drift",
                    "model.type": "mou",
                    "dataset_complexity": "complex",
                    "preset_name": "_CDLG-complex1_MOU-drift",
                    "seed": "42",
                },
                tags={"mlflow.runName": "_CDLG-complex1_MOU-drift-42"},
                metrics={"drift_window_strict_macro_f1": 0.31},
            ),
            FakeRun(
                "gat-learn",
                params={
                    "experiment.mode": "learn",
                    "model.type": "BaselineGATv2",
                    "preset_name": "_CDLG-simple1_GATv2",
                },
                tags={"mlflow.runName": "_CDLG-simple1_GATv2-42"},
                metrics={"strict_val_macro_f1": 0.73},
            ),
        ]
    )
    _install_fake_mlflow(monkeypatch, client)

    assert exporter.main(["--experiment-id", "355", "--output-dir", str(tmp_path)]) == 0

    drift_metric = (tmp_path / "drift" / "drift_window_strict_macro_f1.csv").read_text(encoding="utf-8")
    learn_metric = (tmp_path / "learn" / "strict_val_macro_f1.csv").read_text(encoding="utf-8")
    drift_manifest = (tmp_path / "drift" / "run_manifest.csv").read_text(encoding="utf-8")
    learn_manifest = (tmp_path / "learn" / "run_manifest.csv").read_text(encoding="utf-8")

    assert "paper_model" in drift_metric
    assert "dataset_complexity" in drift_metric
    assert "MOU" in drift_metric
    assert "complex" in drift_metric
    assert "mou-drift" in drift_metric
    assert "_CDLG-complex1_MOU-drift" in drift_metric
    assert "experiment_id" in drift_metric
    assert "GATv2" in learn_metric
    assert "dataset_complexity" in learn_metric
    assert ",," in learn_metric
    assert "mou-drift" in drift_manifest
    assert "gat-learn" in learn_manifest
    assert not (tmp_path / "all").exists()
