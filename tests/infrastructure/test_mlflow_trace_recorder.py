from __future__ import annotations

import json
import os
from types import SimpleNamespace

from src.application.ports.trace_recorder_port import StructuralTraceEvent
from src.infrastructure.tracking import mlflow_trace_recorder
from src.infrastructure.tracking.mlflow_trace_recorder import MLflowTraceRecorder


def _event() -> StructuralTraceEvent:
    return StructuralTraceEvent(
        name="structural_prediction_debug",
        inputs={"sample": {"trace_idx": 1}},
        outputs={"prediction": {"pred_index": 2}},
        attributes={
            "stage": "eval_drift_one_pass",
            "fusion_mode": "StructXAttn",
            "reason": "strict_error_but_allowed",
            "strict_correct": False,
        },
    )


class _FakeSpan:
    def __init__(self, calls: list[dict], name: str, attributes: dict) -> None:
        self._calls = calls
        self._name = name
        self._attributes = attributes

    def __enter__(self):
        self._calls.append({"name": self._name, "attributes": self._attributes})
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_inputs(self, inputs):
        self._calls[-1]["inputs"] = inputs

    def set_outputs(self, outputs):
        self._calls[-1]["outputs"] = outputs


def test_mlflow_trace_recorder_uses_start_span_when_available(monkeypatch):
    calls: list[dict] = []

    def fake_start_span(*, name: str, attributes: dict):
        return _FakeSpan(calls, name, attributes)

    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "start_span", fake_start_span, raising=False)
    recorder = MLflowTraceRecorder(enabled=True)

    recorder.record(_event())

    assert calls[0]["name"] == "structural_prediction_debug"
    assert calls[0]["attributes"]["fusion_mode"] == "StructXAttn"
    assert calls[0]["attributes"]["reason"] == "strict_error_but_allowed"
    assert calls[0]["outputs"]["prediction"]["pred_index"] == 2


def test_mlflow_trace_recorder_noops_when_start_span_missing_and_no_fallback(monkeypatch, tmp_path):
    monkeypatch.delattr(mlflow_trace_recorder.mlflow, "start_span", raising=False)
    recorder = MLflowTraceRecorder(enabled=True, fallback_dir=None)

    recorder.record(_event())

    assert not list(tmp_path.glob("*.jsonl"))


def test_json_fallback_writes_jsonl_event(monkeypatch, tmp_path):
    monkeypatch.delattr(mlflow_trace_recorder.mlflow, "start_span", raising=False)
    monkeypatch.setenv("RANK", "0")
    recorder = MLflowTraceRecorder(enabled=True, fallback_dir=str(tmp_path), run_id="abc")

    recorder.record(_event())

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["attributes"]["fusion_mode"] == "StructXAttn"


def test_json_fallback_path_includes_pid_and_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "0")
    recorder = MLflowTraceRecorder(enabled=True, fallback_dir=str(tmp_path), run_id="abc")

    assert f"pid_{os.getpid()}" in str(recorder.fallback_json_path)
    assert "rank_0" in str(recorder.fallback_json_path)


def test_json_fallback_skips_nonzero_rank_by_default(monkeypatch, tmp_path):
    monkeypatch.delattr(mlflow_trace_recorder.mlflow, "start_span", raising=False)
    monkeypatch.setenv("RANK", "1")
    recorder = MLflowTraceRecorder(enabled=True, fallback_dir=str(tmp_path), run_id="abc")

    recorder.record(_event())

    assert not list(tmp_path.glob("*.jsonl"))
