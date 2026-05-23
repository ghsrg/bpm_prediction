from __future__ import annotations

import json
import os
import random
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


class _FakeRunContext:
    def __init__(self, calls: list[str], run_id: str) -> None:
        self._calls = calls
        self._run_id = run_id

    def __enter__(self):
        self._calls.append(self._run_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_mlflow_trace_recorder_uses_start_span_when_available(monkeypatch, tmp_path):
    calls: list[dict] = []
    flush_calls: list[bool] = []

    def fake_start_span(*, name: str, attributes: dict):
        return _FakeSpan(calls, name, attributes)

    def fake_flush_trace_async_logging():
        flush_calls.append(True)

    def fake_active_run():
        return SimpleNamespace(info=SimpleNamespace(run_id="abc", experiment_id="exp-1"))

    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "active_run", fake_active_run, raising=False)
    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "start_span", fake_start_span, raising=False)
    monkeypatch.setattr(
        mlflow_trace_recorder.mlflow,
        "flush_trace_async_logging",
        fake_flush_trace_async_logging,
        raising=False,
    )
    recorder = MLflowTraceRecorder(
        enabled=True,
        fallback_dir=str(tmp_path),
        run_id="abc",
        experiment_id="exp-1",
    )

    recorder.record(_event())

    assert calls[0]["name"] == "structural_prediction_debug"
    assert calls[0]["attributes"]["fusion_mode"] == "StructXAttn"
    assert calls[0]["attributes"]["reason"] == "strict_error_but_allowed"
    assert calls[0]["attributes"]["mlflow_expected_run_id"] == "abc"
    assert calls[0]["attributes"]["mlflow_active_run_id"] == "abc"
    assert calls[0]["outputs"]["prediction"]["pred_index"] == 2
    assert flush_calls == [True]
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert row["attributes"]["mlflow_expected_experiment_id"] == "exp-1"


def test_mlflow_trace_recorder_reopens_expected_run_when_no_active_run(monkeypatch):
    calls: list[dict] = []
    start_run_calls: list[str] = []

    def fake_start_span(*, name: str, attributes: dict):
        return _FakeSpan(calls, name, attributes)

    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "active_run", lambda: None, raising=False)
    monkeypatch.setattr(
        mlflow_trace_recorder.mlflow,
        "start_run",
        lambda *, run_id: _FakeRunContext(start_run_calls, run_id),
        raising=False,
    )
    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "start_span", fake_start_span, raising=False)
    monkeypatch.delattr(mlflow_trace_recorder.mlflow, "flush_trace_async_logging", raising=False)
    recorder = MLflowTraceRecorder(enabled=True, run_id="expected-run")

    recorder.record(_event())

    assert start_run_calls == ["expected-run"]
    assert calls[0]["attributes"]["mlflow_expected_run_id"] == "expected-run"
    assert calls[0]["attributes"]["mlflow_active_run_id"] == "__none__"


def test_mlflow_trace_recorder_isolates_mlflow_trace_id_random(monkeypatch):
    consumed_trace_ids: list[int] = []

    def fake_start_span(*, name: str, attributes: dict):
        consumed_trace_ids.append(random.getrandbits(128))
        return _FakeSpan([], name, attributes)

    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "active_run", lambda: None, raising=False)
    monkeypatch.setattr(mlflow_trace_recorder.mlflow, "start_span", fake_start_span, raising=False)
    monkeypatch.delattr(mlflow_trace_recorder.mlflow, "flush_trace_async_logging", raising=False)

    random.seed(42)
    first_seeded_value = random.getrandbits(128)
    random.seed(42)

    recorder = MLflowTraceRecorder(enabled=True)
    recorder.record(_event())

    assert random.getrandbits(128) == first_seeded_value
    assert consumed_trace_ids
    assert consumed_trace_ids[0] != first_seeded_value


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
