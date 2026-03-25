from __future__ import annotations

import json

from src.infrastructure.runtime.progress_events import (
    PROGRESS_EVENT_PREFIX,
    ProgressReporter,
    emit_progress_event,
    progress_events_enabled,
)


def test_progress_events_disabled_by_default(monkeypatch, capsys) -> None:
    monkeypatch.delenv("BPM_PROGRESS_EVENTS", raising=False)
    assert progress_events_enabled() is False
    emit_progress_event(stage="x", message="hidden")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_progress_events_emit_when_enabled(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "1")
    assert progress_events_enabled() is True
    emit_progress_event(stage="prepare_data", status="update", current=2, total=4, message="running")
    captured = capsys.readouterr()
    line = captured.out.strip()
    assert line.startswith(PROGRESS_EVENT_PREFIX)
    payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
    assert payload["stage"] == "prepare_data"
    assert payload["status"] == "update"
    assert payload["percent"] == 50.0


def test_progress_events_done_status(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "true")
    emit_progress_event(stage="train.epochs", status="done", current=10, total=10, message="done")
    line = capsys.readouterr().out.strip()
    payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
    assert payload["status"] == "done"
    assert payload["percent"] == 100.0


def test_progress_reporter_start_update_done(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "1")
    reporter = ProgressReporter(stage="build_graph.train", total=10, min_interval_sec=0.0)
    reporter.start(message="start", current=0)
    reporter.update(message="mid", current=5)
    reporter.done(message="done", current=10)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3
    events = [json.loads(line[len(PROGRESS_EVENT_PREFIX) :]) for line in lines]
    assert events[0]["status"] == "start"
    assert events[1]["status"] == "update"
    assert events[1]["percent"] == 50.0
    assert events[2]["status"] == "done"
    assert events[2]["percent"] == 100.0


def test_progress_reporter_throttle(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "1")
    reporter = ProgressReporter(stage="train.batches", total=100, min_interval_sec=3600.0)
    reporter.start(message="start", current=0)
    emitted = reporter.update(message="skip", current=1)
    assert emitted is False
    emitted = reporter.update(message="force", current=1, force=True)
    assert emitted is True

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    events = [json.loads(line[len(PROGRESS_EVENT_PREFIX) :]) for line in lines]
    assert events[0]["status"] == "start"
    assert events[1]["status"] == "update"
