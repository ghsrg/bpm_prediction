from __future__ import annotations

import json

from src.infrastructure.runtime.progress_events import (
    PROGRESS_EVENT_PREFIX,
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

