"""Structured progress events for UI-friendly run status tracking.

Events are emitted to stdout as single-line JSON payloads with a fixed prefix.
They are opt-in and enabled only when environment variable
`BPM_PROGRESS_EVENTS` is set to a truthy value.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict


PROGRESS_EVENT_PREFIX = "__BPM_PROGRESS__"


def progress_events_enabled() -> bool:
    """Return True when structured progress output is enabled."""
    raw = str(os.getenv("BPM_PROGRESS_EVENTS", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def emit_progress_event(
    *,
    stage: str,
    status: str = "update",
    message: str = "",
    current: int | float | None = None,
    total: int | float | None = None,
    level: str = "info",
    payload: Dict[str, Any] | None = None,
) -> None:
    """Emit one structured progress event to stdout.

    Output format:
    __BPM_PROGRESS__{"kind":"progress","stage":"...","status":"...","..."}
    """
    if not progress_events_enabled():
        return

    event: Dict[str, Any] = {
        "kind": "progress",
        "stage": str(stage).strip() or "unknown",
        "status": str(status).strip() or "update",
        "level": str(level).strip() or "info",
        "message": str(message),
        "ts": float(time.time()),
    }
    if current is not None:
        event["current"] = float(current)
    if total is not None:
        event["total"] = float(total)
        if isinstance(current, (int, float)) and float(total) > 0:
            event["percent"] = max(0.0, min(100.0, (float(current) / float(total)) * 100.0))
    if payload:
        event["payload"] = payload

    line = f"{PROGRESS_EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        # Best-effort telemetry; never fail the main pipeline.
        return


class ProgressReporter:
    """Stateful helper over emit_progress_event with throttled updates."""

    def __init__(
        self,
        stage: str,
        *,
        total: int | float | None = None,
        min_interval_sec: float = 0.8,
        default_level: str = "info",
    ) -> None:
        self.stage = str(stage).strip() or "unknown"
        self.total: float | None = self._to_float(total)
        self.current: float | None = None
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.default_level = str(default_level).strip() or "info"
        self._started = False
        self._last_emit_monotonic = 0.0

    @staticmethod
    def _to_float(value: int | float | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def start(
        self,
        *,
        message: str = "",
        current: int | float | None = None,
        total: int | float | None = None,
        level: str | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        self.current = self._to_float(current) if current is not None else self.current
        self.total = self._to_float(total) if total is not None else self.total
        self._started = True
        self._last_emit_monotonic = time.monotonic()
        emit_progress_event(
            stage=self.stage,
            status="start",
            message=message,
            current=self.current,
            total=self.total,
            level=(level or self.default_level),
            payload=payload,
        )

    def update(
        self,
        *,
        message: str = "",
        current: int | float | None = None,
        total: int | float | None = None,
        level: str | None = None,
        payload: Dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        if not self._started:
            self.start(message=message, current=current, total=total, level=level, payload=payload)
            return True

        if current is not None:
            self.current = self._to_float(current)
        if total is not None:
            self.total = self._to_float(total)

        now = time.monotonic()
        if not force and (now - self._last_emit_monotonic) < self.min_interval_sec:
            return False

        self._last_emit_monotonic = now
        emit_progress_event(
            stage=self.stage,
            status="update",
            message=message,
            current=self.current,
            total=self.total,
            level=(level or self.default_level),
            payload=payload,
        )
        return True

    def advance(
        self,
        step: int | float = 1.0,
        *,
        message: str = "",
        total: int | float | None = None,
        level: str | None = None,
        payload: Dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        base = self.current if self.current is not None else 0.0
        self.current = base + float(step)
        return self.update(
            message=message,
            current=self.current,
            total=total,
            level=level,
            payload=payload,
            force=force,
        )

    def done(
        self,
        *,
        message: str = "",
        current: int | float | None = None,
        total: int | float | None = None,
        level: str | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        if current is not None:
            self.current = self._to_float(current)
        elif self.current is None and self.total is not None:
            self.current = float(self.total)
        if total is not None:
            self.total = self._to_float(total)

        emit_progress_event(
            stage=self.stage,
            status="done",
            message=message,
            current=self.current,
            total=self.total,
            level=(level or self.default_level),
            payload=payload,
        )

    def error(
        self,
        *,
        message: str,
        level: str = "error",
        payload: Dict[str, Any] | None = None,
    ) -> None:
        emit_progress_event(
            stage=self.stage,
            status="error",
            message=message,
            current=self.current,
            total=self.total,
            level=(level or "error"),
            payload=payload,
        )
