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

