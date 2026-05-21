"""MLflow trace recorder adapter for selective structural prediction traces."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow

from src.application.ports.trace_recorder_port import ITraceRecorder, StructuralTraceEvent


logger = logging.getLogger(__name__)


class MLflowTraceRecorder(ITraceRecorder):
    """Record structural trace events through MLflow spans or JSONL fallback."""

    def __init__(
        self,
        *,
        enabled: bool,
        fallback_dir: str | None = None,
        run_id: str | None = None,
        allow_nonzero_rank_fallback: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.allow_nonzero_rank_fallback = bool(allow_nonzero_rank_fallback)
        self.rank = _resolve_rank()
        self.fallback_json_path: Path | None = None
        self.events_recorded = 0
        if fallback_dir:
            safe_run_id = _safe_token(str(run_id or "active"))
            rank_token = str(self.rank if self.rank is not None else 0)
            self.fallback_json_path = (
                Path(fallback_dir)
                / f"structural_traces_run_{safe_run_id}_pid_{os.getpid()}_rank_{rank_token}.jsonl"
            )

    def record(self, event: StructuralTraceEvent) -> None:
        if not self.enabled:
            return
        start_span = getattr(mlflow, "start_span", None)
        if callable(start_span):
            try:
                with start_span(name=event.name, attributes=dict(event.attributes)) as span:
                    span.set_inputs(event.inputs)
                    span.set_outputs(event.outputs)
                self.events_recorded += 1
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow trace span write failed; falling back when configured: %s", exc)
        if self._append_json_fallback(event):
            self.events_recorded += 1

    def _append_json_fallback(self, event: StructuralTraceEvent) -> bool:
        if self.fallback_json_path is None:
            return False
        if self.rank not in (None, 0) and not self.allow_nonzero_rank_fallback:
            return False
        self.fallback_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_json_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return True


def _resolve_rank() -> int | None:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value) or "active"
