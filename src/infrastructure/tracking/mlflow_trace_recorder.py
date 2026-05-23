"""MLflow trace recorder adapter for selective structural prediction traces."""

from __future__ import annotations

import json
import logging
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Iterator

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
        experiment_id: str | None = None,
        allow_nonzero_rank_fallback: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.allow_nonzero_rank_fallback = bool(allow_nonzero_rank_fallback)
        self.rank = _resolve_rank()
        self.run_id = str(run_id or "").strip() or None
        self.experiment_id = str(experiment_id or "").strip() or None
        self.fallback_json_path: Path | None = None
        self.events_recorded = 0
        if fallback_dir:
            safe_run_id = _safe_token(str(self.run_id or "active"))
            rank_token = str(self.rank if self.rank is not None else 0)
            self.fallback_json_path = (
                Path(fallback_dir)
                / f"structural_traces_run_{safe_run_id}_pid_{os.getpid()}_rank_{rank_token}.jsonl"
            )

    def record(self, event: StructuralTraceEvent) -> None:
        if not self.enabled:
            return
        record_event = self._with_runtime_attributes(event)
        fallback_written = self._append_json_fallback(record_event)
        span_written = False
        start_span = getattr(mlflow, "start_span", None)
        if callable(start_span):
            try:
                with self._active_run_context_if_needed():
                    with _isolated_trace_id_random():
                        span = self._start_span(start_span, record_event)
                        with span as active_span:
                            active_span.set_inputs(record_event.inputs)
                            active_span.set_outputs(record_event.outputs)
                flush_trace_async_logging = getattr(mlflow, "flush_trace_async_logging", None)
                if callable(flush_trace_async_logging):
                    flush_trace_async_logging()
                span_written = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow trace span write failed; falling back when configured: %s", exc)
        if span_written or fallback_written:
            self.events_recorded += 1

    def _with_runtime_attributes(self, event: StructuralTraceEvent) -> StructuralTraceEvent:
        attributes = dict(event.attributes)
        active_run_id = _active_run_id()
        if self.run_id:
            attributes.setdefault("mlflow_expected_run_id", self.run_id)
        if self.experiment_id:
            attributes.setdefault("mlflow_expected_experiment_id", self.experiment_id)
        if active_run_id:
            attributes.setdefault("mlflow_active_run_id", active_run_id)
        else:
            attributes.setdefault("mlflow_active_run_id", "__none__")
        return replace(event, attributes=attributes)

    def _active_run_context_if_needed(self):
        if self.run_id and _active_run_id() is None:
            return mlflow.start_run(run_id=self.run_id)
        return nullcontext()

    def _start_span(self, start_span, event: StructuralTraceEvent):
        return start_span(name=event.name, attributes=dict(event.attributes))

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


def _active_run_id() -> str | None:
    try:
        active = mlflow.active_run()
    except Exception:  # noqa: BLE001
        return None
    if active is None or getattr(active, "info", None) is None:
        return None
    return str(getattr(active.info, "run_id", "") or "").strip() or None


@contextmanager
def _isolated_trace_id_random() -> Iterator[None]:
    """Avoid MLflow trace-id collisions without changing experiment RNG state."""
    state = random.getstate()
    random.seed(int.from_bytes(os.urandom(16), byteorder="big", signed=False))
    try:
        yield
    finally:
        random.setstate(state)


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value) or "active"
