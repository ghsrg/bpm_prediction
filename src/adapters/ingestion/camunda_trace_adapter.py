"""IXESAdapter-compatible bridge from Camunda runtime exports to RawTrace."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, Iterator, List, Optional

from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter
from src.application.ports.xes_adapter_port import IXESAdapter
from src.domain.entities.event_record import EventRecord
from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.raw_trace import RawTrace


logger = logging.getLogger(__name__)


class CamundaTraceAdapter(IXESAdapter):
    """Adapter that converts Camunda runtime rows into canonical RawTrace objects."""

    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        """Read runtime events and emit ordered traces."""
        _ = file_path  # Kept for IXESAdapter compatibility.
        cfg = self._resolve_camunda_config(mapping_config)
        runtime_cfg = self._resolve_runtime_config(cfg)

        process_name = str(
            cfg.get("process_name")
            or cfg.get("dataset_name")
            or mapping_config.get("dataset_name")
            or mapping_config.get("dataset_label")
            or "default_process"
        ).strip() or "default_process"
        version_filter = str(cfg.get("version_key", "")).strip()
        dataset_fallback_version = str(
            cfg.get("dataset_name")
            or mapping_config.get("dataset_name")
            or mapping_config.get("dataset_label")
            or process_name
        ).strip() or process_name

        since, until = self._resolve_time_range(cfg)
        runtime = CamundaRuntimeAdapter(runtime_cfg)
        events, diagnostics = runtime.fetch_historic_activity_events(
            process_name=process_name,
            version_key=version_filter,
            since=since,
            until=until,
        )
        logger.info(
            "Camunda trace adapter fetched events=%d (source=%s, process=%s, version_filter=%s, dedup=%d).",
            len(events),
            runtime_cfg.get("runtime_source", "files"),
            process_name,
            version_filter or "<all>",
            diagnostics.rows_after_dedup,
        )

        traces = self._group_events_to_traces(
            events=events,
            dataset_fallback_version=dataset_fallback_version,
            process_name=process_name,
            version_filter=version_filter,
        )
        for trace in traces:
            yield trace

    @staticmethod
    def _resolve_camunda_config(mapping_config: Dict[str, Any]) -> Dict[str, Any]:
        block = mapping_config.get("camunda_adapter", {})
        return dict(block) if isinstance(block, dict) else {}

    @staticmethod
    def _resolve_runtime_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        runtime_cfg = cfg.get("runtime", {})
        runtime = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}
        if not runtime:
            for key in (
                "runtime_source",
                "export_dir",
                "sql_dir",
                "history_cleanup_aware",
                "legacy_removal_time_policy",
                "on_missing_removal_time",
                "mssql",
            ):
                if key in cfg:
                    runtime[key] = cfg[key]
        runtime.setdefault("runtime_source", "files")
        runtime.setdefault("history_cleanup_aware", True)
        runtime.setdefault("legacy_removal_time_policy", "treat_as_eternal")
        runtime.setdefault("on_missing_removal_time", "auto_fallback")
        return runtime

    def _resolve_time_range(self, cfg: Dict[str, Any]) -> tuple[Optional[datetime], Optional[datetime]]:
        since = self._to_utc_naive(cfg.get("since"))
        until = self._to_utc_naive(cfg.get("until"))
        if since is not None or until is not None:
            return since, until

        lookback_hours = int(cfg.get("lookback_hours", 0) or 0)
        if lookback_hours <= 0:
            return None, None
        now = datetime.utcnow()
        return now - timedelta(hours=lookback_hours), now

    def _group_events_to_traces(
        self,
        *,
        events: List[ProcessEventDTO],
        dataset_fallback_version: str,
        process_name: str,
        version_filter: str,
    ) -> List[RawTrace]:
        by_case: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
        for event in events:
            case_id = str(event.case_id).strip() or "unknown_case"
            by_case[case_id].append(event)

        traces: List[RawTrace] = []
        for case_id, case_events in by_case.items():
            ordered = sorted(case_events, key=self._event_sort_key)
            normalized_events: List[EventRecord] = []
            first_ts: Optional[float] = None
            prev_ts: Optional[float] = None

            for idx, event in enumerate(ordered):
                ts_dt = event.start_time or event.end_time
                if ts_dt is None:
                    continue
                ts = float(ts_dt.timestamp())
                if first_ts is None:
                    first_ts = ts
                duration_sec = float(event.duration_ms or 0.0) / 1000.0
                resource_id = self._resolve_resource_id(event)
                since_start = float(ts - first_ts)
                since_prev = 0.0 if prev_ts is None else float(ts - prev_ts)
                prev_ts = ts

                extra = {
                    "concept:name": str(event.activity_def_id),
                    "activity": str(event.activity_def_id),
                    "activity_def_id": str(event.activity_def_id),
                    "activity_name": event.activity_name or str(event.activity_def_id),
                    "activity_type": event.activity_type or "",
                    "org:resource": resource_id,
                    "resource_id": resource_id,
                    "assignee": event.assignee or "",
                    "candidate_groups": ",".join(event.candidate_groups or []),
                    "time:timestamp": ts,
                    "duration": duration_sec,
                    "time_since_case_start": since_start,
                    "time_since_previous_event": since_prev,
                    "task_id": event.task_id or "",
                    "act_inst_id": event.act_inst_id or "",
                    "execution_id": event.execution_id or "",
                    "parent_execution_id": event.parent_execution_id or "",
                    "call_proc_inst_id": event.call_proc_inst_id or "",
                    "proc_def_id": event.proc_def_id or "",
                    "proc_def_key": event.proc_def_key or "",
                    "proc_def_version": event.proc_def_version or "",
                }
                extra.update(event.extra or {})

                normalized_events.append(
                    EventRecord(
                        activity_id=str(event.activity_def_id),
                        timestamp=ts,
                        resource_id=resource_id,
                        lifecycle="complete",
                        position_in_trace=idx,
                        duration=duration_sec,
                        time_since_case_start=since_start,
                        time_since_previous_event=since_prev,
                        extra=extra,
                        activity_instance_id=event.act_inst_id,
                    )
                )

            if len(normalized_events) < 2:
                continue

            first = ordered[0]
            trace_version = (
                str(first.proc_def_version or "").strip()
                or version_filter
                or dataset_fallback_version
                or process_name
            )

            trace_attributes = {
                "process_name": process_name,
                "proc_def_id": first.proc_def_id or "",
                "proc_def_key": first.proc_def_key or "",
                "proc_def_version": first.proc_def_version or "",
                "version_key": trace_version,
            }

            traces.append(
                RawTrace(
                    case_id=case_id,
                    process_version=trace_version,
                    events=normalized_events,
                    trace_attributes=trace_attributes,
                )
            )

        traces.sort(key=lambda tr: tr.events[0].timestamp if tr.events else float("inf"))
        return traces

    @staticmethod
    def _event_sort_key(event: ProcessEventDTO) -> tuple[float, float]:
        start = event.start_time.timestamp() if event.start_time is not None else float("inf")
        end = event.end_time.timestamp() if event.end_time is not None else start
        return start, end

    @staticmethod
    def _resolve_resource_id(event: ProcessEventDTO) -> str:
        if event.assignee and str(event.assignee).strip():
            return str(event.assignee).strip()
        if event.candidate_groups:
            first_group = str(event.candidate_groups[0]).strip()
            if first_group:
                return first_group
        return "UNKNOWN"

    @staticmethod
    def _to_utc_naive(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            text = str(value).strip()
            if not text:
                return None
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None

        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
