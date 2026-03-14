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
        task_rows = runtime.fetch_historic_task_events(
            process_name=process_name,
            version_key=version_filter,
            since=since,
            until=until,
        )
        identity_rows = runtime.fetch_identity_links(
            process_name=process_name,
            version_key=version_filter,
            since=since,
            until=until,
        )
        process_variable_rows = runtime.fetch_process_variables(
            process_name=process_name,
            version_key=version_filter,
            since=since,
            until=until,
        )
        events = self._enrich_events(
            events=events,
            task_rows=task_rows,
            identity_rows=identity_rows,
            process_variable_rows=process_variable_rows,
        )
        logger.info(
            "Camunda trace adapter fetched events=%d (source=%s, process=%s, version_filter=%s, dedup=%d, tasks=%d, identity=%d, proc_vars=%d).",
            len(events),
            runtime_cfg.get("runtime_source", "files"),
            process_name,
            version_filter or "<all>",
            diagnostics.rows_after_dedup,
            len(task_rows),
            len(identity_rows),
            len(process_variable_rows),
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
                    "assigned_executor": event.assigned_executor or "",
                    "executed_by": event.executed_by or "",
                    "potential_executor_users": ",".join(event.potential_executor_users or []),
                    "potential_executor_groups": ",".join(event.potential_executor_groups or []),
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
                    "call_activity_link": event.call_activity_link or {},
                    "process_variables": event.process_variables or {},
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
        if event.executed_by and str(event.executed_by).strip():
            return str(event.executed_by).strip()
        if event.assigned_executor and str(event.assigned_executor).strip():
            return str(event.assigned_executor).strip()
        if event.assignee and str(event.assignee).strip():
            return str(event.assignee).strip()
        if event.candidate_groups:
            first_group = str(event.candidate_groups[0]).strip()
            if first_group:
                return first_group
        return "UNKNOWN"

    @classmethod
    def _enrich_events(
        cls,
        *,
        events: List[ProcessEventDTO],
        task_rows: List[Dict[str, Any]],
        identity_rows: List[Dict[str, Any]],
        process_variable_rows: List[Dict[str, Any]],
    ) -> List[ProcessEventDTO]:
        assignee_by_task: Dict[str, str] = {}
        candidate_groups_by_task: Dict[str, set[str]] = defaultdict(set)
        candidate_users_by_task: Dict[str, set[str]] = defaultdict(set)
        process_vars_by_case_execution: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(dict)
        process_vars_by_case: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for row in task_rows:
            task_id = cls._norm_text(row.get("task_id"))
            assignee = cls._norm_text(row.get("assignee"))
            if task_id and assignee:
                assignee_by_task[task_id] = assignee

        for row in identity_rows:
            task_id = cls._norm_text(row.get("task_id"))
            if not task_id:
                continue
            link_type = cls._norm_text(row.get("link_type") or row.get("type")) or ""
            user_id = cls._norm_text(row.get("candidate_user_id") or row.get("user_id"))
            group_id = cls._norm_text(row.get("candidate_group_id") or row.get("group_id"))
            if link_type in {"assignee", "owner"} and user_id:
                assignee_by_task[task_id] = user_id
                continue
            if user_id:
                candidate_users_by_task[task_id].add(user_id)
            if group_id:
                candidate_groups_by_task[task_id].add(group_id)

        for row in process_variable_rows:
            case_id = cls._norm_text(row.get("case_id"))
            var_name = cls._norm_text(row.get("var_name") or row.get("name"))
            if not case_id or not var_name:
                continue
            value = cls._resolve_variable_value(row)
            execution_id = cls._norm_text(row.get("execution_id"))
            if execution_id:
                process_vars_by_case_execution[(case_id, execution_id)][var_name] = value
            else:
                process_vars_by_case[case_id][var_name] = value

        enriched: List[ProcessEventDTO] = []
        for event in events:
            task_id = cls._norm_text(event.task_id)
            case_id = cls._norm_text(event.case_id)
            execution_id = cls._norm_text(event.execution_id)

            potential_users = set(event.potential_executor_users or [])
            potential_groups = set(event.potential_executor_groups or event.candidate_groups or [])
            if task_id:
                potential_users.update(candidate_users_by_task.get(task_id, set()))
                potential_groups.update(candidate_groups_by_task.get(task_id, set()))

            assigned = event.assigned_executor or assignee_by_task.get(task_id) or event.assignee
            executed_by = event.executed_by or event.assignee or assigned

            process_variables = dict(process_vars_by_case.get(case_id, {}))
            if case_id and execution_id:
                process_variables.update(process_vars_by_case_execution.get((case_id, execution_id), {}))

            call_link = event.call_activity_link
            if event.call_proc_inst_id and not call_link:
                call_link = {
                    "parent_case_id": case_id,
                    "parent_execution_id": execution_id or None,
                    "parent_activity_def_id": event.activity_def_id,
                    "child_case_id": event.call_proc_inst_id,
                }

            enriched.append(
                event.model_copy(
                    update={
                        "assigned_executor": assigned,
                        "executed_by": executed_by,
                        "assignee": event.assignee or executed_by or assigned,
                        "potential_executor_users": sorted(potential_users) if potential_users else None,
                        "potential_executor_groups": sorted(potential_groups) if potential_groups else None,
                        "candidate_groups": sorted(potential_groups) if potential_groups else None,
                        "process_variables": process_variables or None,
                        "call_activity_link": call_link,
                    }
                )
            )
        return enriched

    @staticmethod
    def _resolve_variable_value(row: Dict[str, Any]) -> Any:
        for key in ("typed_value", "double_value", "long_value", "text_value", "text2_value", "value"):
            value = row.get(key)
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def _norm_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"", "nan", "nat", "none", "null", "<na>", "na", "n/a"}:
            return ""
        return text

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
