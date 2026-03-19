"""Camunda Stage 3.1 runtime adapter with source switch: files or MSSQL."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO
from src.domain.ports.camunda_runtime_port import ICamundaRuntimePort
from src.infrastructure.config.connection_resolver import (
    resolve_mssql_connection_string,
)


logger = logging.getLogger(__name__)


class CamundaRuntimeAdapter(ICamundaRuntimePort):
    """Read Camunda runtime datasets from manual file exports or MSSQL."""

    _TABLE_TO_BASE = {
        "historic_activity_events": "historic_activity_events",
        "historic_tasks": "historic_tasks",
        "identity_links": "identity_links",
        "execution_tree": "execution_tree",
        "multi_instance_variables": "multi_instance_variables",
        "process_variables": "process_variables",
        "process_instance_links": "process_instance_links",
    }
    _NULLISH_TEXT = {"", "nan", "nat", "none", "null", "<na>", "na", "n/a"}

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        source = str(self.config.get("runtime_source", "files")).strip().lower()
        self.runtime_source = source if source in {"files", "mssql"} else "files"
        self.export_dir = Path(str(self.config.get("export_dir", "data/camunda_exports"))).resolve()
        self.sql_dir = Path(str(self.config.get("sql_dir", "src/adapters/ingestion/camunda/sql/runtime"))).resolve()
        self.history_cleanup_aware = bool(self.config.get("history_cleanup_aware", True))
        self.legacy_removal_time_policy = str(self.config.get("legacy_removal_time_policy", "treat_as_eternal")).strip().lower()
        self.on_missing_removal_time = str(self.config.get("on_missing_removal_time", "auto_fallback")).strip().lower()

        mssql_cfg = self.config.get("mssql", {})
        if not isinstance(mssql_cfg, dict):
            mssql_cfg = {}
        self.mssql_connection_string = resolve_mssql_connection_string(
            cfg={
                "mssql": mssql_cfg,
                "connections_file": self.config.get("connections_file"),
                "connection_profile": self.config.get("connection_profile"),
                "profile": self.config.get("profile"),
            }
        )

    def fetch_historic_activity_events(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Tuple[List[ProcessEventDTO], RuntimeFetchDiagnosticsDTO]:
        rows = self._fetch_table("historic_activity_events")
        scoped = self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)
        rows_raw = len(scoped)

        filtered, supported_removal, warnings = self._apply_removal_time_guard(scoped)
        rows_after_cleanup = len(filtered)

        events = [self._row_to_event(row) for row in filtered]
        events = self._deduplicate_events(events)
        rows_after_dedup = len(events)

        cleaned_percent = 0.0
        coverage_percent: Optional[float] = None
        if rows_raw > 0:
            cleaned_percent = float(((rows_raw - rows_after_cleanup) / rows_raw) * 100.0)
            coverage_percent = float((rows_after_cleanup / rows_raw) * 100.0)

        diagnostics = RuntimeFetchDiagnosticsDTO(
            rows_raw=rows_raw,
            rows_after_cleanup_filter=rows_after_cleanup,
            rows_after_dedup=rows_after_dedup,
            cleaned_instances_percent=cleaned_percent,
            history_coverage_percent=coverage_percent,
            fallback_triggered=False,
            fallback_reason=None,
            removal_time_supported_tables=["historic_activity_events"] if supported_removal else [],
            warnings=warnings,
            meta={"runtime_source": self.runtime_source},
        )
        return events, diagnostics

    def fetch_historic_task_events(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_table("historic_tasks")
        return self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)

    def fetch_identity_links(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_table("identity_links")
        return self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)

    def fetch_runtime_execution_tree(
        self,
        process_name: str,
        version_key: str,
        depth_limit: int,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_table("execution_tree")
        scoped = self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)
        if depth_limit < 0:
            return scoped
        if not scoped:
            return scoped
        if "depth" in scoped[0]:
            return [row for row in scoped if self._safe_int(row.get("depth")) <= depth_limit]
        if "scope_depth" in scoped[0]:
            return [row for row in scoped if self._safe_int(row.get("scope_depth")) <= depth_limit]
        return scoped

    def fetch_multi_instance_variables(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_table("multi_instance_variables")
        return self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)

    def fetch_process_variables(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_table("process_variables")
        return self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)

    def fetch_process_instance_links(
        self,
        process_name: str,
        version_key: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_table("process_instance_links")
        scoped = self._filter_scope(rows, process_name=process_name, version_key=version_key, since=since, until=until)
        normalized: List[Dict[str, Any]] = []
        for row in scoped:
            item = dict(row)
            super_id = self._str_or_none(
                item.get("super_proc_inst_id")
                or item.get("super_process_instance_id")
                or item.get("SUPER_PROCESS_INSTANCE_ID_")
            )
            if super_id is not None:
                item["super_proc_inst_id"] = super_id
                item["super_process_instance_id"] = super_id
            normalized.append(item)
        return normalized

    def _fetch_table(self, table_name: str) -> List[Dict[str, Any]]:
        if self.runtime_source == "mssql":
            return self._fetch_table_mssql(table_name)
        return self._fetch_table_files(table_name)

    def _fetch_table_files(self, table_name: str) -> List[Dict[str, Any]]:
        base_name = self._TABLE_TO_BASE[table_name]
        file_path = self._find_input_file(base_name)
        if file_path is None:
            logger.warning("Camunda files source: '%s' file was not found in %s", base_name, self.export_dir)
            return []

        try:
            if file_path.suffix.lower() == ".csv":
                frame = pd.read_csv(file_path)
            else:
                frame = pd.read_excel(file_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return []

        if frame.empty:
            return []
        frame = frame.where(pd.notna(frame), None)
        return frame.to_dict(orient="records")

    def _fetch_table_mssql(self, table_name: str) -> List[Dict[str, Any]]:
        sql_file = self.sql_dir / f"{self._TABLE_TO_BASE[table_name]}.sql"
        if not sql_file.exists():
            logger.warning("MSSQL source: SQL file not found for %s: %s", table_name, sql_file)
            return []
        if not self.mssql_connection_string:
            logger.warning("MSSQL source selected but connection string is empty.")
            return []

        try:
            import pyodbc  # type: ignore
        except Exception:
            logger.warning("MSSQL source selected but 'pyodbc' is not installed. Returning empty dataset.")
            return []

        sql_text = sql_file.read_text(encoding="utf-8")
        connection = None
        try:
            connection = pyodbc.connect(self.mssql_connection_string)
            frame = pd.read_sql_query(sql_text, connection)
            if frame.empty:
                return []
            frame = frame.where(pd.notna(frame), None)
            return frame.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Failed to query MSSQL table '%s': %s", table_name, exc)
            return []
        finally:
            if connection is not None:
                connection.close()

    def _find_input_file(self, base_name: str) -> Optional[Path]:
        if not self.export_dir.exists():
            return None

        # 1) Fast path: exact canonical/mock names.
        exact_candidates = [
            self.export_dir / f"{base_name}.csv",
            self.export_dir / f"{base_name}.xlsx",
            self.export_dir / f"{base_name}.xls",
            self.export_dir / f"mock_{base_name}.csv",
            self.export_dir / f"mock_{base_name}.xlsx",
            self.export_dir / f"mock_{base_name}.xls",
        ]
        for path in exact_candidates:
            if path.exists():
                return path

        # 2) Flexible fallback: allow slight naming variations like:
        #    historic_activity_events_2026-03.csv, export_historic_tasks.xlsx, etc.
        allowed_ext = {".csv", ".xlsx", ".xls"}
        matches: List[Path] = []
        for path in self.export_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in allowed_ext:
                continue
            name = path.stem.lower()
            if base_name.lower() in name:
                matches.append(path)

        if not matches:
            return None

        matches.sort(key=lambda item: (self._file_name_priority(item.stem.lower(), base_name.lower()), item.name))
        return matches[0]

    @staticmethod
    def _file_name_priority(stem: str, base_name: str) -> int:
        """Lower is better; prefer closest match first."""
        if stem == base_name:
            return 0
        if stem == f"mock_{base_name}":
            return 1
        if stem.startswith(base_name):
            return 2
        if stem.startswith(f"mock_{base_name}"):
            return 3
        if base_name in stem:
            return 4
        return 99

    def _apply_removal_time_guard(
        self,
        rows: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
        warnings: List[str] = []
        if not rows or not self.history_cleanup_aware:
            return rows, False, warnings

        removal_col = self._find_removal_time_column(rows[0].keys())
        if removal_col is None:
            if self.on_missing_removal_time == "strict":
                warnings.append("removal_time_missing_strict")
                return [], False, warnings
            warnings.append("removal_time_missing_auto_fallback")
            return rows, False, warnings

        parsed_values = [self._parse_datetime(row.get(removal_col)) for row in rows]
        non_null_count = sum(1 for value in parsed_values if value is not None)
        if non_null_count == 0 and self.legacy_removal_time_policy == "treat_as_eternal":
            warnings.append("legacy_removal_time_treated_as_eternal")
            return rows, True, warnings

        now = datetime.utcnow()
        kept: List[Dict[str, Any]] = []
        for row, removal_ts in zip(rows, parsed_values):
            if removal_ts is None or removal_ts > now:
                kept.append(row)
        return kept, True, warnings

    @staticmethod
    def _find_removal_time_column(columns: Iterable[Any]) -> Optional[str]:
        normalized = {str(col).lower(): str(col) for col in columns}
        for candidate in ("removal_time_", "removal_time"):
            if candidate in normalized:
                return normalized[candidate]
        return None

    def _filter_scope(
        self,
        rows: List[Dict[str, Any]],
        *,
        process_name: str,
        version_key: str,
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []
        filtered: List[Dict[str, Any]] = []
        process_name = str(process_name).strip()
        version_key = str(version_key).strip()
        for row in rows:
            if process_name and not self._matches_process_name(row, process_name):
                continue
            if version_key and not self._matches_version(row, version_key):
                continue
            event_ts = self._resolve_event_ts(row)
            if since is not None and event_ts is not None and event_ts < since:
                continue
            if until is not None and event_ts is not None and event_ts > until:
                continue
            filtered.append(row)
        return filtered

    @staticmethod
    def _matches_process_name(row: Dict[str, Any], process_name: str) -> bool:
        keys = ("process_name", "proc_def_key", "dataset_name")
        values = [str(row.get(key, "")).strip() for key in keys]
        non_empty = [value for value in values if value]
        if not non_empty:
            return True
        return process_name in non_empty

    @staticmethod
    def _matches_version(row: Dict[str, Any], version_key: str) -> bool:
        keys = ("version_key", "proc_def_version", "process_version")
        values = [str(row.get(key, "")).strip() for key in keys]
        non_empty = [value for value in values if value]
        if not non_empty:
            return True
        return version_key in non_empty

    def _row_to_event(self, row: Dict[str, Any]) -> ProcessEventDTO:
        candidate_groups = self._parse_string_list(
            row.get("candidate_groups")
            or row.get("potential_executor_groups")
            or row.get("candidate_group_ids")
        )
        potential_users = self._parse_string_list(
            row.get("potential_executor_users")
            or row.get("candidate_user_ids")
            or row.get("candidate_user_id")
        )
        potential_groups = self._parse_string_list(
            row.get("potential_executor_groups")
            or row.get("candidate_group_ids")
            or row.get("candidate_group_id")
            or row.get("candidate_groups")
        )
        assigned_executor = self._choose_non_empty(
            row.get("assigned_executor"),
            row.get("assigned_user_id"),
            row.get("assignee"),
        )
        executed_by = self._choose_non_empty(
            row.get("executed_by"),
            row.get("executed_by_user_id"),
            row.get("completed_by"),
            row.get("assignee"),
        )

        return ProcessEventDTO(
            case_id=str(row.get("case_id", "")).strip() or "unknown_case",
            activity_def_id=str(row.get("activity_def_id", "")).strip() or "unknown_activity",
            activity_name=self._str_or_none(row.get("activity_name")),
            activity_type=self._str_or_none(row.get("activity_type")),
            proc_def_id=self._str_or_none(row.get("proc_def_id")),
            proc_def_key=self._str_or_none(row.get("proc_def_key")),
            proc_def_version=self._str_or_none(row.get("proc_def_version")),
            task_id=self._str_or_none(row.get("task_id")),
            act_inst_id=self._str_or_none(row.get("act_inst_id")),
            parent_act_inst_id=self._str_or_none(row.get("parent_act_inst_id")),
            sequence_counter=self._safe_int_or_none(row.get("sequence_counter")),
            execution_id=self._str_or_none(row.get("execution_id")),
            parent_execution_id=self._str_or_none(row.get("parent_execution_id")),
            scope_depth=self._safe_int_or_none(row.get("scope_depth") or row.get("depth")),
            is_concurrent=self._safe_bool(row.get("is_concurrent")),
            is_scope=self._safe_bool(row.get("is_scope")),
            is_event_scope=self._safe_bool(row.get("is_event_scope")),
            execution_rev=self._safe_int_or_none(row.get("rev") or row.get("execution_rev")),
            call_proc_inst_id=self._str_or_none(row.get("call_proc_inst_id")),
            called_element=self._str_or_none(row.get("called_element")),
            binding_type=self._str_or_none(row.get("binding_type")),
            version_tag=self._str_or_none(row.get("version_tag")),
            version_number=self._str_or_none(row.get("version_number")),
            resolved_child_proc_def_id=self._str_or_none(row.get("resolved_child_proc_def_id")),
            child_process_key=self._str_or_none(row.get("child_process_key")),
            child_version=self._str_or_none(row.get("child_version")),
            start_time=self._parse_datetime(row.get("start_time")),
            end_time=self._parse_datetime(row.get("end_time")),
            duration_ms=self._safe_float(row.get("duration_ms")),
            assignee=self._str_or_none(row.get("assignee")),
            candidate_groups=candidate_groups,
            assigned_executor=self._str_or_none(assigned_executor),
            executed_by=self._str_or_none(executed_by),
            potential_executor_users=potential_users or None,
            potential_executor_groups=potential_groups or None,
            removal_time=self._parse_datetime(row.get("removal_time_") or row.get("removal_time")),
            extra={k: v for k, v in row.items() if k not in self._event_known_fields()},
        )

    @classmethod
    def _resolve_event_ts(cls, row: Dict[str, Any]) -> Optional[datetime]:
        return cls._parse_datetime(row.get("end_time")) or cls._parse_datetime(row.get("start_time"))

    @staticmethod
    def _deduplicate_events(events: List[ProcessEventDTO]) -> List[ProcessEventDTO]:
        unique: Dict[str, ProcessEventDTO] = {}
        for event in events:
            key = "|".join(
                [
                    event.case_id,
                    event.act_inst_id or event.task_id or "",
                    event.activity_def_id,
                    event.start_time.isoformat() if event.start_time else "",
                    event.end_time.isoformat() if event.end_time else "",
                ]
            )
            unique[key] = event
        return sorted(
            unique.values(),
            key=lambda item: (
                item.case_id,
                item.start_time or datetime.min,
                item.end_time or datetime.min,
                item.activity_def_id,
            ),
        )

    @staticmethod
    def _event_known_fields() -> set[str]:
        return {
            "case_id",
            "activity_def_id",
            "activity_name",
            "activity_type",
            "proc_def_id",
            "proc_def_key",
            "proc_def_version",
            "task_id",
            "act_inst_id",
            "parent_act_inst_id",
            "sequence_counter",
            "execution_id",
            "parent_execution_id",
            "scope_depth",
            "depth",
            "is_concurrent",
            "is_scope",
            "is_event_scope",
            "rev",
            "execution_rev",
            "call_proc_inst_id",
            "called_element",
            "binding_type",
            "version_tag",
            "version_number",
            "resolved_child_proc_def_id",
            "child_process_key",
            "child_version",
            "start_time",
            "end_time",
            "duration_ms",
            "assignee",
            "candidate_groups",
            "assigned_executor",
            "assigned_user_id",
            "executed_by",
            "executed_by_user_id",
            "completed_by",
            "potential_executor_users",
            "potential_executor_groups",
            "candidate_user_ids",
            "candidate_group_ids",
            "candidate_user_id",
            "candidate_group_id",
            "removal_time",
            "removal_time_",
        }

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_int_or_none(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_bool(value: Any) -> Optional[bool]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None

    @staticmethod
    def _str_or_none(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if text.lower() in CamundaRuntimeAdapter._NULLISH_TEXT:
            return None
        return text or None

    @classmethod
    def _parse_string_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return [part for part in parts if part and part.lower() not in cls._NULLISH_TEXT]
        if isinstance(value, (list, tuple, set)):
            result: List[str] = []
            for item in value:
                normalized = cls._str_or_none(item)
                if normalized:
                    result.append(normalized)
            return result
        parsed = cls._str_or_none(value)
        return [parsed] if parsed else []

    @classmethod
    def _choose_non_empty(cls, *values: Any) -> Optional[str]:
        for value in values:
            text = cls._str_or_none(value)
            if text:
                return text
        return None
