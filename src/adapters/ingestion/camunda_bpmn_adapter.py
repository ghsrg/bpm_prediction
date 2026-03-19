"""Camunda BPMN adapter for Stage 3.2 structure ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.domain.ports.camunda_bpmn_port import ICamundaBpmnPort
from src.infrastructure.config.connection_resolver import (
    resolve_mssql_connection_string,
)


logger = logging.getLogger(__name__)


class CamundaBpmnAdapter(ICamundaBpmnPort):
    """Read process-definition catalog and BPMN XML from files or MSSQL."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        structure_cfg = self.config.get("structure", {})
        self.structure_cfg = dict(structure_cfg) if isinstance(structure_cfg, dict) else {}

        raw_source = str(
            self.structure_cfg.get("bpmn_source", self.config.get("bpmn_source", "files"))
        ).strip().lower()
        self.bpmn_source = raw_source if raw_source in {"files", "mssql"} else "files"

        files_cfg_raw = self.structure_cfg.get("files", {})
        self.files_cfg = dict(files_cfg_raw) if isinstance(files_cfg_raw, dict) else {}
        self.files_export_dir = Path(
            str(
                self.files_cfg.get(
                    "export_dir",
                    self.config.get("export_dir", "data/camunda_exports/bpmn"),
                )
            )
        ).resolve()
        self.catalog_file = str(self.files_cfg.get("catalog_file", "process_definitions.csv")).strip()
        self.bpmn_dir_name = str(self.files_cfg.get("bpmn_dir", "bpmn_xml")).strip() or "bpmn_xml"

        self.sql_dir = Path(
            str(self.structure_cfg.get("sql_dir", "src/adapters/ingestion/camunda/sql/structure"))
        ).resolve()
        mssql_cfg = self.structure_cfg.get("mssql", {})
        if not isinstance(mssql_cfg, dict):
            mssql_cfg = {}
        runtime_cfg = self.config.get("runtime", {})
        if not isinstance(runtime_cfg, dict):
            runtime_cfg = {}
        runtime_mssql = runtime_cfg.get("mssql", {})
        if not isinstance(runtime_mssql, dict):
            runtime_mssql = {}
        merged_mssql = {**runtime_mssql, **mssql_cfg}
        self.mssql_connection_string = resolve_mssql_connection_string(
            cfg={
                "mssql": merged_mssql,
                "connections_file": self.structure_cfg.get("connections_file")
                or self.config.get("connections_file")
                or runtime_cfg.get("connections_file"),
                "connection_profile": self.structure_cfg.get("connection_profile")
                or self.config.get("connection_profile")
                or runtime_cfg.get("connection_profile"),
                "profile": self.structure_cfg.get("profile")
                or self.config.get("profile")
                or runtime_cfg.get("profile"),
            }
        )

        self._catalog_by_proc_def_id: Dict[str, Dict[str, Any]] = {}

    def fetch_procdef_catalog(
        self,
        process_name: Optional[str] = None,
        process_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._fetch_catalog_rows()
        normalized = [self._normalize_catalog_row(row) for row in rows]
        filtered = self._apply_process_filters(
            rows=normalized,
            process_name=process_name,
            process_filters=process_filters,
        )
        self._catalog_by_proc_def_id = {
            row["proc_def_id"]: row
            for row in filtered
            if row.get("proc_def_id")
        }
        return filtered

    def fetch_bpmn_xml(self, proc_def_id: str) -> Optional[str]:
        key = str(proc_def_id).strip()
        if not key:
            return None
        if self.bpmn_source == "mssql":
            return self._fetch_bpmn_xml_mssql(key)
        return self._fetch_bpmn_xml_files(key)

    def _fetch_catalog_rows(self) -> List[Dict[str, Any]]:
        if self.bpmn_source == "mssql":
            return self._fetch_catalog_rows_mssql()
        return self._fetch_catalog_rows_files()

    def _fetch_catalog_rows_files(self) -> List[Dict[str, Any]]:
        catalog_path = self.files_export_dir / self.catalog_file
        if not catalog_path.exists():
            logger.warning("Camunda BPMN files source: catalog file not found: %s", catalog_path)
            return []
        try:
            if catalog_path.suffix.lower() == ".csv":
                frame = pd.read_csv(catalog_path)
            else:
                frame = pd.read_excel(catalog_path)
        except Exception as exc:
            logger.warning("Failed to read Camunda BPMN catalog %s: %s", catalog_path, exc)
            return []
        if frame.empty:
            return []
        frame = frame.where(pd.notna(frame), None)
        return frame.to_dict(orient="records")

    def _fetch_catalog_rows_mssql(self) -> List[Dict[str, Any]]:
        sql_text = self._read_sql_text_by_name("procdef_catalog.sql")
        if sql_text is None:
            return []
        return self._run_mssql_query(sql_text, label="procdef_catalog")

    def _fetch_bpmn_xml_files(self, proc_def_id: str) -> Optional[str]:
        row = self._catalog_by_proc_def_id.get(proc_def_id, {})
        inline = row.get("bpmn_xml_content")
        if inline is not None and str(inline).strip():
            return str(inline)

        bpmn_dir = self.files_export_dir / self.bpmn_dir_name
        candidates: List[Path] = []
        raw_path = row.get("bpmn_path") or row.get("bpmn_file")
        if raw_path:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = (self.files_export_dir / path).resolve()
            candidates.append(path)

        resource_name = str(row.get("resource_name", "") or "").strip()
        if resource_name:
            candidates.append((bpmn_dir / resource_name).resolve())

        candidates.append((bpmn_dir / f"{proc_def_id}.bpmn").resolve())
        candidates.append((bpmn_dir / f"{proc_def_id}.xml").resolve())

        for candidate in candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                return candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    return candidate.read_text(encoding="utf-8-sig")
                except Exception as exc:  # pragma: no cover - rare encoding errors
                    logger.warning("Failed to decode BPMN file %s: %s", candidate, exc)
                    return None
            except Exception as exc:
                logger.warning("Failed to read BPMN file %s: %s", candidate, exc)
                return None
        return None

    def _fetch_bpmn_xml_mssql(self, proc_def_id: str) -> Optional[str]:
        sql_text = self._read_sql_text_by_name("bpmn_bytes.sql")
        if sql_text is None:
            return None
        safe_proc_def_id = proc_def_id.replace("'", "''")
        sql_text = sql_text.replace("{{PROC_DEF_ID}}", safe_proc_def_id)
        rows = self._run_mssql_query(sql_text, label=f"bpmn_bytes:{proc_def_id}")
        if not rows:
            return None
        row = rows[0]
        for key in ("bpmn_xml_content", "BPMN_XML_CONTENT", "BYTES_", "bytes_", "BYTEARRAY_", "bytearray_"):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, (bytes, bytearray)):
                try:
                    return bytes(value).decode("utf-8")
                except UnicodeDecodeError:
                    return bytes(value).decode("utf-8-sig", errors="ignore")
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _normalize_catalog_row(row: Dict[str, Any]) -> Dict[str, Any]:
        proc_def_id = CamundaBpmnAdapter._pick_text(
            row,
            "proc_def_id",
            "process_definition_id",
            "id_",
            "id",
        )
        proc_def_key = CamundaBpmnAdapter._pick_text(
            row,
            "proc_def_key",
            "process_key",
            "key_",
        )
        version = CamundaBpmnAdapter._pick_text(row, "version", "version_")
        version_tag = CamundaBpmnAdapter._pick_text(row, "version_tag", "version_tag_")
        deployment_id = CamundaBpmnAdapter._pick_text(row, "deployment_id", "deployment_id_")
        tenant_id = CamundaBpmnAdapter._pick_text(row, "tenant_id", "tenant_id_")
        resource_name = CamundaBpmnAdapter._pick_text(row, "resource_name", "resource_name_")
        bpmn_path = CamundaBpmnAdapter._pick_text(row, "bpmn_path", "bpmn_file", "bpmn_file_path")
        executable_raw = CamundaBpmnAdapter._pick_text(row, "is_executable", "isexecutable")
        is_executable = True
        if executable_raw:
            is_executable = executable_raw.strip().lower() not in {"false", "0", "no"}
        return {
            "proc_def_id": proc_def_id,
            "proc_def_key": proc_def_key,
            "version": version,
            "version_tag": version_tag,
            "deployment_id": deployment_id,
            "tenant_id": tenant_id,
            "resource_name": resource_name,
            "bpmn_path": bpmn_path,
            "bpmn_xml_content": row.get("bpmn_xml_content"),
            "is_executable": is_executable,
            "raw": row,
        }

    def _apply_process_filters(
        self,
        *,
        rows: List[Dict[str, Any]],
        process_name: Optional[str],
        process_filters: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        filters: List[str] = []
        if process_filters:
            filters = [str(item).strip() for item in process_filters if str(item).strip()]
        if not filters:
            from_cfg = self.config.get("process_filters")
            if isinstance(from_cfg, list):
                filters = [str(item).strip() for item in from_cfg if str(item).strip()]
        if not filters and process_name:
            name = str(process_name).strip()
            if name:
                filters = [name]

        tenant_filters: List[str] = []
        from_tenant_filters = self.config.get("tenant_filters")
        if isinstance(from_tenant_filters, list):
            tenant_filters = [
                str(item).strip()
                for item in from_tenant_filters
                if str(item).strip()
            ]
        single_tenant = str(self.config.get("tenant_id", "")).strip()
        if single_tenant and single_tenant not in tenant_filters:
            tenant_filters.append(single_tenant)

        lowered = {item.lower() for item in filters}
        lowered_tenant = {item.lower() for item in tenant_filters}
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            key = str(row.get("proc_def_key", "")).strip().lower()
            tenant = str(row.get("tenant_id", "")).strip().lower()

            if filters and (not key or key not in lowered):
                continue
            if lowered_tenant and (not tenant or tenant not in lowered_tenant):
                continue
            filtered.append(row)
        return filtered

    def _run_mssql_query(self, sql_text: str, *, label: str) -> List[Dict[str, Any]]:
        if not self.mssql_connection_string:
            logger.warning("MSSQL BPMN source selected but connection string is empty.")
            return []
        try:
            import pyodbc  # type: ignore
        except Exception:
            logger.warning("MSSQL BPMN source selected but 'pyodbc' is not installed.")
            return []

        connection = None
        try:
            connection = pyodbc.connect(self.mssql_connection_string)
            frame = pd.read_sql_query(sql_text, connection)
            if frame.empty:
                return []
            frame = frame.where(pd.notna(frame), None)
            return frame.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Failed to query MSSQL (%s): %s", label, exc)
            return []
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _read_sql_text(path: Path, *, warn_missing: bool = True) -> Optional[str]:
        if not path.exists():
            if warn_missing:
                logger.warning("Camunda BPMN SQL file was not found: %s", path)
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read SQL file %s: %s", path, exc)
            return None

    def _read_sql_text_by_name(self, file_name: str) -> Optional[str]:
        candidates = [
            self.sql_dir / file_name,
            Path("src/adapters/ingestion/camunda/sql/structure").resolve() / file_name,
            Path("src/adapters/ingestion/camunda/sql/runtime").resolve() / file_name,
        ]
        for candidate in candidates:
            sql_text = self._read_sql_text(candidate, warn_missing=False)
            if sql_text is not None:
                return sql_text
        logger.warning(
            "Camunda BPMN SQL file '%s' was not found in candidates: %s",
            file_name,
            ", ".join(str(path) for path in candidates),
        )
        return None

    @staticmethod
    def _pick_text(row: Dict[str, Any], *keys: str) -> str:
        lower_map = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = lower_map.get(str(key).lower())
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
                continue
            return text
        return ""
