from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter


def _copy_mock_exports(target_dir: Path) -> None:
    source_dir = Path("data/camunda_exports")
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "mock_historic_activity_events.csv",
        "mock_historic_tasks.csv",
        "mock_identity_links.csv",
        "mock_execution_tree.csv",
        "mock_multi_instance_variables.csv",
    ):
        shutil.copy2(source_dir / file_name, target_dir / file_name)


def test_runtime_adapter_reads_from_files_and_returns_diagnostics(tmp_path: Path):
    export_dir = tmp_path / "exports"
    _copy_mock_exports(export_dir)
    adapter = CamundaRuntimeAdapter(
        {
            "runtime_source": "files",
            "export_dir": str(export_dir),
            "history_cleanup_aware": True,
            "legacy_removal_time_policy": "treat_as_eternal",
        }
    )

    events, diagnostics = adapter.fetch_historic_activity_events(
        process_name="procurement",
        version_key="v1",
        since=datetime.fromisoformat("2026-03-10T00:00:00"),
        until=datetime.fromisoformat("2026-03-12T00:00:00"),
    )

    assert len(events) == 5
    assert diagnostics.rows_raw == 5
    assert diagnostics.rows_after_dedup == 5
    assert diagnostics.fallback_triggered is False
    assert "legacy_removal_time_treated_as_eternal" in diagnostics.warnings


def test_runtime_adapter_missing_removal_time_column_uses_auto_fallback(tmp_path: Path):
    export_dir = tmp_path / "exports"
    _copy_mock_exports(export_dir)
    activity_path = export_dir / "mock_historic_activity_events.csv"
    content = activity_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    header = lines[0].replace(",removal_time_", "")
    trimmed_rows = [",".join(row.split(",")[:-1]) for row in lines[1:]]
    activity_path.write_text("\n".join([header, *trimmed_rows]), encoding="utf-8")

    adapter = CamundaRuntimeAdapter(
        {
            "runtime_source": "files",
            "export_dir": str(export_dir),
            "history_cleanup_aware": True,
            "on_missing_removal_time": "auto_fallback",
        }
    )
    events, diagnostics = adapter.fetch_historic_activity_events(
        process_name="procurement",
        version_key="v1",
    )

    assert len(events) == 5
    assert "removal_time_missing_auto_fallback" in diagnostics.warnings
    assert diagnostics.rows_raw == diagnostics.rows_after_cleanup_filter


def test_runtime_adapter_deduplicates_events(tmp_path: Path):
    export_dir = tmp_path / "exports"
    _copy_mock_exports(export_dir)
    activity_path = export_dir / "mock_historic_activity_events.csv"
    content = activity_path.read_text(encoding="utf-8").splitlines()
    # Duplicate one event row intentionally.
    content.append(content[1])
    activity_path.write_text("\n".join(content), encoding="utf-8")

    adapter = CamundaRuntimeAdapter({"runtime_source": "files", "export_dir": str(export_dir)})
    events, diagnostics = adapter.fetch_historic_activity_events(
        process_name="procurement",
        version_key="v1",
    )

    assert diagnostics.rows_raw == 6
    assert diagnostics.rows_after_dedup == 5
    assert len(events) == 5


def test_runtime_adapter_mssql_mode_without_driver_returns_empty_not_raises():
    adapter = CamundaRuntimeAdapter(
        {
            "runtime_source": "mssql",
            "sql_dir": "src/adapters/ingestion/camunda/sql/runtime",
            "mssql": {"connection_string": "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;"},
        }
    )
    events, diagnostics = adapter.fetch_historic_activity_events(process_name="procurement", version_key="v1")
    tasks = adapter.fetch_historic_task_events(process_name="procurement", version_key="v1")

    assert events == []
    assert tasks == []
    assert diagnostics.rows_raw == 0

