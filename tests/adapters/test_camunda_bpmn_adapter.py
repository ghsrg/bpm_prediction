from __future__ import annotations

from pathlib import Path

from src.adapters.ingestion.camunda_bpmn_adapter import CamundaBpmnAdapter


def test_bpmn_adapter_resolves_mssql_connection_from_profile(tmp_path: Path, monkeypatch):
    profile_path = tmp_path / "mssql_camunda.yaml"
    profile_path.write_text(
        """
mssql_camunda:
  profiles:
    local:
      driver: "ODBC Driver 18 for SQL Server"
      host: "localhost"
      database: "camunda"
      trust_server_certificate: true
      user_env: "BPM_MSSQL_USER"
      password_env: "BPM_MSSQL_PASSWORD"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BPM_MSSQL_USER", "sa")
    monkeypatch.setenv("BPM_MSSQL_PASSWORD", "secret")

    adapter = CamundaBpmnAdapter(
        {
            "structure": {
                "bpmn_source": "mssql",
                "connections_file": str(profile_path),
                "connection_profile": "local",
            }
        }
    )
    assert "SERVER=localhost" in adapter.mssql_connection_string
    assert "UID=sa" in adapter.mssql_connection_string
    assert "PWD=secret" in adapter.mssql_connection_string


def test_bpmn_adapter_accepts_env_connection_string_override(monkeypatch):
    monkeypatch.setenv("BPM_MSSQL_CONNECTION_STRING", "DRIVER={X};SERVER=demo;DATABASE=camunda;")
    adapter = CamundaBpmnAdapter(
        {
            "structure": {
                "bpmn_source": "mssql",
                "mssql": {
                    "connection_string_env": "BPM_MSSQL_CONNECTION_STRING",
                },
            }
        }
    )
    assert adapter.mssql_connection_string == "DRIVER={X};SERVER=demo;DATABASE=camunda;"


def test_bpmn_adapter_filters_catalog_by_process_and_tenant():
    adapter = CamundaBpmnAdapter(
        {
            "process_filters": ["proc_a"],
            "tenant_id": "tenant_a",
            "structure": {"bpmn_source": "files"},
        }
    )

    adapter._fetch_catalog_rows = lambda: [  # type: ignore[assignment]
        {"proc_def_id": "1", "proc_def_key": "proc_a", "version": "1", "tenant_id": "tenant_a"},
        {"proc_def_id": "2", "proc_def_key": "proc_a", "version": "2", "tenant_id": "tenant_b"},
        {"proc_def_id": "3", "proc_def_key": "proc_b", "version": "1", "tenant_id": "tenant_a"},
    ]

    rows = adapter.fetch_procdef_catalog()
    assert len(rows) == 1
    assert rows[0]["proc_def_id"] == "1"
    assert rows[0]["proc_def_key"] == "proc_a"
    assert rows[0]["tenant_id"] == "tenant_a"
