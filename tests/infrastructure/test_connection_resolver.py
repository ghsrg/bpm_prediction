from __future__ import annotations

from pathlib import Path

from src.infrastructure.config.connection_resolver import (
    resolve_mssql_connection_string,
    resolve_neo4j_connection_settings,
)


def test_resolve_mssql_connection_string_from_profile_and_env(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "mssql.yaml"
    cfg_path.write_text(
        """
mssql_camunda:
  profiles:
    local:
      driver: "ODBC Driver 18 for SQL Server"
      host: "localhost"
      port: 1433
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

    conn = resolve_mssql_connection_string(
        cfg={
            "mssql": {
                "connections_file": str(cfg_path),
                "connection_profile": "local",
            }
        }
    )
    assert "SERVER=localhost,1433" in conn
    assert "DATABASE=camunda" in conn
    assert "UID=sa" in conn
    assert "PWD=secret" in conn


def test_resolve_mssql_connection_string_env_override(monkeypatch):
    monkeypatch.setenv("BPM_MSSQL_CONNECTION_STRING", "DRIVER={X};SERVER=s;DATABASE=d;")
    conn = resolve_mssql_connection_string(
        cfg={
            "mssql": {
                "connection_string_env": "BPM_MSSQL_CONNECTION_STRING",
            }
        }
    )
    assert conn == "DRIVER={X};SERVER=s;DATABASE=d;"


def test_resolve_neo4j_connection_settings_from_profile_and_env(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "neo4j.yaml"
    cfg_path.write_text(
        """
neo4j:
  profiles:
    local:
      uri: "bolt://localhost:7687"
      database: "neo4j"
      user_env: "BPM_NEO4J_USER"
      password_env: "BPM_NEO4J_PASSWORD"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BPM_NEO4J_USER", "neo4j")
    monkeypatch.setenv("BPM_NEO4J_PASSWORD", "pass")

    settings = resolve_neo4j_connection_settings(
        cfg={
            "neo4j": {
                "connections_file": str(cfg_path),
                "connection_profile": "local",
            }
        }
    )
    assert settings["uri"] == "bolt://localhost:7687"
    assert settings["database"] == "neo4j"
    assert settings["user"] == "neo4j"
    assert settings["password"] == "pass"
