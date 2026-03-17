from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_runtime_procdef_catalog_template_has_required_columns():
    sql = _read("src/adapters/ingestion/camunda/sql/runtime/procdef_catalog.sql")
    assert "as proc_def_id" in sql
    assert "as proc_def_key" in sql
    assert "as version" in sql
    assert "as deployment_id" in sql
    assert "as resource_name" in sql
    assert "bpms_camunda_mssql_tst.dbo.act_re_procdef" in sql


def test_runtime_bpmn_bytes_template_uses_bytes_column():
    sql = _read("src/adapters/ingestion/camunda/sql/runtime/bpmn_bytes.sql")
    assert "ba.bytes_" in sql
    assert "as bpmn_xml_content" in sql
    assert "bpms_camunda_mssql_tst.dbo.act_ge_bytearray" in sql
