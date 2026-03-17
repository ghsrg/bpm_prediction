from __future__ import annotations

import json
from pathlib import Path

from tools.ingest_topology import main as ingest_topology_main


def _write_catalog(path: Path) -> None:
    path.write_text(
        "proc_def_id,proc_def_key,version,deployment_id,bpmn_path\n"
        "parent_def,parent_proc,22,dep_parent,bpmn_xml/parent_def.bpmn\n"
        "child_def,child_proc,5,dep_child,bpmn_xml/child_def.bpmn\n",
        encoding="utf-8",
    )


def _write_bpmn(path: Path, xml: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


def _parent_bpmn() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
             xmlns:camunda="http://camunda.org/schema/1.0/bpmn"
             targetNamespace="Examples">
  <process id="parent_proc" isExecutable="true">
    <startEvent id="start" />
    <callActivity id="call1"
                  name="Call Child"
                  calledElement="child_proc"
                  camunda:calledElementBinding="latest" />
    <endEvent id="end" />
    <sequenceFlow id="f1" sourceRef="start" targetRef="call1" />
    <sequenceFlow id="f2" sourceRef="call1" targetRef="end" />
  </process>
</definitions>
"""


def _child_bpmn() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Examples">
  <process id="child_proc" isExecutable="true">
    <startEvent id="child_start" />
    <endEvent id="child_end" />
    <sequenceFlow id="f1" sourceRef="child_start" targetRef="child_end" />
  </process>
</definitions>
"""


def _invalid_bpmn() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <process id="parent_proc" isExecutable="true">
    <startEvent id="start" />
    <endEvent id="end" />
    <sequenceFlow id="f1" sourceRef="start" targetRef="end" />
  </process>
</definitions-broken>
"""


def _write_config(path: Path, export_dir: Path, kg_dir: Path) -> None:
    content = f"""
data:
  dataset_name: "camunda_dataset"
  dataset_label: "camunda_dataset"
  log_path: "__camunda__"

mapping:
  adapter: "camunda"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: false
    ingest_split: "train"

  camunda_adapter:
    process_name: "parent_proc"
    process_filters: ["parent_proc", "child_proc"]
    structure:
      source: "bpmn"
      structure_from_logs: false
      bpmn_source: "files"
      parser_mode: "recover"
      subprocess_mode: "flattened-no-subprocess-node"
      files:
        export_dir: "{export_dir.as_posix()}"
        catalog_file: "process_definitions.csv"
        bpmn_dir: "bpmn_xml"
      call_activity:
        inference_fallback_strategy: "use_aggregated_stats"

experiment:
  mode: "train"
  fraction: 1.0
  split_strategy: "temporal"
  train_ratio: 1.0
  split_ratio: [1.0, 0.0, 0.0]
"""
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_ingest_topology_camunda_bpmn_files_mode(tmp_path: Path):
    export_dir = tmp_path / "camunda_bpmn"
    bpmn_dir = export_dir / "bpmn_xml"
    kg_dir = tmp_path / "knowledge_graph"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    export_dir.mkdir(parents=True, exist_ok=True)
    _write_catalog(export_dir / "process_definitions.csv")
    _write_bpmn(bpmn_dir / "parent_def.bpmn", _parent_bpmn())
    _write_bpmn(bpmn_dir / "child_def.bpmn", _child_bpmn())
    _write_config(cfg_path, export_dir, kg_dir)

    rc = ingest_topology_main(["--config", str(cfg_path), "--out", str(out_path)])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["adapter"] == "camunda"
    assert summary["structure_source"] == "bpmn"
    assert summary["parsed_procdefs"] == 2
    assert summary["quarantine_error_codes"] == []
    assert "v22" in summary["versions_saved"]

    artifact_path = kg_dir / "camunda_dataset" / "v22" / "process_structure.json"
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    dto = artifact["dto"]
    assert dto["version"] == "v22"
    assert "call1" in dto["call_bindings"]
    assert dto["call_bindings"]["call1"]["inference_fallback_strategy"] == "use_aggregated_stats"


def test_ingest_topology_camunda_bpmn_quarantine_summary(tmp_path: Path):
    export_dir = tmp_path / "camunda_bpmn"
    bpmn_dir = export_dir / "bpmn_xml"
    kg_dir = tmp_path / "knowledge_graph"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    export_dir.mkdir(parents=True, exist_ok=True)
    _write_catalog(export_dir / "process_definitions.csv")
    _write_bpmn(bpmn_dir / "parent_def.bpmn", _invalid_bpmn())
    _write_bpmn(bpmn_dir / "child_def.bpmn", _child_bpmn())
    _write_config(cfg_path, export_dir, kg_dir)

    rc = ingest_topology_main(["--config", str(cfg_path), "--out", str(out_path)])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["quarantined_procdefs"] == 1
    assert "xml_parse_error" in summary["quarantine_error_codes"]
    assert len(summary["quarantine_records"]) == 1
    record = summary["quarantine_records"][0]
    assert record["error_code"] == "xml_parse_error"
    assert "source_hint" in record
