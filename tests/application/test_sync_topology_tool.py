from __future__ import annotations

import json
from pathlib import Path

from tools.sync_topology import main as sync_topology_main


def _write_bpmn(path: Path, process_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Examples">
  <process id="{process_id}" isExecutable="true">
    <startEvent id="start" />
    <userTask id="task_1" name="Task 1" />
    <endEvent id="end" />
    <sequenceFlow id="f1" sourceRef="start" targetRef="task_1" />
    <sequenceFlow id="f2" sourceRef="task_1" targetRef="end" />
  </process>
</definitions>
"""
    path.write_text(xml, encoding="utf-8")


def _write_xes(path: Path, case_id: str, a1: str, a2: str) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="{case_id}" />
    <event>
      <string key="concept:name" value="{a1}" />
      <date key="time:timestamp" value="2024-01-01T00:00:00+00:00" />
      <string key="org:resource" value="u1" />
    </event>
    <event>
      <string key="concept:name" value="{a2}" />
      <date key="time:timestamp" value="2024-01-01T01:00:00+00:00" />
      <string key="org:resource" value="u2" />
    </event>
  </trace>
</log>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sync_topology_camunda_files_syncs_all_definitions(tmp_path: Path):
    export_dir = tmp_path / "camunda_exports"
    bpmn_dir = export_dir / "bpmn_xml"
    kg_dir = tmp_path / "knowledge_graph"
    cfg_path = tmp_path / "cfg_camunda.yaml"
    out_path = tmp_path / "sync_summary.json"
    export_dir.mkdir(parents=True, exist_ok=True)

    (export_dir / "process_definitions.csv").write_text(
        "proc_def_id,proc_def_key,version,tenant_id,deployment_id,bpmn_path\n"
        "def_a,proc_a,1,tenant_a,dep_a,bpmn_xml/def_a.bpmn\n"
        "def_b,proc_b,1,tenant_b,dep_b,bpmn_xml/def_b.bpmn\n",
        encoding="utf-8",
    )
    _write_bpmn(bpmn_dir / "def_a.bpmn", process_id="proc_a")
    _write_bpmn(bpmn_dir / "def_b.bpmn", process_id="proc_b")

    cfg_path.write_text(
        f"""
data:
  dataset_name: "camunda_sync"
  dataset_label: "camunda_sync"
  log_path: "__camunda__"

mapping:
  adapter: "camunda"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: false
    ingest_split: "full"

  camunda_adapter:
    structure:
      source: "bpmn"
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
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = sync_topology_main(["--config", str(cfg_path), "--out", str(out_path)])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["adapter"] == "camunda"
    assert summary["total_procdefs"] == 2
    assert summary["parsed_procdefs"] == 2
    assert summary["missing_bpmn_count"] == 0

    artifact_a = kg_dir / "proc_a@tenant_a" / "v1" / "process_structure.json"
    artifact_b = kg_dir / "proc_b@tenant_b" / "v1" / "process_structure.json"
    assert artifact_a.exists()
    assert artifact_b.exists()


def test_sync_topology_xes_directory_syncs_all_xes_files(tmp_path: Path):
    xes_dir = tmp_path / "xes_dir"
    kg_dir = tmp_path / "knowledge_graph"
    cfg_path = tmp_path / "cfg_xes.yaml"
    out_path = tmp_path / "sync_xes_summary.json"

    _write_xes(xes_dir / "alpha.xes", case_id="C1", a1="A", a2="B")
    _write_xes(xes_dir / "beta.xes", case_id="C2", a1="X", a2="Y")

    cfg_path.write_text(
        f"""
data:
  dataset_name: "ignored_for_sync"
  dataset_label: "ignored_for_sync"
  log_path: "{xes_dir.as_posix()}"

mapping:
  adapter: "xes"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: false
    ingest_split: "full"

experiment:
  mode: "train"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = sync_topology_main(["--config", str(cfg_path), "--out", str(out_path)])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["adapter"] == "xes"
    assert summary["files_scanned"] == 2
    assert summary["datasets_synced"] == 2

    artifact_alpha = kg_dir / "alpha" / "alpha" / "process_structure.json"
    artifact_beta = kg_dir / "beta" / "beta" / "process_structure.json"
    assert artifact_alpha.exists()
    assert artifact_beta.exists()
