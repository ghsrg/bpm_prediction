from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from tools.sync_stats import main as sync_stats_main


def _copy_mock_exports(target_dir: Path) -> None:
    source_dir = Path("data/camunda_exports")
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "mock_historic_activity_events.csv",
        "mock_historic_tasks.csv",
        "mock_identity_links.csv",
        "mock_execution_tree.csv",
        "mock_multi_instance_variables.csv",
        "mock_process_variables.csv",
        "mock_process_instance_links.csv",
    ):
        shutil.copy2(source_dir / file_name, target_dir / file_name)


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


def test_sync_stats_builds_tier_a_snapshot_and_indexes(tmp_path: Path):
    export_dir = tmp_path / "exports"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg_sync_stats.yaml"
    out_path = tmp_path / "sync_stats_summary.json"

    _copy_mock_exports(export_dir)

    repo = FileBasedKnowledgeGraphRepository(base_dir=kg_dir)
    repo.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("StartEvent_1", "Task_Approve"), ("Task_Approve", "EndEvent_1")],
            nodes=[
                {"id": "StartEvent_1", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
                {"id": "Task_Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {"id": "EndEvent_1", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
            ],
        ),
        process_name="procurement",
    )

    cfg_path.write_text(
        f"""
data:
  dataset_name: "procurement"
  dataset_label: "procurement"
  log_path: "__camunda__"

mapping:
  adapter: "camunda"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: true

  camunda_adapter:
    process_name: "procurement"
    runtime:
      runtime_source: "files"
      export_dir: "{export_dir.as_posix()}"
      history_cleanup_aware: true
      legacy_removal_time_policy: "treat_as_eternal"
      on_missing_removal_time: "auto_fallback"

sync_stats:
  enabled: true
  stats_time_policy: "strict_asof"
  process_scope_policy: "up_to_target_version"
  windows_days: [7, 30, 90]
  process_filters: ["procurement"]
  confidence_weights:
    sample_size: 0.4
    freshness: 0.3
    coverage: 0.3
  show_progress: false

experiment:
  mode: "train"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = sync_stats_main(["--config", str(cfg_path), "--out", str(out_path), "--as-of", "2026-03-12T00:00:00Z"])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["processed_versions"] == 1

    loaded = repo.get_process_structure("v1", process_name="procurement")
    assert loaded is not None
    assert loaded.node_stats is not None
    assert loaded.edge_stats is not None
    assert loaded.gnn_features is not None
    assert loaded.stats_diagnostics is not None
    assert loaded.metadata is not None
    assert "stats_index" in loaded.metadata

    snapshot_dir = kg_dir / "procurement" / "v1" / "snapshots"
    snapshots = list(snapshot_dir.glob("*.json"))
    assert snapshots


def test_sync_stats_supports_xes_lite_source(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg_sync_stats_xes.yaml"
    out_path = tmp_path / "sync_stats_xes_summary.json"

    _write_xes(xes_path, case_id="C1", a1="A", a2="B")

    repo = FileBasedKnowledgeGraphRepository(base_dir=kg_dir)
    repo.save_process_structure(
        "alpha",
        ProcessStructureDTO(
            version="alpha",
            allowed_edges=[("A", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {"id": "B", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            ],
        ),
        process_name="alpha",
    )

    cfg_path.write_text(
        f"""
data:
  dataset_name: "alpha"
  dataset_label: "alpha"
  log_path: "{xes_path.as_posix()}"

mapping:
  adapter: "xes"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: true
  xes_adapter:
    case_id_key: "concept:name"
    activity_key: "concept:name"
    timestamp_key: "time:timestamp"
    resource_key: "org:resource"
    lifecycle_key: "lifecycle:transition"
    version_key: "concept:version"
    pairing_strategy: "lifo"
    use_classifier: false

sync_stats:
  enabled: true
  stats_time_policy: "strict_asof"
  process_scope_policy: "up_to_target_version"
  windows_days: [7, 30, 90]
  process_filters: ["alpha"]
  show_progress: false

experiment:
  mode: "train"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = sync_stats_main(["--config", str(cfg_path), "--out", str(out_path), "--as-of", "2024-01-15T00:00:00Z"])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["adapter"] == "xes"
    assert summary["processed_versions"] == 1

    loaded = repo.get_process_structure("alpha", process_name="alpha")
    assert loaded is not None
    assert loaded.node_stats is not None
    assert loaded.edge_stats is not None
    assert loaded.gnn_features is not None


def test_sync_stats_xes_matches_process_by_file_stem_when_dataset_alias_differs(tmp_path: Path):
    xes_path = tmp_path / "BPI_Challenge_2012.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg_sync_stats_xes_alias.yaml"
    out_path = tmp_path / "sync_stats_xes_alias_summary.json"

    _write_xes(xes_path, case_id="C1", a1="A", a2="B")

    repo = FileBasedKnowledgeGraphRepository(base_dir=kg_dir)
    repo.save_process_structure(
        "BPI_Challenge_2012",
        ProcessStructureDTO(
            version="BPI_Challenge_2012",
            allowed_edges=[("A", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {"id": "B", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            ],
        ),
        process_name="BPI_Challenge_2012",
    )

    cfg_path.write_text(
        f"""
data:
  dataset_name: "bpi2012"
  dataset_label: "bpi2012"
  log_path: "{xes_path.as_posix()}"

mapping:
  adapter: "xes"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: true
  xes_adapter:
    case_id_key: "concept:name"
    activity_key: "concept:name"
    timestamp_key: "time:timestamp"
    resource_key: "org:resource"
    lifecycle_key: "lifecycle:transition"
    version_key: "concept:version"
    pairing_strategy: "lifo"
    use_classifier: false

sync_stats:
  enabled: true
  stats_time_policy: "strict_asof"
  process_scope_policy: "up_to_target_version"
  windows_days: [7, 30, 90]
  process_filters: ["BPI_Challenge_2012"]
  show_progress: false

experiment:
  mode: "train"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = sync_stats_main(["--config", str(cfg_path), "--out", str(out_path), "--as-of", "2024-01-15T00:00:00Z"])
    assert rc == 0

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["processed_versions"] == 1
    assert summary["details"]
    assert float(summary["details"][0]["history_coverage_percent"]) > 0.0

    loaded = repo.get_process_structure("BPI_Challenge_2012", process_name="BPI_Challenge_2012")
    assert loaded is not None and loaded.node_stats is not None
    all_time = loaded.node_stats.get("windows", {}).get("all_time", {})
    version_exec = all_time.get("version", {}).get("exec_count", {})
    assert float(version_exec.get("A", 0.0)) > 0.0
