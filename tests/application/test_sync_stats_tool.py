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
