from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from tools.sync_stats import main as sync_stats_main


def _event(activity: str, ts: str, resource: str = "u1") -> str:
    return f"""
    <event>
      <string key="concept:name" value="{activity}" />
      <string key="lifecycle:transition" value="complete" />
      <date key="time:timestamp" value="{ts}" />
      <string key="org:resource" value="{resource}" />
    </event>"""


def _trace(case_id: str, version: str, activities: list[str]) -> str:
    events = [
        _event(activity, f"2024-01-01T00:{idx + 1:02d}:00+00:00", f"u{idx + 1}")
        for idx, activity in enumerate(activities)
    ]
    return f"""
  <trace>
    <string key="concept:name" value="{case_id}" />
    <string key="concept:version" value="{version}" />
{''.join(events)}
  </trace>"""


def _write_xes(path: Path, activities: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
{_trace("C1", "alpha", activities)}
</log>
""",
        encoding="utf-8",
    )


def _save_structure(
    kg_dir: Path,
    nodes: list[dict],
) -> FileBasedKnowledgeGraphRepository:
    repo = FileBasedKnowledgeGraphRepository(base_dir=kg_dir)
    node_ids = [str(node["id"]) for node in nodes]
    edges = list(zip(node_ids[:-1], node_ids[1:])) or [(node_ids[0], node_ids[0])]
    repo.save_process_structure(
        "alpha",
        ProcessStructureDTO(version="alpha", allowed_edges=edges, nodes=nodes),
        process_name="alpha",
    )
    return repo


def _write_config(
    path: Path,
    *,
    xes_path: Path,
    kg_dir: Path,
    profile: str,
    on_fail: str,
    thresholds: tuple[float, float, float],
) -> None:
    min_event, min_unique, min_node = thresholds
    path.write_text(
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
  windows_days: [7, 30]
  process_filters: ["alpha"]
  alignment_gate:
    enabled: true
    profile: "{profile}"
    min_event_match_ratio: {min_event}
    min_unique_activity_coverage: {min_unique}
    min_node_coverage: {min_node}
    on_fail: "{on_fail}"
    warn_on_fail: false
  show_progress: false

experiment:
  mode: "sync-stats"
  split_strategy: "none"
  train_ratio: 1.0
  fraction: 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _run_sync(cfg_path: Path, out_path: Path) -> dict:
    rc = sync_stats_main(
        ["--config", str(cfg_path), "--out", str(out_path), "--as-of", "2024-01-02T00:00:00Z"]
    )
    assert rc == 0
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_safe_normalized_matches_unique_node_names_in_real_sync_stats(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    _write_xes(xes_path, ["Approve", "Archive"])
    repo = _save_structure(
        kg_dir,
        [
            {
                "id": "Task_Approve",
                "name": "Approve",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
            {
                "id": "Task_Archive",
                "name": "Archive",
                "bpmn_tag": "serviceTask",
                "type": "serviceTask",
                "activity_type": "serviceTask",
            },
        ],
    )
    _write_config(
        cfg_path,
        xes_path=xes_path,
        kg_dir=kg_dir,
        profile="safe_normalized",
        on_fail="raise",
        thresholds=(1.0, 1.0, 1.0),
    )

    summary = _run_sync(cfg_path, out_path)

    assert summary["processed_versions"] == 1
    loaded = repo.get_process_structure("alpha", process_name="alpha")
    alignment = loaded.metadata["stats_contract"]["alignment"]
    assert alignment["is_aligned"] is True
    assert alignment["event_match_ratio"] == pytest.approx(1.0)
    assert alignment["node_coverage"] == pytest.approx(1.0)
    assert alignment["match_counts_by_strategy"]["exact_name"] == 2


def test_safe_normalized_ignores_structural_nodes_in_real_sync_stats(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    _write_xes(xes_path, ["TaskA", "TaskB"])
    repo = _save_structure(
        kg_dir,
        [
            {
                "id": "StartEvent",
                "name": "Start",
                "bpmn_tag": "startEvent",
                "type": "startEvent",
                "activity_type": "startEvent",
            },
            {
                "id": "TaskA",
                "name": "Task A",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
            {
                "id": "GatewayChoice",
                "name": "Gateway",
                "bpmn_tag": "exclusiveGateway",
                "type": "exclusiveGateway",
                "activity_type": "exclusiveGateway",
            },
            {
                "id": "TaskB",
                "name": "Task B",
                "bpmn_tag": "serviceTask",
                "type": "serviceTask",
                "activity_type": "serviceTask",
            },
            {
                "id": "EndEvent",
                "name": "End",
                "bpmn_tag": "endEvent",
                "type": "endEvent",
                "activity_type": "endEvent",
            },
        ],
    )
    _write_config(
        cfg_path,
        xes_path=xes_path,
        kg_dir=kg_dir,
        profile="safe_normalized",
        on_fail="raise",
        thresholds=(1.0, 1.0, 1.0),
    )

    summary = _run_sync(cfg_path, out_path)

    assert summary["processed_versions"] == 1
    loaded = repo.get_process_structure("alpha", process_name="alpha")
    alignment = loaded.metadata["stats_contract"]["alignment"]
    assert alignment["structure_node_count"] == 5
    assert alignment["loggable_node_count"] == 2
    assert alignment["ignored_structural_node_count"] == 3
    assert alignment["node_coverage"] == pytest.approx(1.0)


def test_safe_normalized_ambiguous_name_raises_in_real_sync_stats(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    _write_xes(xes_path, ["Approve"])
    _save_structure(
        kg_dir,
        [
            {
                "id": "TaskA1",
                "name": "Approve",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
            {
                "id": "TaskA2",
                "name": "approve",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
        ],
    )
    _write_config(
        cfg_path,
        xes_path=xes_path,
        kg_dir=kg_dir,
        profile="safe_normalized",
        on_fail="raise",
        thresholds=(1.0, 1.0, 0.0),
    )

    with pytest.raises(ValueError, match="ambiguous_activity_mapping"):
        sync_stats_main(
            [
                "--config",
                str(cfg_path),
                "--out",
                str(out_path),
                "--as-of",
                "2024-01-02T00:00:00Z",
            ]
        )
