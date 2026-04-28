from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from tools.sync_stats import main as sync_stats_main
from tools.sync_topology import main as sync_topology_main


def _write_gateway_bpmn(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Tests">
  <process id="loan_proc" isExecutable="true">
    <startEvent id="StartEvent" name="Start" />
    <userTask id="TaskA" name="Task A" />
    <exclusiveGateway id="GatewayChoice" name="Choose branch" />
    <userTask id="TaskB" name="Task B" />
    <userTask id="TaskC" name="Task C" />
    <serviceTask id="TaskD" name="Task D" />
    <endEvent id="EndEvent" name="End" />
    <sequenceFlow id="f_start_a" sourceRef="StartEvent" targetRef="TaskA" />
    <sequenceFlow id="f_a_choice" sourceRef="TaskA" targetRef="GatewayChoice" />
    <sequenceFlow id="f_choice_b" sourceRef="GatewayChoice" targetRef="TaskB" />
    <sequenceFlow id="f_choice_c" sourceRef="GatewayChoice" targetRef="TaskC" />
    <sequenceFlow id="f_b_d" sourceRef="TaskB" targetRef="TaskD" />
    <sequenceFlow id="f_c_d" sourceRef="TaskC" targetRef="TaskD" />
    <sequenceFlow id="f_d_end" sourceRef="TaskD" targetRef="EndEvent" />
  </process>
</definitions>
""",
        encoding="utf-8",
    )


def _event(activity: str, lifecycle: str, ts: str, resource: str) -> str:
    return f"""
    <event>
      <string key="concept:name" value="{activity}" />
      <string key="lifecycle:transition" value="{lifecycle}" />
      <date key="time:timestamp" value="{ts}" />
      <string key="org:resource" value="{resource}" />
    </event>"""


def _trace(case_id: str, version: str, events: list[str]) -> str:
    return f"""
  <trace>
    <string key="concept:name" value="{case_id}" />
    <string key="concept:version" value="{version}" />
{''.join(events)}
  </trace>"""


def _write_versioned_xes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c1 = _trace(
        "C1",
        "v1",
        [
            _event("TaskA", "start", "2024-01-01T00:00:00+00:00", "u1"),
            _event("TaskA", "complete", "2024-01-01T00:10:00+00:00", "u1"),
            _event("TaskB", "start", "2024-01-01T00:10:00+00:00", "u2"),
            _event("TaskB", "complete", "2024-01-01T00:30:00+00:00", "u2"),
            _event("TaskD", "start", "2024-01-01T00:30:00+00:00", "svc"),
            _event("TaskD", "complete", "2024-01-01T00:45:00+00:00", "svc"),
        ],
    )
    c2 = _trace(
        "C2",
        "v1",
        [
            _event("TaskA", "start", "2024-01-02T00:00:00+00:00", "u1"),
            _event("TaskA", "complete", "2024-01-02T00:12:00+00:00", "u1"),
            _event("TaskC", "start", "2024-01-02T00:12:00+00:00", "u3"),
            _event("TaskC", "complete", "2024-01-02T00:32:00+00:00", "u3"),
            _event("TaskD", "start", "2024-01-02T00:32:00+00:00", "svc"),
            _event("TaskD", "complete", "2024-01-02T00:50:00+00:00", "svc"),
        ],
    )
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
{c1}
{c2}
</log>
""",
        encoding="utf-8",
    )


def _write_sync_topology_config(path: Path, export_dir: Path, kg_dir: Path) -> None:
    path.write_text(
        f"""
data:
  dataset_name: "loan_proc"
  dataset_label: "loan_proc"
  log_path: "__camunda__"

mapping:
  adapter: "camunda"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: false
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

experiment:
  mode: "sync-topology"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_sync_stats_config(path: Path, xes_path: Path, kg_dir: Path) -> None:
    path.write_text(
        f"""
data:
  dataset_name: "loan_proc"
  dataset_label: "loan_proc"
  log_path: "{xes_path.as_posix()}"

mapping:
  adapter: "xes"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: true
  xes_adapter:
    dataset_name: "loan_proc"
    case_id_key: "concept:name"
    activity_key: "concept:name"
    timestamp_key: "time:timestamp"
    resource_key: "org:resource"
    lifecycle_key: "lifecycle:transition"
    version_key: "concept:version"
    start_transitions: ["start"]
    complete_transitions: ["complete"]

sync_stats:
  enabled: true
  stats_time_policy: "strict_asof"
  process_scope_policy: "up_to_target_version"
  windows_days: [7, 30]
  process_filters: ["loan_proc"]
  show_progress: false
  alignment_gate:
    enabled: true
    profile: "safe_normalized"
    min_event_match_ratio: 1.0
    min_unique_activity_coverage: 1.0
    min_node_coverage: 0.75
    on_fail: "raise"

experiment:
  mode: "sync-stats"
  split_strategy: "none"
  train_ratio: 1.0
  fraction: 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _edge_payload(contract: dict, activity_vocab: dict[str, int]) -> dict[tuple[str, str], float]:
    reverse_vocab = {idx: activity for activity, idx in activity_vocab.items()}
    edge_index = contract["structural_edge_index"]
    edge_weight = contract["structural_edge_weight"]
    result: dict[tuple[str, str], float] = {}
    for col in range(int(edge_index.shape[1])):
        src = reverse_vocab[int(edge_index[0, col].item())]
        dst = reverse_vocab[int(edge_index[1, col].item())]
        result[(src, dst)] = float(edge_weight[col].item())
    return result


def test_bpmn_gateway_topology_and_xes_stats_feed_gnn_contract(tmp_path: Path, mock_feature_configs):
    export_dir = tmp_path / "camunda_exports"
    bpmn_dir = export_dir / "bpmn_xml"
    kg_dir = tmp_path / "kg"
    xes_path = tmp_path / "logs" / "loan_proc.xes"
    topology_cfg = tmp_path / "sync_topology.yaml"
    stats_cfg = tmp_path / "sync_stats.yaml"
    topology_out = tmp_path / "sync_topology_summary.json"
    stats_out = tmp_path / "sync_stats_summary.json"

    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "process_definitions.csv").write_text(
        "proc_def_id,proc_def_key,version,tenant_id,deployment_id,bpmn_path\n"
        "loan_proc_v1,loan_proc,1,,dep_1,bpmn_xml/loan_proc_v1.bpmn\n",
        encoding="utf-8",
    )
    _write_gateway_bpmn(bpmn_dir / "loan_proc_v1.bpmn")
    _write_versioned_xes(xes_path)
    _write_sync_topology_config(topology_cfg, export_dir=export_dir, kg_dir=kg_dir)
    _write_sync_stats_config(stats_cfg, xes_path=xes_path, kg_dir=kg_dir)

    assert sync_topology_main(["--config", str(topology_cfg), "--out", str(topology_out)]) == 0
    topology_summary = json.loads(topology_out.read_text(encoding="utf-8"))
    assert topology_summary["adapter"] == "camunda"
    assert topology_summary["parsed_procdefs"] == 1

    repo = FileBasedKnowledgeGraphRepository(base_dir=kg_dir)
    topology_dto = repo.get_process_structure("v1", process_name="loan_proc")
    assert topology_dto is not None
    assert ("TaskA", "GatewayChoice") in set(topology_dto.allowed_edges)
    assert ("GatewayChoice", "TaskB") in set(topology_dto.allowed_edges)
    assert ("GatewayChoice", "TaskC") in set(topology_dto.allowed_edges)
    assert ("TaskA", "TaskB") not in set(topology_dto.allowed_edges)

    assert sync_stats_main(["--config", str(stats_cfg), "--out", str(stats_out), "--as-of", "2024-01-03T00:00:00Z"]) == 0
    stats_summary = json.loads(stats_out.read_text(encoding="utf-8"))
    assert stats_summary["status"] == "ok"
    assert stats_summary["processed_versions"] == 1

    stats_dto = repo.get_process_structure("v1", process_name="loan_proc")
    assert stats_dto is not None
    alignment = stats_dto.metadata["stats_contract"]["alignment"]
    assert alignment["profile"] == "safe_normalized"
    assert alignment["is_aligned"] is True
    assert alignment["event_match_ratio"] == pytest.approx(1.0)
    assert alignment["unique_activity_coverage"] == pytest.approx(1.0)
    assert alignment["loggable_node_count"] == 4
    assert alignment["ignored_structural_node_count"] == 3
    stats_index = (stats_dto.metadata or {}).get("stats_index", {})
    assert stats_index["node"]["all_time.version.exec_count"]["TaskA"] == pytest.approx(2.0)
    assert stats_index["node"]["all_time.version.exec_count"]["TaskB"] == pytest.approx(1.0)
    assert stats_index["node"]["all_time.version.exec_count"]["TaskC"] == pytest.approx(1.0)
    assert stats_index["edge"]["all_time.version.transition_probability"]["TaskA|||TaskB"] == pytest.approx(0.5)
    assert stats_index["edge"]["all_time.version.transition_probability"]["TaskA|||TaskC"] == pytest.approx(0.5)

    mapping = {
        "xes_adapter": {
            "dataset_name": "loan_proc",
            "case_id_key": "concept:name",
            "activity_key": "concept:name",
            "timestamp_key": "time:timestamp",
            "resource_key": "org:resource",
            "lifecycle_key": "lifecycle:transition",
            "version_key": "concept:version",
            "start_transitions": ["start"],
            "complete_transitions": ["complete"],
        }
    }
    traces = list(XESAdapter().read(str(xes_path), mapping))
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[traces[0].events[0]],
        target_event=traces[0].events[1],
    )

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="loan_proc",
        graph_feature_mapping={
            "enabled": True,
            "topology_projection": {"gateway_mode": "collapse_for_prediction"},
            "node_numeric": [
                {
                    "name": "node_exec_count",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": 0.0,
                    "encoding": ["identity"],
                }
            ],
            "edge_weight": {
                "metric": "transition_probability",
                "window": "all_time",
                "scope": "version",
                "default": 0.0,
                "encoding": ["identity"],
            },
        },
    )
    contract = builder.build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert "GatewayChoice" not in activity_vocab

    edges = _edge_payload(contract, activity_vocab)
    assert edges[("TaskA", "TaskB")] == pytest.approx(0.5)
    assert edges[("TaskA", "TaskC")] == pytest.approx(0.5)
    assert ("TaskA", "GatewayChoice") not in edges

    struct_x = contract["struct_x"]
    assert float(struct_x[int(activity_vocab["TaskA"]), 0]) == pytest.approx(2.0)
    assert float(struct_x[int(activity_vocab["TaskB"]), 0]) == pytest.approx(1.0)
    assert float(struct_x[int(activity_vocab["TaskC"]), 0]) == pytest.approx(1.0)

    mask = contract["allowed_target_mask"]
    assert bool(mask[int(activity_vocab["TaskB"])]) is True
    assert bool(mask[int(activity_vocab["TaskC"])]) is True
