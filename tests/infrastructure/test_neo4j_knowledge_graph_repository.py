from __future__ import annotations

import json
from datetime import datetime, timezone
from types import MethodType

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.neo4j_knowledge_graph_repository import (
    Neo4jKnowledgeGraphRepository,
)


class _DummyDriver:
    def close(self) -> None:
        return None


def _build_repo() -> Neo4jKnowledgeGraphRepository:
    return Neo4jKnowledgeGraphRepository(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pass",
        database="neo4j",
        verify_connectivity=False,
        driver=_DummyDriver(),
    )


def test_save_uses_apoc_free_cypher_and_typed_bpmn_semantics():
    repo = _build_repo()
    writes = []

    def _capture_write(self, *, operation, query, params):
        writes.append((operation, query, params))

    repo._run_write = MethodType(_capture_write, repo)

    dto = ProcessStructureDTO(
        version="v22",
        allowed_edges=[("StartEvent_1", "Task_1"), ("Task_1", "EndEvent_1")],
        edge_statistics={
            ("StartEvent_1", "Task_1"): {"count": 10.0},
            ("Task_1", "EndEvent_1"): {"count": 8.0},
        },
        nodes=[
            {"id": "StartEvent_1", "name": "Start", "bpmn_tag": "startEvent", "activity_type": "startEvent"},
            {"id": "Task_1", "name": "Approve", "bpmn_tag": "userTask", "activity_type": "userTask"},
            {"id": "EndEvent_1", "name": "End", "bpmn_tag": "endEvent", "activity_type": "endEvent"},
        ],
        edges=[
            {"id": "flow_1", "source": "StartEvent_1", "target": "Task_1", "edge_type": "sequence_flow"},
            {"id": "flow_2", "source": "Task_1", "target": "EndEvent_1", "edge_type": "sequence_flow"},
        ],
    )
    repo.save_process_structure(version="v22", dto=dto, process_name="procurement")

    all_queries = "\n".join(query for _, query, _ in writes).lower()
    assert "apoc." not in all_queries
    assert any(operation == "upsert_process_version" for operation, _, _ in writes)
    assert any("set n:startevent" in query.lower() for operation, query, _ in writes if operation == "upsert_node")
    assert any("set n:task:usertask" in query.lower() for operation, query, _ in writes if operation == "upsert_node")
    assert any(
        "merge (src)-[r:sequence_flow" in query.lower()
        for operation, query, _ in writes
        if operation == "upsert_edge"
    )
    assert any(
        "$version_key in n.versions" in query.lower()
        for operation, query, _ in writes
        if operation == "upsert_node"
    )


def test_get_process_structure_reconstructs_dto_from_neo4j_rows():
    repo = _build_repo()

    def _fake_read(self, *, operation, query, params):
        del query, params
        if operation == "resolve_process_for_version":
            return [{"process_name": "procurement"}]
        if operation == "load_process_version":
            return [
                {
                    "proc_def_id": "PROC_DEF_22",
                    "proc_def_key": "procurement_flow",
                    "deployment_id": "DEP_99",
                    "call_bindings_json": '{"call_1":{"status":"unresolved"}}',
                    "graph_topology_json": '{"cycles_detected": false}',
                    "metadata_json": '{"source":"neo4j"}',
                }
            ]
        if operation == "load_nodes_for_version":
            return [
                {
                    "node_id": "Task_1",
                    "name": "Approve",
                    "bpmn_tag": "userTask",
                    "camunda_type": "userTask",
                    "logical_type": "Task",
                    "activity_type": "userTask",
                    "scope_level": 0,
                    "parent_scope_id": "",
                    "is_event_subprocess": False,
                    "is_multi_instance": False,
                    "is_sequential_mi": False,
                    "loop_cardinality_expr": None,
                    "attached_to": None,
                    "extensions_json": '{"candidateGroups":"legal"}',
                    "extra_json": '{"custom_flag": true}',
                }
            ]
        if operation == "load_edges_for_version":
            return [
                {
                    "source_id": "Task_1",
                    "target_id": "Task_2",
                    "edge_id": "flow_1",
                    "edge_type": "sequence_flow",
                    "is_default": False,
                    "condition_expr": "",
                    "condition_complexity": 0.0,
                    "scope_level": 0,
                    "is_interrupting": False,
                    "frequency_count": 4.0,
                    "stats_json": '{"count": 4}',
                    "extra_json": '{"source_system":"camunda"}',
                }
            ]
        return []

    repo._run_read = MethodType(_fake_read, repo)

    dto = repo.get_process_structure(version="v22")
    assert dto is not None
    assert dto.proc_def_id == "PROC_DEF_22"
    assert dto.version == "v22"
    assert dto.nodes is not None and dto.nodes[0]["id"] == "Task_1"
    assert dto.edges is not None and dto.edges[0]["source"] == "Task_1"
    assert dto.allowed_edges == [("Task_1", "Task_2")]
    assert dto.edge_statistics is not None
    assert dto.edge_statistics[("Task_1", "Task_2")]["count"] == 4.0
    assert dto.call_bindings is not None and "call_1" in dto.call_bindings


def test_get_process_structure_returns_none_for_ambiguous_version_lookup():
    repo = _build_repo()

    def _fake_read(self, *, operation, query, params):
        del query, params
        if operation == "resolve_process_for_version":
            return [{"process_name": "proc_a"}, {"process_name": "proc_b"}]
        return []

    repo._run_read = MethodType(_fake_read, repo)
    assert repo.get_process_structure(version="v_shared") is None


def test_get_graph_for_visualization_filters_edges_by_frequency():
    repo = _build_repo()
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("A", "B"), ("B", "C"), ("C", "D")],
        edge_statistics={
            ("A", "B"): {"count": 10.0},
            ("B", "C"): {"count": 3.0},
            ("C", "D"): {"count": 1.0},
        },
        nodes=[
            {"id": "A", "bpmn_tag": "startEvent", "activity_type": "startEvent"},
            {"id": "B", "bpmn_tag": "userTask", "activity_type": "userTask"},
            {"id": "C", "bpmn_tag": "userTask", "activity_type": "userTask"},
            {"id": "D", "bpmn_tag": "endEvent", "activity_type": "endEvent"},
        ],
    )
    repo.get_process_structure = MethodType(lambda self, version, process_name=None: dto, repo)
    graph = repo.get_graph_for_visualization("proc", "v1", min_edge_frequency=3)
    assert set(graph.edges()) == {("A", "B"), ("B", "C")}


def test_save_process_structure_snapshot_uses_composite_key_and_json_payload():
    repo = _build_repo()
    writes = []

    def _capture_write(self, *, operation, query, params):
        writes.append((operation, query, params))

    def _fake_read(self, *, operation, query, params):
        del query, params
        if operation == "resolve_next_stats_kv_seq":
            return [{"max_seq": 3}]
        return []

    repo._run_write = MethodType(_capture_write, repo)
    repo._run_read = MethodType(_fake_read, repo)

    dto = ProcessStructureDTO(
        version="v22",
        proc_def_id="PROC_DEF_22",
        allowed_edges=[("A", "B")],
        node_stats={"windows": {"last_30d": {"version": {"exec_count": {"A": 10}}}}},
        edge_stats={"windows": {"last_30d": {"version": {"transition_probability": {"A|||B": 0.7}}}}},
        gnn_features={"windows": {"last_30d": {"version": {"resource_handover_entropy": 0.2}}}},
        stats_diagnostics={"history_coverage_percent": 88.0},
        metadata={"source": "unit-test"},
    )
    kv = repo.save_process_structure_snapshot(
        version="v22",
        dto=dto,
        process_name="procurement@tenant_a",
        as_of_ts=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert kv == "k000004"
    stats_writes = [item for item in writes if item[0] == "upsert_stats_snapshot"]
    assert len(stats_writes) == 1
    _, query, params = stats_writes[0]
    assert "StatsSnapshot" in query
    assert params["tenant_id"] == "tenant_a"
    assert params["process_name"] == "procurement@tenant_a"
    assert params["version_key"] == "v22"
    assert params["proc_def_id"] == "PROC_DEF_22"
    assert params["knowledge_version"] == "k000004"
    assert json.loads(params["node_stats_json"])["windows"]["last_30d"]["version"]["exec_count"]["A"] == 10
    process_version_writes = [item for item in writes if item[0] == "upsert_process_version"]
    assert process_version_writes
    _, _, pv_params = process_version_writes[-1]
    pv_meta = json.loads(pv_params.get("metadata_json", "{}"))
    assert pv_meta.get("knowledge_version") == "k000004"
    stats_contract = pv_meta.get("stats_contract", {})
    identity = stats_contract.get("identity", {}) if isinstance(stats_contract, dict) else {}
    assert identity.get("knowledge_version") == "k000004"


def test_save_process_structure_synthesizes_nodes_from_allowed_edges_when_nodes_missing():
    repo = _build_repo()
    writes = []

    def _capture_write(self, *, operation, query, params):
        writes.append((operation, query, params))

    repo._run_write = MethodType(_capture_write, repo)

    dto = ProcessStructureDTO(
        version="BPI_Challenge_2012",
        allowed_edges=[("A_START", "B_TASK"), ("B_TASK", "C_END")],
        node_metadata={
            "A_START": {"activity_name": "Start", "activity_type": "startEvent"},
            "B_TASK": {"activity_name": "Approve", "activity_type": "userTask"},
            "C_END": {"activity_name": "End", "activity_type": "endEvent"},
        },
        # Important: nodes intentionally omitted to emulate XES topology payload.
        nodes=[],
        edges=[],
    )

    repo.save_process_structure(version="BPI_Challenge_2012", dto=dto, process_name="BPI_Challenge_2012")

    node_writes = [item for item in writes if item[0] == "upsert_node"]
    edge_writes = [item for item in writes if item[0] == "upsert_edge"]

    assert len(node_writes) == 3
    assert len(edge_writes) == 2
    assert any("set n:startevent" in query.lower() for _, query, _ in node_writes)
    assert any("set n:task:usertask" in query.lower() for _, query, _ in node_writes)
    assert any("set n:endevent" in query.lower() for _, query, _ in node_writes)


def test_get_process_structure_asof_overlays_stats_snapshot_json():
    repo = _build_repo()

    base_dto = ProcessStructureDTO(
        version="v22",
        proc_def_id="PROC_DEF_22",
        allowed_edges=[("A", "B")],
        metadata={"base": "yes"},
    )
    repo.get_process_structure = MethodType(
        lambda self, version, process_name=None: base_dto.model_copy(deep=True),
        repo,
    )

    def _fake_read(self, *, operation, query, params):
        del query, params
        if operation == "load_stats_snapshot_as_of":
            return [
                {
                    "knowledge_version": "k000010",
                    "as_of_ts": "2026-03-19T09:00:00Z",
                    "node_stats_json": '{"windows":{"last_30d":{"version":{"exec_count":{"A":3}}}}}',
                    "edge_stats_json": '{"windows":{"last_30d":{"version":{"transition_probability":{"A|||B":0.5}}}}}',
                    "gnn_features_json": '{"windows":{"last_30d":{"version":{"resource_handover_entropy":0.25}}}}',
                    "stats_diagnostics_json": '{"history_coverage_percent":77.7}',
                    "metadata_json": '{"snapshot_meta":"ok"}',
                }
            ]
        return []

    repo._run_read = MethodType(_fake_read, repo)

    loaded = repo.get_process_structure_as_of(
        version="v22",
        process_name="procurement@tenant_a",
        as_of_ts=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )
    assert loaded is not None
    assert loaded.node_stats is not None
    assert loaded.edge_stats is not None
    assert loaded.gnn_features is not None
    assert loaded.stats_diagnostics is not None
    assert loaded.metadata is not None
    assert loaded.metadata["knowledge_version"] == "k000010"
    assert loaded.metadata["snapshot_meta"] == "ok"


def test_get_process_structure_asof_falls_back_without_proc_def_filter_when_primary_empty():
    repo = _build_repo()

    base_dto = ProcessStructureDTO(
        version="v22",
        proc_def_id="PROC_DEF_22_DIFFERENT",
        allowed_edges=[("A", "B")],
        metadata={"base": "yes"},
    )
    repo.get_process_structure = MethodType(
        lambda self, version, process_name=None: base_dto.model_copy(deep=True),
        repo,
    )

    def _fake_read(self, *, operation, query, params):
        del query, params
        if operation == "load_stats_snapshot_as_of":
            return []
        if operation == "load_stats_snapshot_as_of_fallback_no_proc_def":
            return [
                {
                    "knowledge_version": "k000042",
                    "as_of_ts": "2026-03-19T09:00:00Z",
                    "node_stats_json": '{"windows":{"all_time":{"version":{"exec_count":{"A":7}}}}}',
                    "edge_stats_json": '{"windows":{"all_time":{"version":{"transition_probability":{"A|||B":0.8}}}}}',
                    "gnn_features_json": '{"windows":{"all_time":{"version":{"resource_handover_entropy":0.1}}}}',
                    "stats_diagnostics_json": '{"history_coverage_percent":100.0}',
                    "metadata_json": '{"snapshot_meta":"fallback"}',
                }
            ]
        return []

    repo._run_read = MethodType(_fake_read, repo)

    loaded = repo.get_process_structure_as_of(
        version="v22",
        process_name="procurement@tenant_a",
        as_of_ts=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )
    assert loaded is not None
    assert loaded.metadata is not None
    assert loaded.metadata["knowledge_version"] == "k000042"
    assert loaded.metadata["snapshot_meta"] == "fallback"


def test_get_process_structure_asof_marks_missing_snapshot_when_no_rows():
    repo = _build_repo()
    base_dto = ProcessStructureDTO(
        version="v22",
        proc_def_id="PROC_DEF_22",
        allowed_edges=[("A", "B")],
        metadata={"base": "yes"},
    )
    repo.get_process_structure = MethodType(
        lambda self, version, process_name=None: base_dto.model_copy(deep=True),
        repo,
    )

    def _fake_read(self, *, operation, query, params):
        del operation, query, params
        return []

    repo._run_read = MethodType(_fake_read, repo)

    loaded = repo.get_process_structure_as_of(
        version="v22",
        process_name="procurement@tenant_a",
        as_of_ts=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )
    assert loaded is not None
    assert loaded.metadata is not None
    assert loaded.metadata.get("asof_snapshot_found") is False
    assert loaded.metadata.get("asof_resolution") == "missing_snapshot_fallback_base"
