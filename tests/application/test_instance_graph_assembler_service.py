from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter
from src.application.services.instance_graph_assembler_service import InstanceGraphAssemblerService
from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(
    case_id: str,
    activity: str,
    idx: int,
    *,
    activity_name: str | None = None,
    activity_type: str = "userTask",
    execution_id: str | None = None,
    parent_execution_id: str | None = None,
) -> ProcessEventDTO:
    base = datetime.fromisoformat("2026-03-10T08:00:00")
    start = base + timedelta(minutes=idx)
    end = start + timedelta(seconds=30)
    return ProcessEventDTO(
        case_id=case_id,
        activity_def_id=activity,
        activity_name=activity_name or activity,
        activity_type=activity_type,
        task_id=f"task_{case_id}_{idx}",
        act_inst_id=f"ai_{case_id}_{idx}",
        execution_id=execution_id or f"ex_{case_id}_{idx}",
        parent_execution_id=parent_execution_id,
        start_time=start,
        end_time=end,
        duration_ms=30000.0,
    )


def _diag(coverage_percent: float | None = 100.0) -> RuntimeFetchDiagnosticsDTO:
    return RuntimeFetchDiagnosticsDTO(
        rows_raw=10,
        rows_after_cleanup_filter=10,
        rows_after_dedup=10,
        cleaned_instances_percent=0.0,
        history_coverage_percent=coverage_percent,
    )


def test_assembler_fallback_structure_only_on_empty_events():
    knowledge_repo = InMemoryNetworkXRepository()
    knowledge_repo.save_process_structure(
        "v1",
        ProcessStructureDTO(version="v1", allowed_edges=[("A", "B"), ("B", "C")]),
        process_name="procurement",
    )
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=[],
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(),
        config={},
    )

    assert result.mode == "fallback_structure_only"
    assert result.graph["metadata"]["mode"] == "fallback_structure_only"
    assert len(result.graph["edges"]) == 2


def test_assembler_execution_centric_degrades_when_missing_execution_linkage():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    events = [
        _event("case_1", "A", 0),
        _event("case_1", "B", 1),
        _event("case_2", "A", 0),
        _event("case_2", "B", 1),
    ]
    execution_rows = [{"case_id": "case_1", "execution_id": "ex_case_1_0", "parent_execution_id": ""}]

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=execution_rows,
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={
            "canonical_mode": "execution-centric",
            "execution_missing_degrade_threshold": 0.4,
        },
    )

    assert result.mode == "canonical_activity_centric"
    assert "execution_centric_degraded_to_activity" in result.diagnostics.warnings


def test_assembler_uses_hysteresis_before_low_coverage_fallback():
    knowledge_repo = InMemoryNetworkXRepository()
    knowledge_repo.save_process_structure(
        "v1",
        ProcessStructureDTO(version="v1", allowed_edges=[("A", "B")]),
        process_name="procurement",
    )
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    events = [_event("case_1", "A", 0), _event("case_1", "B", 1)]

    first = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=1.0),
        config={"fallback_min_coverage_ratio": 0.05, "coverage_hysteresis_fetches": 2},
    )
    second = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=1.0),
        config={"fallback_min_coverage_ratio": 0.05, "coverage_hysteresis_fetches": 2},
    )

    assert first.mode.startswith("canonical_")
    assert second.mode == "fallback_structure_only"
    assert second.diagnostics.fallback_reason == "low_coverage_hysteresis"


def test_assembler_high_mi_guard_limits_case_payload():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    events = []
    for idx in range(200):
        activity = f"Task_{idx % 8}"
        events.append(_event("case_1", activity, idx))

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={
            "canonical_mode": "activity-centric",
            "max_execution_nodes_per_case": 10,
            "extreme_mi_strategy": "aggregate",
        },
    )

    assert result.mode == "canonical_activity_centric"
    assert result.graph["metadata"]["num_nodes"] <= 10


def test_assembler_attaches_identity_groups_null_safe():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    event = _event("case_1", "Task_Approve", 0)
    event.task_id = "task_1"

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=[event],
        execution_rows=[],
        variables_rows=[],
        identity_rows=[
            {"task_id": "task_1", "candidate_group_id": "managers"},
            {"task_id": "task_1", "candidate_group_id": "legal"},
            {"task_id": "", "candidate_group_id": "ignored"},
        ],
        diagnostics=_diag(coverage_percent=100.0),
        config={},
    )

    nodes = result.graph["nodes"]
    assert len(nodes) == 1
    assert result.diagnostics.meta["identity_rows"] == 3


def test_assembler_preserves_call_activity_link_and_process_variables():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    event = _event("case_1", "CallActivity_ChildFlow", 0)
    event.task_id = "task_1"
    event.call_proc_inst_id = "child_case_55"

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=[event],
        execution_rows=[],
        variables_rows=[],
        identity_rows=[
            {"task_id": "task_1", "candidate_group_id": "finance", "link_type": "candidate"},
            {"task_id": "task_1", "candidate_user_id": "analyst_1", "link_type": "candidate"},
            {"task_id": "task_1", "candidate_user_id": "owner_user", "link_type": "assignee"},
        ],
        diagnostics=_diag(coverage_percent=100.0),
        config={},
        task_rows=[
            {
                "task_id": "task_1",
                "assignee": "john_assignee",
                "executed_by": "john_executor",
            }
        ],
        process_variables_rows=[
            {"case_id": "case_1", "var_name": "purchase_amount", "double_value": 1200.5},
            {"case_id": "case_1", "execution_id": "ex_case_1_0", "var_name": "priority", "text_value": "HIGH"},
        ],
    )

    assert result.mode == "canonical_activity_centric"
    node = result.graph["nodes"][0]
    assert node["activity_name"] == "CallActivity_ChildFlow"
    assert node["assigned_executor"] == "owner_user"
    assert node["executed_by"] == "john_executor"
    assert set(node["potential_executor_groups"] or []) == {"finance"}
    assert set(node["potential_executor_users"] or []) == {"analyst_1"}
    assert node["process_variables"]["purchase_amount"] == 1200.5
    assert node["process_variables"]["priority"] == "HIGH"
    assert node["call_activity_link"]["child_case_id"] == "child_case_55"
    assert result.graph["metadata"]["num_call_activity_links"] == 1


def test_assembler_preserves_parallel_branches_without_forced_sequential_chain():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    events = [
        _event("case_1", "StartEvent_1", 0, activity_type="startEvent", execution_id="ex_root"),
        _event("case_1", "Gateway_Parallel", 1, activity_type="parallelGateway", execution_id="ex_root"),
        _event(
            "case_1",
            "Task_A",
            2,
            activity_name="Approve request",
            activity_type="userTask",
            execution_id="ex_branch_a",
            parent_execution_id="ex_root",
        ),
        _event(
            "case_1",
            "Task_B",
            3,
            activity_name="Notify supplier",
            activity_type="serviceTask",
            execution_id="ex_branch_b",
            parent_execution_id="ex_root",
        ),
    ]

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={"canonical_mode": "activity-centric"},
    )

    edges = result.graph["edges"]
    edge_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in edges}

    assert ("ai_case_1_1", "ai_case_1_2", "parallel_branch") in edge_keys
    assert ("ai_case_1_1", "ai_case_1_3", "parallel_branch") in edge_keys
    assert ("ai_case_1_2", "ai_case_1_3", "sequence") not in edge_keys
    assert ("ai_case_1_3", "ai_case_1_2", "sequence") not in edge_keys


def test_assembler_uses_execution_tree_when_parent_execution_missing_in_events():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    events = [
        _event("case_1", "Gateway_Parallel", 1, activity_type="parallelGateway", execution_id="ex_root"),
        _event("case_1", "Task_A", 2, activity_type="userTask", execution_id="ex_branch_a"),
        _event("case_1", "Task_B", 3, activity_type="serviceTask", execution_id="ex_branch_b"),
    ]
    execution_rows = [
        {"case_id": "case_1", "execution_id": "ex_branch_a", "parent_execution_id": "ex_root"},
        {"case_id": "case_1", "execution_id": "ex_branch_b", "parent_execution_id": "ex_root"},
    ]

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=execution_rows,
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={"canonical_mode": "activity-centric"},
    )

    edge_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in result.graph["edges"]}
    assert ("ai_case_1_1", "ai_case_1_2", "parallel_branch") in edge_keys
    assert ("ai_case_1_1", "ai_case_1_3", "parallel_branch") in edge_keys


def test_assembler_flattens_embedded_subprocess_as_container_with_internal_flow():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    base = datetime.fromisoformat("2026-03-10T08:00:00")
    case_id = "case_sp"
    sub_a_inst = "sp_a_inst"
    sub_b_inst = "sp_b_inst"

    events = [
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="StartEvent_1",
            activity_type="startEvent",
            act_inst_id="root_start",
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base,
            end_time=base,
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Gateway_split",
            activity_type="parallelGateway",
            act_inst_id="gw_split",
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base + timedelta(milliseconds=10),
            end_time=base + timedelta(milliseconds=10),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Activity_0z3oyip",
            activity_type="subProcess",
            act_inst_id=sub_a_inst,
            parent_act_inst_id=case_id,
            execution_id="ex_a",
            parent_execution_id="ex_root",
            start_time=base + timedelta(milliseconds=20),
            end_time=base + timedelta(seconds=5),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Activity_0ztae70",
            activity_type="subProcess",
            act_inst_id=sub_b_inst,
            parent_act_inst_id=case_id,
            execution_id="ex_b",
            parent_execution_id="ex_root",
            start_time=base + timedelta(milliseconds=30),
            end_time=base + timedelta(seconds=5, milliseconds=50),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Gateway_join",
            activity_type="parallelGateway",
            act_inst_id="gw_join",
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base + timedelta(seconds=6),
            end_time=base + timedelta(seconds=6),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="EndEvent_1",
            activity_type="noneEndEvent",
            act_inst_id="root_end",
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base + timedelta(seconds=7),
            end_time=base + timedelta(seconds=7),
        ),
        # Subprocess A internal nodes.
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Event_A_start",
            activity_type="startEvent",
            act_inst_id="sp_a_start",
            parent_act_inst_id=sub_a_inst,
            execution_id="ex_a",
            start_time=base + timedelta(milliseconds=25),
            end_time=base + timedelta(milliseconds=25),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Task_A",
            activity_type="serviceTask",
            act_inst_id="sp_a_task",
            parent_act_inst_id=sub_a_inst,
            execution_id="ex_a",
            start_time=base + timedelta(milliseconds=40),
            end_time=base + timedelta(seconds=4),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Event_A_end",
            activity_type="noneEndEvent",
            act_inst_id="sp_a_end",
            parent_act_inst_id=sub_a_inst,
            execution_id="ex_a",
            start_time=base + timedelta(seconds=4, milliseconds=50),
            end_time=base + timedelta(seconds=4, milliseconds=50),
        ),
        # Subprocess B internal nodes.
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Event_B_start",
            activity_type="startEvent",
            act_inst_id="sp_b_start",
            parent_act_inst_id=sub_b_inst,
            execution_id="ex_b",
            start_time=base + timedelta(milliseconds=35),
            end_time=base + timedelta(milliseconds=35),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Task_B",
            activity_type="serviceTask",
            act_inst_id="sp_b_task",
            parent_act_inst_id=sub_b_inst,
            execution_id="ex_b",
            start_time=base + timedelta(milliseconds=50),
            end_time=base + timedelta(seconds=4, milliseconds=500),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Event_B_end",
            activity_type="noneEndEvent",
            act_inst_id="sp_b_end",
            parent_act_inst_id=sub_b_inst,
            execution_id="ex_b",
            start_time=base + timedelta(seconds=4, milliseconds=550),
            end_time=base + timedelta(seconds=4, milliseconds=550),
        ),
    ]

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={"subprocess_graph_mode": "flattened"},
    )
    edge_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in result.graph["edges"]}
    node_ids = {str(node["id"]) for node in result.graph["nodes"]}

    assert ("gw_split", sub_a_inst, "parallel_branch") in edge_keys
    assert ("gw_split", sub_b_inst, "parallel_branch") in edge_keys
    assert (sub_a_inst, sub_b_inst, "sequence") not in edge_keys
    assert (sub_a_inst, "sp_a_task", "sequence") in edge_keys
    assert ("sp_a_task", sub_a_inst, "sequence") in edge_keys
    assert ("sp_a_start", "sp_a_task", "sequence") not in edge_keys
    assert "sp_a_start" not in node_ids
    assert "sp_a_end" not in node_ids
    assert "sp_b_start" not in node_ids
    assert "sp_b_end" not in node_ids


def test_assembler_hierarchical_mode_uses_subprocess_edge_types():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    case_id = "case_h"
    base = datetime.fromisoformat("2026-03-10T09:00:00")
    sub_inst = "sp_h_inst"
    events = [
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Start",
            activity_type="startEvent",
            act_inst_id="start_h",
            parent_act_inst_id=case_id,
            execution_id="ex_h",
            start_time=base,
            end_time=base,
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="SubP",
            activity_type="subProcess",
            act_inst_id=sub_inst,
            parent_act_inst_id=case_id,
            execution_id="ex_h",
            start_time=base + timedelta(milliseconds=10),
            end_time=base + timedelta(seconds=1),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Inner",
            activity_type="serviceTask",
            act_inst_id="inner_h",
            parent_act_inst_id=sub_inst,
            execution_id="ex_h",
            start_time=base + timedelta(milliseconds=20),
            end_time=base + timedelta(milliseconds=500),
        ),
    ]

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={"subprocess_graph_mode": "hierarchical"},
    )
    edge_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in result.graph["edges"]}
    assert (sub_inst, "inner_h", "subprocess_entry") in edge_keys
    assert ("inner_h", sub_inst, "subprocess_exit") in edge_keys
    assert result.graph["metadata"]["hierarchical_mode"] is True


def test_assembler_flattened_no_subprocess_node_removes_container_nodes():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    case_id = "case_flat_no_sp"
    base = datetime.fromisoformat("2026-03-10T09:20:00")
    sub_inst = "sp_no_node_inst"
    events = [
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Start",
            activity_type="startEvent",
            act_inst_id="start_node",
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base,
            end_time=base,
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="SubProcess_A",
            activity_type="subProcess",
            act_inst_id=sub_inst,
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base + timedelta(seconds=1),
            end_time=base + timedelta(seconds=5),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="Task_inside",
            activity_type="serviceTask",
            act_inst_id="inner_task",
            parent_act_inst_id=sub_inst,
            execution_id="ex_root",
            start_time=base + timedelta(seconds=2),
            end_time=base + timedelta(seconds=4),
        ),
        ProcessEventDTO(
            case_id=case_id,
            activity_def_id="After_SP",
            activity_type="userTask",
            act_inst_id="after_sp",
            parent_act_inst_id=case_id,
            execution_id="ex_root",
            start_time=base + timedelta(seconds=6),
            end_time=base + timedelta(seconds=7),
        ),
    ]

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={"subprocess_graph_mode": "flattened-no-subprocess-node"},
    )

    node_ids = {str(node["id"]) for node in result.graph["nodes"]}
    edge_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in result.graph["edges"]}
    assert sub_inst not in node_ids
    assert ("start_node", "inner_task", "sequence") in edge_keys
    assert ("inner_task", "after_sp", "sequence") in edge_keys
    assert result.graph["metadata"]["subprocess_nodes_removed"] is True


def test_assembler_call_activity_supernode_flags_unresolved_binding_without_forced_failure():
    knowledge_repo = InMemoryNetworkXRepository()
    service = InstanceGraphAssemblerService(knowledge_port=knowledge_repo)
    event = ProcessEventDTO(
        case_id="case_call",
        activity_def_id="CallActivity_Child",
        activity_type="callActivity",
        activity_name="Call Child Process",
        act_inst_id="call_ai_1",
        parent_act_inst_id="case_call",
        execution_id="ex_call",
        start_time=datetime.fromisoformat("2026-03-10T10:00:00"),
        end_time=datetime.fromisoformat("2026-03-10T10:00:05"),
        call_proc_inst_id="child_case_1",
        called_element="child_process_key",
        binding_type="versionTag",
        version_tag="v2",
    )

    result = service.build(
        process_name="procurement",
        version_key="v1",
        events=[event],
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=_diag(coverage_percent=100.0),
        config={"fallback_on_unresolved_call": False},
    )

    assert result.mode.startswith("canonical_")
    assert result.graph["metadata"]["unresolved_call_count"] == 1
    assert result.diagnostics.warnings.count("unresolved_call") == 1
    node = result.graph["nodes"][0]
    assert node["requires_separate_inference"] is True
    assert node["inference_entry_point"] == "child_process_root"
    call_link = result.graph["metadata"]["call_activity_links"][0]
    assert call_link["requires_separate_inference"] is True


@pytest.mark.mvp1_regression
def test_assembler_real_case_subprocess_edges_not_fan_out():
    export_path = Path("data/camunda_exports/historic_activity_events.csv")
    if not export_path.exists():
        pytest.skip("Camunda export fixture not found.")

    adapter = CamundaRuntimeAdapter({"runtime_source": "files", "export_dir": "data/camunda_exports"})
    events, diagnostics = adapter.fetch_historic_activity_events(
        process_name="B2BContracts_ApproveProject",
        version_key="v5",
    )
    case_id = "29537244-fdf3-11f0-bc36-26cbd5bba463"
    case_events = [event for event in events if event.case_id == case_id]
    if not case_events:
        pytest.skip("Target case id fixture is missing in export file.")

    service = InstanceGraphAssemblerService(knowledge_port=InMemoryNetworkXRepository())
    result = service.build(
        process_name="B2BContracts_ApproveProject",
        version_key="v5",
        events=case_events,
        execution_rows=[],
        variables_rows=[],
        identity_rows=[],
        diagnostics=diagnostics,
        config={"subprocess_graph_mode": "flattened", "fallback_on_unresolved_call": False},
    )

    node_by_activity = {}
    for node in result.graph["nodes"]:
        activity = node.get("activity_def_id")
        if activity and activity not in node_by_activity:
            node_by_activity[activity] = str(node.get("id"))

    split_id = node_by_activity["Gateway_1ag44zb"]
    sub_a_id = node_by_activity["Activity_0z3oyip"]
    sub_b_id = node_by_activity["Activity_0ztae70"]
    edge_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in result.graph["edges"]}
    nested_start_end_nodes = [
        node
        for node in result.graph["nodes"]
        if str(node.get("parent_subprocess_id") or "").strip()
        and (
            "start" in str(node.get("activity_type") or "").lower()
            or "end" in str(node.get("activity_type") or "").lower()
        )
    ]

    assert (split_id, sub_a_id, "parallel_branch") in edge_keys
    assert (split_id, sub_b_id, "parallel_branch") in edge_keys
    assert (sub_a_id, sub_b_id, "sequence") not in edge_keys
    assert (sub_b_id, sub_a_id, "sequence") not in edge_keys
    assert nested_start_end_nodes == []
