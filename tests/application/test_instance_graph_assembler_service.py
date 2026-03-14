from __future__ import annotations

from datetime import datetime, timedelta

from src.application.services.instance_graph_assembler_service import InstanceGraphAssemblerService
from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(case_id: str, activity: str, idx: int) -> ProcessEventDTO:
    base = datetime.fromisoformat("2026-03-10T08:00:00")
    start = base + timedelta(minutes=idx)
    end = start + timedelta(seconds=30)
    return ProcessEventDTO(
        case_id=case_id,
        activity_def_id=activity,
        activity_name=activity,
        activity_type="userTask",
        task_id=f"task_{case_id}_{idx}",
        act_inst_id=f"ai_{case_id}_{idx}",
        execution_id=f"ex_{case_id}_{idx}",
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

