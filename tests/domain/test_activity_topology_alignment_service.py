from __future__ import annotations

import pytest

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.services.activity_topology_alignment_service import (
    ActivityTopologyAlignmentService,
    AlignmentGateConfig,
)


def _event(activity: str, name: str | None = None) -> ProcessEventDTO:
    return ProcessEventDTO(
        case_id="C1",
        activity_def_id=activity,
        activity_name=name,
        proc_def_key="loan_proc",
        proc_def_version="v1",
    )


def _dto(
    nodes: list[dict],
    edges: list[tuple[str, str]] | None = None,
) -> ProcessStructureDTO:
    return ProcessStructureDTO(
        version="v1",
        allowed_edges=edges or [("TaskA", "TaskB")],
        nodes=nodes,
    )


def test_exact_id_profile_preserves_legacy_matching():
    dto = _dto(
        [
            {
                "id": "TaskA",
                "name": "Task A",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
            {
                "id": "TaskB",
                "name": "Task B",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
        ],
        [("TaskA", "TaskB")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("TaskA"), _event("TaskB")],
        dto=dto,
        config=AlignmentGateConfig(
            profile="legacy_exact",
            min_event_match_ratio=1.0,
            min_unique_activity_coverage=1.0,
            min_node_coverage=1.0,
        ),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.event_match_ratio == pytest.approx(1.0)
    assert summary.node_coverage == pytest.approx(1.0)
    assert summary.match_counts_by_strategy["exact_id"] == 2


def test_legacy_profile_does_not_use_normalized_fallbacks():
    dto = _dto(
        [
            {
                "id": "task_a",
                "name": "Task A",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
        ],
        [("task_a", "task_a")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Task-A")],
        dto=dto,
        config=AlignmentGateConfig(
            profile="legacy_exact",
            min_event_match_ratio=1.0,
            min_unique_activity_coverage=1.0,
            min_node_coverage=0.0,
        ),
        scope_used="version",
    )

    assert summary.is_aligned is False
    assert summary.match_counts_by_strategy["normalized_id"] == 0
    assert summary.alignment_reason == "below_min_event_match_ratio"


def test_safe_profile_matches_unique_node_name_when_id_differs():
    dto = _dto(
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
        [("Task_Approve", "Task_Archive")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Approve"), _event("Archive")],
        dto=dto,
        config=AlignmentGateConfig(
            profile="safe_normalized",
            min_event_match_ratio=1.0,
            min_unique_activity_coverage=1.0,
            min_node_coverage=1.0,
        ),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.event_match_ratio == pytest.approx(1.0)
    assert summary.match_counts_by_strategy["exact_name"] == 2


def test_safe_profile_normalizes_case_and_separators():
    dto = _dto(
        [
            {
                "id": "task_a",
                "name": "Task A",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
            {
                "id": "task_b",
                "name": "Task B",
                "bpmn_tag": "userTask",
                "type": "userTask",
                "activity_type": "userTask",
            },
        ],
        [("task_a", "task_b")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Task-A"), _event("TASK B")],
        dto=dto,
        config=AlignmentGateConfig(
            profile="safe_normalized",
            min_event_match_ratio=1.0,
            min_unique_activity_coverage=1.0,
            min_node_coverage=1.0,
        ),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert (
        summary.match_counts_by_strategy["normalized_id"]
        + summary.match_counts_by_strategy["normalized_name"]
    ) == 2


def test_node_coverage_ignores_structural_only_nodes():
    dto = _dto(
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
                "name": "Choice",
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
        [
            ("StartEvent", "TaskA"),
            ("TaskA", "GatewayChoice"),
            ("GatewayChoice", "TaskB"),
            ("TaskB", "EndEvent"),
        ],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("TaskA"), _event("TaskB")],
        dto=dto,
        config=AlignmentGateConfig(
            profile="safe_normalized",
            min_event_match_ratio=1.0,
            min_unique_activity_coverage=1.0,
            min_node_coverage=1.0,
        ),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.structure_node_count == 5
    assert summary.loggable_node_count == 2
    assert summary.ignored_structural_node_count == 3
    assert summary.node_coverage == pytest.approx(1.0)


def test_ambiguous_normalized_name_fails_closed():
    dto = _dto(
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
        [("TaskA1", "TaskA2")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Approve")],
        dto=dto,
        config=AlignmentGateConfig(
            profile="safe_normalized",
            min_event_match_ratio=1.0,
            min_unique_activity_coverage=1.0,
            min_node_coverage=0.0,
        ),
        scope_used="version",
    )

    assert summary.is_aligned is False
    assert "ambiguous_activity_mapping" in summary.alignment_failures
    assert summary.ambiguous_event_activities_top == ["Approve"]
