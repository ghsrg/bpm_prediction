from __future__ import annotations

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.services.topology_projection_alignment import TopologyProjectionCompiler


def _node(node_id: str, bpmn_tag: str, *, name: str | None = None) -> dict:
    return {
        "id": node_id,
        "name": name or node_id,
        "bpmn_tag": bpmn_tag,
        "type": bpmn_tag,
        "activity_type": bpmn_tag,
    }


def test_projection_alignment_collapses_gateway_path_and_reports_alignment():
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("TaskA", "Gateway_XOR"), ("Gateway_XOR", "TaskB")],
        nodes=[
            _node("TaskA", "userTask"),
            _node("Gateway_XOR", "exclusiveGateway"),
            _node("TaskB", "serviceTask"),
        ],
    )

    result = TopologyProjectionCompiler(gateway_mode="collapse_for_prediction").project(
        dto=dto,
        activity_vocab={"TaskA": 0, "TaskB": 1},
    )

    assert result.projected_edge_paths == {
        ("TaskA", "TaskB"): [["TaskA", "Gateway_XOR", "TaskB"]],
    }
    assert result.prediction_nodes == {"TaskA", "TaskB"}
    assert result.transparent_nodes == {"Gateway_XOR"}
    assert result.diagnostics.is_aligned is True
    assert result.diagnostics.failure_reasons == []
    assert result.diagnostics.projected_edge_count == 1
    assert result.diagnostics.skipped_projected_edges == []


def test_projection_alignment_reports_missing_vocab_endpoint():
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("TaskA", "Gateway_XOR"), ("Gateway_XOR", "TaskB")],
        nodes=[
            _node("TaskA", "userTask"),
            _node("Gateway_XOR", "exclusiveGateway"),
            _node("TaskB", "serviceTask"),
        ],
    )

    result = TopologyProjectionCompiler(gateway_mode="collapse_for_prediction").project(
        dto=dto,
        activity_vocab={"TaskA": 0},
    )

    assert result.projected_edge_paths == {
        ("TaskA", "TaskB"): [["TaskA", "Gateway_XOR", "TaskB"]],
    }
    assert result.diagnostics.is_aligned is False
    assert result.diagnostics.missing_vocab_nodes == ["TaskB"]
    assert result.diagnostics.skipped_projected_edges == [
        {"src": "TaskA", "dst": "TaskB", "reason": "missing_dst_vocab"}
    ]
    assert "missing_vocab_nodes" in result.diagnostics.failure_reasons


def test_projection_alignment_reports_missing_node_metadata_for_collapse():
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("TaskA", "Gateway_XOR"), ("Gateway_XOR", "TaskB")],
    )

    result = TopologyProjectionCompiler(gateway_mode="collapse_for_prediction").project(
        dto=dto,
        activity_vocab={"TaskA": 0, "TaskB": 1},
    )

    assert result.projected_edge_paths == {
        ("Gateway_XOR", "TaskB"): [["Gateway_XOR", "TaskB"]],
        ("TaskA", "Gateway_XOR"): [["TaskA", "Gateway_XOR"]],
    }
    assert result.diagnostics.missing_node_metadata is True
    assert result.diagnostics.is_aligned is False
    assert "missing_node_metadata" in result.diagnostics.failure_reasons


def test_projection_alignment_preserves_multiple_transparent_paths():
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[
            ("TaskA", "Gateway_1"),
            ("TaskA", "Gateway_2"),
            ("Gateway_1", "TaskB"),
            ("Gateway_2", "TaskB"),
        ],
        nodes=[
            _node("TaskA", "userTask"),
            _node("Gateway_1", "exclusiveGateway"),
            _node("Gateway_2", "exclusiveGateway"),
            _node("TaskB", "serviceTask"),
        ],
    )

    result = TopologyProjectionCompiler(gateway_mode="collapse_for_prediction").project(
        dto=dto,
        activity_vocab={"TaskA": 0, "TaskB": 1},
    )

    assert result.projected_edge_paths == {
        ("TaskA", "TaskB"): [
            ["TaskA", "Gateway_1", "TaskB"],
            ["TaskA", "Gateway_2", "TaskB"],
        ],
    }
    assert result.diagnostics.projected_edge_count == 1
    assert result.diagnostics.source_path_count == 2
    assert result.diagnostics.is_aligned is True


def test_projection_alignment_reports_duplicate_activity_labels_as_identity_debt():
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("TaskA_1", "TaskA_2")],
        nodes=[
            _node("TaskA_1", "userTask", name="Review"),
            _node("TaskA_2", "userTask", name="Review"),
        ],
    )

    result = TopologyProjectionCompiler(gateway_mode="preserve").project(
        dto=dto,
        activity_vocab={"TaskA_1": 0, "TaskA_2": 1},
    )

    assert result.diagnostics.duplicate_activity_labels == {"Review": ["TaskA_1", "TaskA_2"]}
    assert result.diagnostics.is_aligned is True
