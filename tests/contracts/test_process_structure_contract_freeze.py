from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.entities.process_structure import ProcessStructureDTO


def test_process_structure_normalizes_allowed_edges_and_call_bindings():
    dto = ProcessStructureDTO(
        version="v22",
        allowed_edges=[("B", "C"), ("A", "B"), ("A", "B")],
        call_bindings={
            "call_1": {
                "status": "unresolved",
                "binding_type": "latest",
            }
        },
    )
    assert dto.allowed_edges == [("A", "B"), ("B", "C")]
    assert dto.call_bindings is not None
    assert dto.call_bindings["call_1"]["requires_separate_inference"] is True
    assert dto.call_bindings["call_1"]["inference_fallback_strategy"] == "use_aggregated_stats"


def test_process_structure_rejects_node_without_required_id():
    with pytest.raises(ValidationError):
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("A", "B")],
            nodes=[{"bpmn_tag": "task", "type": "task"}],
        )


def test_process_structure_rejects_edge_without_source_target():
    with pytest.raises(ValidationError):
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("A", "B")],
            edges=[{"id": "e1", "edge_type": "sequence"}],
        )
