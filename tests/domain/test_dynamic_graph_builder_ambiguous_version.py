from __future__ import annotations

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700600000 + idx),
        resource_id="R4",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R4", "amount": float(idx + 1)},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_dynamic_graph_builder_returns_none_mask_for_ambiguous_version_without_process_scope(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()

    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("Start", "Approve")],
            edge_statistics={("Start", "Approve"): {"count": 1.0}},
        ),
        process_name="process_a",
    )
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("Start", "Reject")],
            edge_statistics={("Start", "Reject"): {"count": 1.0}},
        ),
        process_name="process_b",
    )

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)
    assert contract["allowed_target_mask"] is None
