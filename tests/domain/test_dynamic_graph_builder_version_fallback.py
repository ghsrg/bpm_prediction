from __future__ import annotations

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700400000 + idx),
        resource_id="R3",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R3", "amount": float(idx + 1)},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_dynamic_graph_builder_uses_dataset_name_version_key_when_trace_version_empty(mock_feature_configs):
    traces = [_trace("c1", "", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()
    service = TopologyExtractorService(knowledge_port=repository, process_name="dataset_alpha")
    service.extract_from_logs(traces, process_name="dataset_alpha")

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="dataset_alpha",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)
    assert contract["allowed_target_mask"] is not None


def test_dynamic_graph_builder_does_not_use_hardcoded_version_one_fallback(mock_feature_configs):
    traces = [_trace("c1", "dataset_alpha", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()
    service = TopologyExtractorService(knowledge_port=repository, process_name="dataset_alpha")
    service.extract_from_logs(traces, process_name="dataset_alpha")

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)
    assert contract["allowed_target_mask"] is None


def test_dynamic_graph_builder_accepts_numeric_or_v_prefixed_version_keys(mock_feature_configs):
    traces = [_trace("c1", "v22", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()
    service = TopologyExtractorService(knowledge_port=repository, process_name="dataset_alpha")
    service.extract_from_logs(traces, process_name="dataset_alpha")

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="22",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)
    assert contract["allowed_target_mask"] is not None
