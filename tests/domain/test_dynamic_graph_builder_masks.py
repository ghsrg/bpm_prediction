from __future__ import annotations

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository
import torch


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700100000 + idx),
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R1", "amount": float(idx + 1)},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_dynamic_graph_builder_builds_allowed_mask_for_known_version(mock_feature_configs):
    train_traces = [
        _trace("c1", "v1", ["Start", "Approve", "End"]),
        _trace("c2", "v1", ["Start", "Approve", "Rework"]),
    ]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    knowledge = TopologyExtractorService(knowledge_port=repository, process_name="dataset_a")
    knowledge.extract_from_logs(train_traces, process_name="dataset_a")

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)

    mask = contract["allowed_target_mask"]
    assert mask is not None
    assert "structural_edge_index" in contract
    assert "structural_edge_weight" in contract
    assert contract["structural_edge_index"].dtype == torch.long
    assert contract["structural_edge_weight"].dtype == torch.float32
    assert int(contract["structural_edge_index"].shape[0]) == 2
    assert int(contract["structural_edge_weight"].shape[0]) == int(contract["structural_edge_index"].shape[1])
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert mask.dtype == torch.bool
    assert int(mask.shape[0]) == len(activity_vocab)
    assert bool(mask[activity_vocab["End"]]) is True
    assert bool(mask[activity_vocab["Rework"]]) is True
    assert bool(mask[activity_vocab["Start"]]) is False


def test_dynamic_graph_builder_fallbacks_to_none_mask_for_unknown_version(mock_feature_configs):
    train_traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    knowledge = TopologyExtractorService(knowledge_port=repository, process_name="dataset_a")
    knowledge.extract_from_logs(train_traces, process_name="dataset_a")

    prefix_unknown = PrefixSlice(
        case_id="eval_case",
        process_version="v_unknown",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix_unknown)
    assert contract["allowed_target_mask"] is None
