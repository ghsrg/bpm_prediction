from __future__ import annotations

import torch

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
        timestamp=float(1700200000 + idx),
        resource_id="R2",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R2", "amount": float(idx + 1)},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def _build_contract(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()
    knowledge = TopologyExtractorService(knowledge_port=repository, process_name="dataset_a")
    knowledge.extract_from_logs(traces, process_name="dataset_a")
    builder = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository)
    prefix = PrefixSlice(
        case_id="case_eval",
        process_version="v1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    return builder.build_graph(prefix)


def test_dynamic_graph_builder_contract_contains_only_tensors_and_scalars(mock_feature_configs):
    contract = _build_contract(mock_feature_configs)
    tensor_keys = {"x_cat", "x_num", "edge_index", "edge_type", "y", "batch", "allowed_target_mask"}

    for key, value in contract.items():
        if key in tensor_keys:
            if key == "allowed_target_mask" and value is None:
                continue
            assert isinstance(value, torch.Tensor), f"{key} must be torch.Tensor, got {type(value)}"
            continue
        if key == "num_nodes":
            assert isinstance(value, int), f"{key} must be int, got {type(value)}"
            continue
        assert not isinstance(value, (list, tuple, dict, str)), f"{key} violates Tensor Purity: {type(value)}"


def test_dynamic_graph_builder_mask_dtype_is_bool_when_present(mock_feature_configs):
    contract = _build_contract(mock_feature_configs)
    mask = contract["allowed_target_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
