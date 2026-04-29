from __future__ import annotations

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository
import torch
import pytest


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


def test_dynamic_graph_builder_unions_active_and_struct_masks(mock_feature_configs):
    train_traces = [
        _trace("c1", "v1", ["Start", "Appraise_property", "Assess_eligibility"]),
        _trace("c2", "v1", ["Start", "Check_credit_history", "Assess_eligibility"]),
    ]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("Start", "Appraise_property"),
                ("Appraise_property", "Assess_eligibility"),
                ("Check_credit_history", "Assess_eligibility"),
            ],
        ),
    )

    prefix_events = [
        _event(0, "Start"),
        _event(1, "Appraise_property").model_copy(
            update={
                "extra": {
                    "concept:name": "Appraise_property",
                    "org:resource": "R1",
                    "amount": 2.0,
                    "active_activities_after_complete": ["Check_credit_history"],
                    "active_activity_counts_after_complete": {"Check_credit_history": 1},
                }
            }
        ),
    ]
    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=prefix_events,
        target_event=_event(2, "Assess_eligibility"),
    )
    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)
    mask = contract["allowed_target_mask"]
    assert isinstance(mask, torch.Tensor)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert bool(mask[activity_vocab["Assess_eligibility"]]) is True
    assert bool(mask[activity_vocab["Check_credit_history"]]) is True


def test_dynamic_graph_builder_collapses_gateway_path_for_prediction_mask(mock_feature_configs):
    train_traces = [_trace("c1", "v1", ["TaskA", "TaskB"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("TaskA", "Gateway_XOR",), ("Gateway_XOR", "TaskB")],
            nodes=[
                {"id": "TaskA", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {
                    "id": "Gateway_XOR",
                    "bpmn_tag": "exclusiveGateway",
                    "type": "exclusiveGateway",
                    "activity_type": "exclusiveGateway",
                },
                {"id": "TaskB", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "TaskA")],
        target_event=_event(1, "TaskB"),
    )
    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
    ).build_graph(prefix)

    mask = contract["allowed_target_mask"]
    assert isinstance(mask, torch.Tensor)
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert bool(mask[activity_vocab["TaskB"]]) is True


def test_dynamic_graph_builder_gateway_collapse_stops_at_next_prediction_node(mock_feature_configs):
    train_traces = [_trace("c1", "v1", ["TaskA", "TaskB", "TaskC"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("TaskA", "Gateway_1"),
                ("Gateway_1", "TaskB"),
                ("TaskB", "Gateway_2"),
                ("Gateway_2", "TaskC"),
            ],
            nodes=[
                {"id": "TaskA", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {"id": "Gateway_1", "bpmn_tag": "parallelGateway", "type": "parallelGateway", "activity_type": "parallelGateway"},
                {"id": "TaskB", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
                {"id": "Gateway_2", "bpmn_tag": "exclusiveGateway", "type": "exclusiveGateway", "activity_type": "exclusiveGateway"},
                {"id": "TaskC", "bpmn_tag": "scriptTask", "type": "scriptTask", "activity_type": "scriptTask"},
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "TaskA")],
        target_event=_event(1, "TaskB"),
    )
    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
    ).build_graph(prefix)

    mask = contract["allowed_target_mask"]
    assert isinstance(mask, torch.Tensor)
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert bool(mask[activity_vocab["TaskB"]]) is True
    assert bool(mask[activity_vocab["TaskC"]]) is False


def test_dynamic_graph_builder_strict_projection_alignment_raises_on_missing_vocab(mock_feature_configs):
    train_traces = [_trace("c1", "v1", ["TaskA"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("TaskA", "Gateway_XOR"), ("Gateway_XOR", "TaskB")],
            nodes=[
                {"id": "TaskA", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {
                    "id": "Gateway_XOR",
                    "bpmn_tag": "exclusiveGateway",
                    "type": "exclusiveGateway",
                    "activity_type": "exclusiveGateway",
                },
                {"id": "TaskB", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "TaskA")],
        target_event=_event(1, "TaskA"),
    )

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        graph_feature_mapping={
            "topology_projection": {
                "gateway_mode": "collapse_for_prediction",
                "on_fail": "raise",
            }
        },
    )

    with pytest.raises(ValueError, match="Topology projection alignment failed"):
        builder.build_graph(prefix)


def test_dynamic_graph_builder_attaches_projection_summary_scalars(mock_feature_configs):
    train_traces = [_trace("c1", "v1", ["TaskA", "TaskB"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("TaskA", "Gateway_XOR"), ("Gateway_XOR", "TaskB")],
            nodes=[
                {"id": "TaskA", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {
                    "id": "Gateway_XOR",
                    "bpmn_tag": "exclusiveGateway",
                    "type": "exclusiveGateway",
                    "activity_type": "exclusiveGateway",
                },
                {"id": "TaskB", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "TaskA")],
        target_event=_event(1, "TaskB"),
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
    ).build_graph(prefix)

    assert contract["topology_projection_aligned"] is True
    assert contract["topology_projection_projected_edge_count"] == 1
    assert contract["topology_projection_source_path_count"] == 1
    assert contract["topology_projection_skipped_edge_count"] == 0
    assert contract["topology_projection_missing_vocab_count"] == 0
    assert contract["topology_projection_duplicate_label_count"] == 0
    assert contract["topology_projection_missing_node_metadata"] is False


def test_dynamic_graph_builder_topology_cache_key_changes_when_node_metadata_changes(mock_feature_configs):
    train_traces = [_trace("c1", "v1", ["TaskA", "TaskB"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    repository = InMemoryNetworkXRepository()
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
    )
    gateway_dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("TaskA", "Gateway_XOR"), ("Gateway_XOR", "TaskB")],
        nodes=[
            {"id": "TaskA", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {
                "id": "Gateway_XOR",
                "bpmn_tag": "exclusiveGateway",
                "type": "exclusiveGateway",
                "activity_type": "exclusiveGateway",
            },
            {"id": "TaskB", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
        ],
    )
    task_dto = gateway_dto.model_copy(
        update={
            "nodes": [
                {"id": "TaskA", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {"id": "Gateway_XOR", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
                {"id": "TaskB", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
            ]
        }
    )

    assert builder._topology_cache_key(dto=gateway_dto, activity_vocab=activity_vocab, stats_allowed=True) != (
        builder._topology_cache_key(dto=task_dto, activity_vocab=activity_vocab, stats_allowed=True)
    )
