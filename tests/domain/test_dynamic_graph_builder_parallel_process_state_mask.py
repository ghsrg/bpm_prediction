from __future__ import annotations

import pytest

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(
    idx: int,
    activity: str,
    *,
    lifecycle: str = "complete",
    instance_id: str | None = None,
) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700100000 + idx),
        resource_id="R1",
        lifecycle=lifecycle,
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={
            "concept:name": activity,
            "org:resource": "R1",
            "lifecycle:transition": lifecycle,
            "sim:activity_instance_id": instance_id or f"ai_{idx}_{activity}",
        },
        activity_instance_id=instance_id or f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, events: list[EventRecord]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=events,
        trace_attributes={},
    )


def test_process_state_mask_keeps_parallel_active_candidate_in_allowed_mask(mock_feature_configs):
    train_events = [
        _event(0, "A", instance_id="A_1"),
        _event(1, "B", lifecycle="start", instance_id="B_1"),
        _event(2, "C", lifecycle="start", instance_id="C_1"),
        _event(3, "B", lifecycle="complete", instance_id="B_1"),
        _event(4, "C", lifecycle="complete", instance_id="C_1"),
        _event(5, "D", instance_id="D_1"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", train_events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=train_events[:4],
        target_event=train_events[4],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    c_idx = int(activity_vocab["C"])

    assert bool(contract["allowed_target_mask"][c_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][c_idx]) is True
    assert float(contract["struct_prefix_state_x"][c_idx, 5]) == pytest.approx(1.0)


def test_relaxed_reachability_allows_not_yet_started_parallel_sibling(mock_feature_configs):
    events = [
        _event(0, "A", instance_id="A_1"),
        _event(1, "B", instance_id="B_1"),
        _event(2, "C", instance_id="C_1"),
        _event(3, "D", instance_id="D_1"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:2],
        target_event=events[2],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_lookback_events=2,
        process_state_mask_relaxed_max_depth=1,
        process_state_mask_relaxed_max_cardinality_ratio=0.75,
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    c_idx = int(activity_vocab["C"])
    d_idx = int(activity_vocab["D"])

    assert bool(contract["allowed_target_mask"][c_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][c_idx]) is True
    assert bool(contract["allowed_target_mask"][d_idx]) is True
    assert int(contract["process_state_mask_relaxed_candidate_count"].item()) >= 1


def test_relaxed_reachability_suppresses_completed_parallel_siblings(mock_feature_configs):
    events = [
        _event(0, "A", instance_id="A_1"),
        _event(1, "B", instance_id="B_1"),
        _event(2, "C", instance_id="C_1"),
        _event(3, "D", instance_id="D_1"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:3],
        target_event=events[3],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_lookback_events=3,
        process_state_mask_relaxed_max_depth=1,
        process_state_mask_relaxed_max_cardinality_ratio=1.0,
        process_state_mask_relaxed_suppress_completed=True,
        process_state_mask_relaxed_anchor_policy="recent_prefix",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    b_idx = int(activity_vocab["B"])
    c_idx = int(activity_vocab["C"])
    d_idx = int(activity_vocab["D"])

    assert bool(contract["allowed_target_mask"][d_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][d_idx]) is True
    assert bool(contract["allowed_target_mask"][b_idx]) is False
    assert bool(contract["candidate_allowed_target_mask"][b_idx]) is False
    assert bool(contract["allowed_target_mask"][c_idx]) is False
    assert bool(contract["candidate_allowed_target_mask"][c_idx]) is False
    assert int(contract["process_state_mask_relaxed_raw_candidate_count"].item()) >= 2
    assert int(contract["process_state_mask_relaxed_suppressed_completed_count"].item()) >= 1


def test_relaxed_reachability_keeps_completed_direct_successor_for_loop(mock_feature_configs):
    events = [
        _event(0, "A", instance_id="A_1"),
        _event(1, "B", instance_id="B_1"),
        _event(2, "A", instance_id="A_2"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("B", "A"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:2],
        target_event=events[2],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_suppress_completed=True,
        process_state_mask_relaxed_anchor_policy="open_successors",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    a_idx = int(activity_vocab["A"])

    assert bool(contract["allowed_target_mask"][a_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][a_idx]) is True


def test_relaxed_reachability_keeps_completed_token_when_new_instance_is_active(mock_feature_configs):
    events = [
        _event(0, "A", lifecycle="complete", instance_id="A_1"),
        _event(1, "B", lifecycle="start", instance_id="B_1"),
        _event(2, "B", lifecycle="complete", instance_id="B_1"),
        _event(3, "B", lifecycle="start", instance_id="B_2"),
        _event(4, "C", lifecycle="complete", instance_id="C_1"),
        _event(5, "B", lifecycle="complete", instance_id="B_2"),
    ]
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", events)],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:5],
        target_event=events[5],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        process_state_mask_relaxed_lookback_events=5,
        process_state_mask_relaxed_max_depth=1,
        process_state_mask_relaxed_max_cardinality_ratio=1.0,
        process_state_mask_relaxed_suppress_completed=True,
        process_state_mask_relaxed_anchor_policy="open_successors",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    b_idx = int(activity_vocab["B"])

    assert bool(contract["allowed_target_mask"][b_idx]) is True
    assert bool(contract["candidate_allowed_target_mask"][b_idx]) is True
    assert int(contract["process_state_mask_target_suppressed_by_completed_filter_count"].item()) == 0
