from __future__ import annotations

import json
import pytest
import torch

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _audit_policy(**overrides):
    policy = {
        "source": "lifecycle_active_set",
        "include_direct_successors": True,
        "include_active_candidates": True,
        "relaxed_lookback_events": 8,
        "relaxed_max_depth": 1,
        "relaxed_max_cardinality_ratio": 0.35,
        "relaxed_suppress_completed": True,
        "relaxed_anchor_policy": "open_successors",
        "relaxed_loop_policy": "keep_direct_successor_repeats",
    }
    policy.update(overrides)
    return policy


def _audit_payload(builder, prefix):
    contract = builder.build_graph(prefix)
    return json.loads(contract["audit_payload_json"]), contract


def test_common_audit_policy_preserves_decoding_and_unseen_labels(mock_feature_configs):
    events = [_event(0, "A"), _event(1, "B"), _event(2, "unseen")]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs,
                             traces=[_trace("train", "v1", events[:2])])
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure("v1", ProcessStructureDTO(
        version="v1", allowed_edges=[("A", "B"), ("A", "unseen")]))
    prefix = PrefixSlice(case_id="eval", process_version="v1",
                         prefix_events=events[:1], target_event=events[2])
    policy = dict(source="lifecycle_active_set", include_direct_successors=True,
                  include_active_candidates=True, relaxed_lookback_events=8,
                  relaxed_max_depth=1, relaxed_max_cardinality_ratio=0.35,
                  relaxed_suppress_completed=True, relaxed_anchor_policy="open_successors",
                  relaxed_loop_policy="keep_direct_successor_repeats")
    plain = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)
    audited = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository,
                                  rs01_admissibility_policy=policy).build_graph(prefix)
    assert set(audited["audit_allowed_activity_labels"]) == {"B", "unseen"}
    assert audited["audit_mask_status"] == "resolved"
    for key, value in plain.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, audited[key]), key


def test_common_mask_is_invariant_across_eopkg_gatv2_mask_lstm_target_and_future(mock_feature_configs):
    shared_prefix = [_event(0, "A")]
    trace_a = _trace("eval-a", "v1", [*shared_prefix, _event(1, "B"), _event(2, "C")])
    trace_b = _trace("eval-b", "v1", [*shared_prefix, _event(1, "B"), _event(2, "Z")])
    prefix_a = PrefixPolicy().generate_slices(trace_a)[0]
    prefix_b = PrefixPolicy().generate_slices(trace_b)[0]
    target_changed = PrefixSlice(
        case_id="eval-target",
        process_version="v1",
        prefix_events=shared_prefix,
        target_event=_event(1, "unseen_target"),
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(version="v1", allowed_edges=[("A", "B"), ("A", "C")]),
    )
    encoder_small = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train-small", "v1", [_event(0, "A"), _event(1, "B")])],
    )
    encoder_other = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train-other", "v1", [_event(0, "A"), _event(1, "Z")])],
    )
    encoder_lstm = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train-lstm", "v1", [_event(0, "A"), _event(1, "C"), _event(2, "Y")])],
    )
    builders = [
        DynamicGraphBuilder(
            feature_encoder=encoder,
            knowledge_port=repository,
            process_state_mask_enabled=enabled,
            process_state_mask_include_direct_successors=enabled,
            rs01_admissibility_policy=_audit_policy(),
        )
        for _model_family, encoder, enabled in (
            ("EOPKG", encoder_small, True),
            ("GATv2+Mask", encoder_other, False),
            ("LSTM", encoder_lstm, False),
        )
    ]
    payloads = [
        _audit_payload(builder, prefix)[0]
        for builder in builders
        for prefix in (prefix_a, prefix_b, target_changed)
    ]
    assert {tuple(payload["audit_allowed_activity_labels"]) for payload in payloads} == {("B", "C")}
    assert len({payload["audit_mask_policy_id"] for payload in payloads}) == 1
    assert not torch.equal(
        builders[0].build_graph(prefix_a)["allowed_target_mask"],
        builders[1].build_graph(prefix_a)["allowed_target_mask"],
    )


def test_common_mask_changes_with_topology_and_policy_fingerprint_changes_with_policy(mock_feature_configs):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", [_event(0, "A"), _event(1, "B"), _event(2, "C")])],
    )
    prefix = PrefixSlice(
        case_id="eval", process_version="v1",
        prefix_events=[_event(0, "A")], target_event=_event(1, "B"),
    )
    payloads = []
    for edge, policy in (
        (("A", "B"), _audit_policy()),
        (("A", "C"), _audit_policy()),
        (("A", "B"), _audit_policy(include_direct_successors=False)),
    ):
        repository = InMemoryNetworkXRepository()
        repository.save_process_structure("v1", ProcessStructureDTO(version="v1", allowed_edges=[edge]))
        payloads.append(_audit_payload(DynamicGraphBuilder(
            feature_encoder=encoder,
            knowledge_port=repository,
            rs01_admissibility_policy=policy,
        ), prefix)[0])
    assert payloads[0]["audit_allowed_activity_labels"] == ["B"]
    assert payloads[1]["audit_allowed_activity_labels"] == ["C"]
    assert payloads[0]["audit_mask_policy_id"] == payloads[1]["audit_mask_policy_id"]
    assert payloads[0]["audit_mask_policy_id"] != payloads[2]["audit_mask_policy_id"]


def test_reference_cardinality_preserves_structural_identities_before_label_projection(mock_feature_configs):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs,
                             traces=[_trace("train", "v1", [_event(0, "A")])])
    builder = DynamicGraphBuilder(feature_encoder=encoder,
                                  knowledge_port=InMemoryNetworkXRepository(),
                                  process_state_mask_relaxed_max_cardinality_ratio=0.5)
    compiled = {
        "candidate_count": 4,
        "candidate_ids": ("a1", "a2", "b", "c"),
        "candidate_labels": ("A", "A", "B", "C"),
        "candidate_indices_by_label": {"start": [0], "A": [0, 1], "B": [2], "C": [3]},
        "candidate_indices_by_id": {"a1": [0], "a2": [1], "b": [2], "c": [3]},
        "candidate_allowed_masks_by_src": {
            0: torch.tensor([False, True, True, True]),
        },
        "initial_candidate_labels": (),
    }
    prefix = PrefixSlice(case_id="eval", process_version="v1",
                         prefix_events=[_event(0, "start")], target_event=_event(1, "C"))
    selected = builder._relaxed_reachability_candidate_indices(
        prefix=prefix, compiled=compiled, active_candidates=set())
    assert selected == {1, 2}
    assert {compiled["candidate_labels"][idx] for idx in selected} == {"A", "B"}


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


def test_relaxed_reachability_keeps_lifecycle_start_token_when_direct_successors_are_transparent(
    mock_feature_configs,
):
    events = [
        _event(0, "A", lifecycle="start", instance_id="A_1"),
        _event(1, "B", lifecycle="start", instance_id="B_1"),
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
            nodes=[
                {"id": "node_A", "bpmn_tag": "task", "type": "task", "name": "A"},
                {"id": "join", "bpmn_tag": "parallelGateway", "type": "parallelGateway", "name": "join"},
                {"id": "node_B", "bpmn_tag": "task", "type": "task", "name": "B"},
            ],
            allowed_edges=[
                ("node_A", "join"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:1],
        target_event=events[1],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="topology_native",
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
        cache_policy="none",
    ).build_graph(prefix)

    assert int(contract["candidate_allowed_target_mask"].sum().item()) == 1
    assert int(contract["process_state_mask_active_candidate_count"].item()) == 1


def test_relaxed_reachability_keeps_not_completed_initial_parallel_sibling_with_collapsed_gateways(
    mock_feature_configs,
):
    events = [
        _event(0, "A", lifecycle="complete", instance_id="A_1"),
        _event(1, "B", lifecycle="complete", instance_id="B_1"),
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
            nodes=[
                {"id": "start", "bpmn_tag": "startEvent", "type": "startEvent", "name": "start"},
                {"id": "parallel_split", "bpmn_tag": "parallelGateway", "type": "parallelGateway", "name": "split"},
                {"id": "node_A", "bpmn_tag": "task", "type": "task", "name": "A"},
                {"id": "node_B", "bpmn_tag": "task", "type": "task", "name": "B"},
                {"id": "join", "bpmn_tag": "parallelGateway", "type": "parallelGateway", "name": "join"},
            ],
            allowed_edges=[
                ("start", "parallel_split"),
                ("parallel_split", "node_A"),
                ("parallel_split", "node_B"),
                ("node_A", "join"),
                ("node_B", "join"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval",
        process_version="v1",
        prefix_events=events[:1],
        target_event=events[1],
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="topology_native",
        process_state_mask_enabled=True,
        process_state_mask_source="relaxed_reachability",
        graph_feature_mapping={"topology_projection": {"gateway_mode": "collapse_for_prediction"}},
        cache_policy="none",
    ).build_graph(prefix)

    allowed_by_label = dict(zip(contract["candidate_labels"], contract["candidate_allowed_target_mask"].tolist()))
    assert allowed_by_label["B"] is True
    assert allowed_by_label["A"] is False
