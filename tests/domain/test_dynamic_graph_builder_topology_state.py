from __future__ import annotations

import math

import pytest
import torch

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(idx: int, activity: str, *, extra: dict | None = None) -> EventRecord:
    payload = {"concept:name": activity, "org:resource": "R1", "amount": float(idx + 1)}
    if extra:
        payload.update(extra)
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700100000 + idx),
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra=payload,
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_dynamic_graph_builder_adds_topology_prefix_state_features(mock_feature_configs):
    train_traces = [
        _trace("c1", "v1", ["Start", "Approve", "Approve", "End"]),
        _trace("c2", "v1", ["Start", "Approve", "End"]),
    ]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=train_traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[
                ("Start", "Approve"),
                ("Approve", "End"),
            ],
        ),
    )

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[
            _event(0, "Start"),
            _event(1, "Approve"),
            _event(
                2,
                "Approve",
                extra={"active_activities_after_complete": ["End"]},
            ),
        ],
        target_event=_event(3, "End"),
    )

    contract = DynamicGraphBuilder(feature_encoder=encoder, knowledge_port=repository).build_graph(prefix)

    state = contract["struct_prefix_state_x"]
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert state.dtype == torch.float32
    assert state.shape == torch.Size([len(activity_vocab), 6])

    start = state[int(activity_vocab["Start"])]
    approve = state[int(activity_vocab["Approve"])]
    end = state[int(activity_vocab["End"])]

    assert float(start[0]) == pytest.approx(math.log1p(1), abs=1e-6)
    assert float(start[1]) == pytest.approx(1.0)
    assert float(start[2]) == pytest.approx(0.0)
    assert float(start[3]) == pytest.approx(1.0 / 3.0)
    assert float(start[4]) == pytest.approx(1.0 / 3.0)
    assert float(start[5]) == pytest.approx(0.0)

    assert float(approve[0]) == pytest.approx(math.log1p(2), abs=1e-6)
    assert float(approve[1]) == pytest.approx(1.0)
    assert float(approve[2]) == pytest.approx(1.0)
    assert float(approve[3]) == pytest.approx(1.0)
    assert float(approve[4]) == pytest.approx(1.0)
    assert float(approve[5]) == pytest.approx(0.0)

    assert float(end[0]) == pytest.approx(0.0)
    assert float(end[1]) == pytest.approx(0.0)
    assert float(end[2]) == pytest.approx(0.0)
    assert float(end[3]) == pytest.approx(0.0)
    assert float(end[4]) == pytest.approx(0.0)
    assert float(end[5]) == pytest.approx(1.0)
