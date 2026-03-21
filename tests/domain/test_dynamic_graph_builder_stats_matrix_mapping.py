from __future__ import annotations

from datetime import datetime, timezone
import pytest
import torch

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


def _prefix() -> PrefixSlice:
    return PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )


def _build_repo_with_stats(
    *,
    node_values: dict[str, float],
    edge_values: dict[str, float] | None = None,
) -> InMemoryNetworkXRepository:
    repo = InMemoryNetworkXRepository()
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("Start", "Approve"), ("Approve", "End")],
        nodes=[
            {"id": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        metadata={
            "knowledge_version": "k000123",
            "as_of_ts": "2026-03-20T10:30:00+00:00",
            "stats_index": {
                "node": {
                    "all_time.version.exec_count": dict(node_values),
                },
                "edge": {
                    "all_time.version.transition_probability": dict(edge_values or {}),
                },
                "global": {},
            }
        },
    )
    repo.save_process_structure("v1", dto, process_name="dataset_a")
    return repo


def test_struct_x_identity_mapping_uses_defaults(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _build_repo_with_stats(node_values={"Start": 10.0, "Approve": 20.0})

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [
                {
                    "name": "node_exec_count_v",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": 5.0,
                    "encoding": ["identity"],
                }
            ],
        },
    )
    contract = builder.build_graph(_prefix())
    struct_x = contract.get("struct_x")
    assert struct_x is not None
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert float(struct_x[int(activity_vocab["Start"]), 0]) == pytest.approx(10.0, abs=1e-6)
    assert float(struct_x[int(activity_vocab["Approve"]), 0]) == pytest.approx(20.0, abs=1e-6)
    assert float(struct_x[int(activity_vocab["End"]), 0]) == pytest.approx(5.0, abs=1e-6)


def test_struct_x_applies_log1p_then_zscore(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _build_repo_with_stats(node_values={"Start": 0.0, "Approve": 3.0, "End": 8.0})

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [
                {
                    "name": "node_exec_count_v",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": 0.0,
                    "encoding": ["log1p", "z-score"],
                }
            ],
        },
    )
    contract = builder.build_graph(_prefix())
    struct_x = contract.get("struct_x")
    assert struct_x is not None

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    raw = torch.zeros((len(activity_vocab),), dtype=torch.float32)
    raw[int(activity_vocab["Start"])] = 0.0
    raw[int(activity_vocab["Approve"])] = 3.0
    raw[int(activity_vocab["End"])] = 8.0
    transformed = torch.sign(raw) * torch.log1p(torch.abs(raw))
    std = torch.std(transformed, unbiased=False)
    if float(std) <= 1e-12:
        expected = torch.zeros_like(transformed)
    else:
        expected = (transformed - torch.mean(transformed)) / std

    assert torch.allclose(struct_x[:, 0], expected, atol=1e-6, rtol=0.0)


def test_edge_weight_applies_encoding(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _build_repo_with_stats(
        node_values={"Start": 1.0, "Approve": 1.0, "End": 1.0},
        edge_values={"Start|||Approve": 3.0, "Approve|||End": 8.0},
    )

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [],
            "edge_weight": {
                "metric": "transition_probability",
                "window": "all_time",
                "scope": "version",
                "default": 1.0,
                "encoding": ["log1p"],
            },
        },
    )
    contract = builder.build_graph(_prefix())
    weights = contract["structural_edge_weight"]
    expected = sorted([float(torch.log1p(torch.tensor(3.0))), float(torch.log1p(torch.tensor(8.0)))])
    actual = sorted(float(item) for item in weights.tolist())
    assert actual == pytest.approx(expected, abs=1e-6)


def test_builder_injects_snapshot_metadata_into_contract(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _build_repo_with_stats(node_values={"Start": 2.0, "Approve": 3.0, "End": 4.0})

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [
                {
                    "name": "node_exec_count_v",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": 0.0,
                    "encoding": ["identity"],
                }
            ],
        },
    )
    contract = builder.build_graph(_prefix())
    assert contract.get("stats_snapshot_version_seq") == 123
    expected_epoch = float(datetime(2026, 3, 20, 10, 30, tzinfo=timezone.utc).timestamp())
    assert float(contract.get("stats_snapshot_as_of_epoch") or 0.0) == pytest.approx(expected_epoch, abs=1e-6)
    assert contract.get("stats_allowed") is True


def test_builder_reads_snapshot_metadata_from_stats_contract_identity(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = InMemoryNetworkXRepository()
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("Start", "Approve"), ("Approve", "End")],
        nodes=[
            {"id": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        metadata={
            "stats_contract": {
                "identity": {
                    "knowledge_version": "k000321",
                    "as_of_ts": "2026-03-20T18:15:00+00:00",
                }
            },
            "stats_index": {
                "node": {
                    "all_time.version.exec_count": {"Start": 1.0, "Approve": 2.0, "End": 3.0},
                },
                "edge": {},
                "global": {},
            },
        },
    )
    repo.save_process_structure("v1", dto, process_name="dataset_a")
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [
                {
                    "name": "node_exec_count_v",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": 0.0,
                    "encoding": ["identity"],
                }
            ],
        },
    )

    contract = builder.build_graph(_prefix())
    assert contract.get("stats_snapshot_version_seq") == 321
    expected_epoch = float(datetime(2026, 3, 20, 18, 15, tzinfo=timezone.utc).timestamp())
    assert float(contract.get("stats_snapshot_as_of_epoch") or 0.0) == pytest.approx(expected_epoch, abs=1e-6)


def test_builder_treats_string_none_snapshot_version_as_missing(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = InMemoryNetworkXRepository()
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("Start", "Approve"), ("Approve", "End")],
        nodes=[
            {"id": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        metadata={
            "stats_contract": {
                "identity": {
                    "knowledge_version": "None",
                    "as_of_ts": "2026-03-20T18:15:00+00:00",
                }
            },
            "stats_index": {
                "node": {
                    "all_time.version.exec_count": {"Start": 1.0, "Approve": 2.0, "End": 3.0},
                },
                "edge": {},
                "global": {},
            },
        },
    )
    repo.save_process_structure("v1", dto, process_name="dataset_a")
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [
                {
                    "name": "node_exec_count_v",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": 0.0,
                    "encoding": ["identity"],
                }
            ],
        },
    )
    contract = builder.build_graph(_prefix())
    assert contract.get("stats_snapshot_version_seq") is None
