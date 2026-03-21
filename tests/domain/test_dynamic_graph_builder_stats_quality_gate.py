from __future__ import annotations

import torch
import pytest

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
        timestamp=float(1700500000 + idx),
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


def _graph_feature_mapping() -> dict:
    return {
        "enabled": True,
        "node_numeric": [
            {
                "name": "node_exec_count_v",
                "metric": "exec_count",
                "window": "all_time",
                "scope": "version",
                "default": 0.0,
            }
        ],
        "edge_weight": {
            "metric": "transition_probability",
            "window": "all_time",
            "scope": "version",
            "default": 1.0,
        },
    }


def _dto_with_stats(*, usable: bool, reason: str = "ok") -> ProcessStructureDTO:
    return ProcessStructureDTO(
        version="v1",
        allowed_edges=[("Start", "Approve"), ("Approve", "End")],
        nodes=[
            {"id": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        metadata={
            "stats_index": {
                "node": {
                    "all_time.version.exec_count": {"Start": 10.0, "Approve": 8.0, "End": 7.0},
                },
                "edge": {
                    "all_time.version.transition_probability": {
                        "Start|||Approve": 0.9,
                        "Approve|||End": 0.8,
                    }
                },
                "global": {
                    "all_time.version.coverage_percent": 100.0,
                },
            },
            "stats_contract": {
                "version": "1.0",
                "quality": {
                    "history_coverage_percent": 100.0,
                    "non_zero_ratio_overall": 0.75 if usable else 0.01,
                    "zero_dominant": not usable,
                    "is_usable_for_training": usable,
                    "quality_reason": reason,
                },
            },
        },
    )


def test_dynamic_graph_builder_quality_gate_ignores_stats_on_fail(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure("v1", _dto_with_stats(usable=False, reason="zero_dominant"), process_name="dataset_a")

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_name="dataset_a",
        graph_feature_mapping={
            **_graph_feature_mapping(),
            "stats_quality_gate": {
                "enabled": True,
                "on_fail": "ignore_with_warning",
            },
        },
    )
    contract = builder.build_graph(prefix)

    assert "struct_x" not in contract
    assert contract["structural_edge_weight"].dtype == torch.float32
    assert torch.allclose(contract["structural_edge_weight"], torch.ones_like(contract["structural_edge_weight"]))


def test_dynamic_graph_builder_quality_gate_allows_stats_when_policy_allows(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure("v1", _dto_with_stats(usable=False, reason="zero_dominant"), process_name="dataset_a")

    prefix = PrefixSlice(
        case_id="eval_case",
        process_version="v1",
        prefix_events=[_event(0, "Start"), _event(1, "Approve")],
        target_event=_event(2, "End"),
    )
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_name="dataset_a",
        graph_feature_mapping={
            **_graph_feature_mapping(),
            "stats_quality_gate": {
                "enabled": True,
                "on_fail": "allow_with_warning",
            },
        },
    )
    contract = builder.build_graph(prefix)

    assert "struct_x" in contract
    assert contract["struct_x"] is not None
    assert contract["structural_edge_weight"].dtype == torch.float32
    assert contract["structural_edge_weight"].shape[0] == 2
    sorted_weights = sorted(float(item) for item in contract["structural_edge_weight"].tolist())
    assert sorted_weights == pytest.approx([0.8, 0.9], abs=1e-6)
