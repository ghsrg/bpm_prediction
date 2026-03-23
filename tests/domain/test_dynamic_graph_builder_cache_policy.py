from __future__ import annotations

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700700000 + idx),
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


def _dto() -> ProcessStructureDTO:
    return ProcessStructureDTO(
        version="v1",
        allowed_edges=[("Start", "Approve"), ("Approve", "End")],
        nodes=[
            {"id": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        metadata={
            "knowledge_version": "k000500",
            "as_of_ts": "2026-03-10T09:32:06.153000+00:00",
            "stats_index": {
                "node": {
                    "all_time.version.exec_count": {"Start": 3.0, "Approve": 7.0, "End": 11.0},
                },
                "edge": {
                    "all_time.version.transition_probability": {"Start|||Approve": 0.7, "Approve|||End": 0.9},
                },
                "global": {},
            },
        },
    )


class _CountingRepo:
    def __init__(self, dto: ProcessStructureDTO) -> None:
        self._dto = dto
        self.calls_asof = 0
        self.calls_latest = 0

    def get_process_structure_as_of(self, version: str, process_name: str | None = None, *, as_of_ts=None):
        _ = version
        _ = process_name
        _ = as_of_ts
        self.calls_asof += 1
        return self._dto

    def get_process_structure(self, version: str, process_name: str | None = None):
        _ = version
        _ = process_name
        self.calls_latest += 1
        return self._dto


class _CountingBuilder(DynamicGraphBuilder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.struct_x_calls = 0
        self.edge_stats_calls = 0

    def _build_struct_x(self, *, dto: ProcessStructureDTO, activity_vocab: dict[str, int]):
        self.struct_x_calls += 1
        return super()._build_struct_x(dto=dto, activity_vocab=activity_vocab)

    def _edge_stats_index(self, dto: ProcessStructureDTO):
        self.edge_stats_calls += 1
        return super()._edge_stats_index(dto)


def test_cache_policy_full_caches_dto_for_strict_asof(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _CountingRepo(_dto())
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        stats_time_policy="strict_asof",
        cache_policy="full",
    )

    prefix = _prefix()
    for _ in range(5):
        contract = builder.build_graph(prefix)
        assert contract["allowed_target_mask"] is not None

    assert repo.calls_asof == 1


def test_cache_policy_off_disables_dto_cache_for_strict_asof(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _CountingRepo(_dto())
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        stats_time_policy="strict_asof",
        cache_policy="off",
    )

    prefix = _prefix()
    for _ in range(5):
        contract = builder.build_graph(prefix)
        assert contract["allowed_target_mask"] is not None

    assert repo.calls_asof == 5


def test_cache_policy_full_caches_compiled_topology(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _CountingRepo(_dto())
    graph_feature_mapping = {
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
        "edge_weight": {
            "metric": "transition_probability",
            "window": "all_time",
            "scope": "version",
            "default": 1.0,
            "encoding": ["identity"],
        },
    }
    builder = _CountingBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping=graph_feature_mapping,
        cache_policy="full",
    )
    prefix = _prefix()

    for _ in range(3):
        contract = builder.build_graph(prefix)
        assert contract.get("struct_x") is not None

    assert repo.calls_latest == 1
    assert builder.struct_x_calls == 1
    assert builder.edge_stats_calls == 1


def test_cache_policy_dto_does_not_cache_compiled_topology(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"])]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _CountingRepo(_dto())
    graph_feature_mapping = {
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
        "edge_weight": {
            "metric": "transition_probability",
            "window": "all_time",
            "scope": "version",
            "default": 1.0,
            "encoding": ["identity"],
        },
    }
    builder = _CountingBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping=graph_feature_mapping,
        cache_policy="dto",
    )
    prefix = _prefix()

    for _ in range(3):
        contract = builder.build_graph(prefix)
        assert contract.get("struct_x") is not None

    assert repo.calls_latest == 1
    assert builder.struct_x_calls == 3
    assert builder.edge_stats_calls == 3
