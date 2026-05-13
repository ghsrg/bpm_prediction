from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder


def _event(idx: int, activity: str, ts: float) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=ts,
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R1"},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str], base_ts: float) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, activity, base_ts + float(i * 60)) for i, activity in enumerate(activities)],
        trace_attributes={},
    )


def _prefix(version: str, activities: list[str], *, target: str, last_ts: float) -> PrefixSlice:
    base_ts = last_ts - float(max(0, len(activities) - 1) * 60)
    return PrefixSlice(
        case_id="case_cache",
        process_version=version,
        prefix_events=[_event(i, activity, base_ts + float(i * 60)) for i, activity in enumerate(activities)],
        target_event=_event(len(activities), target, last_ts + 60.0),
    )


class _CountingAsOfPort:
    def __init__(self, dto: ProcessStructureDTO) -> None:
        self.dto = dto
        self.calls: list[datetime | None] = []

    def get_process_structure_as_of(
        self,
        version: str,
        process_name: str | None = None,
        as_of_ts: datetime | None = None,
    ) -> ProcessStructureDTO:
        _ = version
        _ = process_name
        self.calls.append(as_of_ts)
        return self.dto.model_copy(deep=True)

    def get_process_structure(self, version: str, process_name: str | None = None) -> ProcessStructureDTO:
        _ = version
        _ = process_name
        return self.dto.model_copy(deep=True)


def test_dynamic_builder_cache_diagnostics_tracks_unique_snapshot_identity(mock_feature_configs, tmp_path):
    traces = [_trace("c1", "v1", ["A", "B", "C"], base_ts=1772409600.0)]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("A", "B"), ("B", "C")],
        metadata={"knowledge_version": "k000010", "as_of_ts": "2026-03-01T00:00:00+00:00"},
    )
    port = _CountingAsOfPort(dto)
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=port,
        process_name="loan_v1_v4_simulated",
        stats_time_policy="strict_asof",
        graph_feature_mapping={"enabled": True},
        cache_policy="full",
        cache_dir=str(tmp_path / "dynamic_graph_cache"),
    )

    for offset in range(20):
        prefix = _prefix("v1", ["A", "B"], target="C", last_ts=1772409600.0 + offset * 60.0)
        builder.build_graph(prefix)

    diagnostics = builder.cache_diagnostics()
    assert diagnostics["unique_snapshot_identities"] == 1
    assert diagnostics["topology_cache_misses"] == 1
    assert diagnostics["topology_cache_hits"] >= 19
    assert diagnostics["dto_cache_entries"] == 0
