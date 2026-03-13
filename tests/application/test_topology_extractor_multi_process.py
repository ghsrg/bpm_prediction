from __future__ import annotations

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700500000 + idx),
        resource_id="R",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity},
        activity_instance_id=f"ai_{idx}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_topology_extractor_keeps_process_namespace_isolation_in_shared_repository():
    repository = InMemoryNetworkXRepository()
    service = TopologyExtractorService(knowledge_port=repository)

    traces_a = [_trace("a1", "v1", ["A", "B", "C"])]
    traces_b = [_trace("b1", "v1", ["X", "Y", "Z"])]

    service.extract_from_logs(traces_a, process_name="dataset_a")
    service.extract_from_logs(traces_b, process_name="dataset_b")

    dto_a = repository.get_process_structure("v1", process_name="dataset_a")
    dto_b = repository.get_process_structure("v1", process_name="dataset_b")

    assert dto_a is not None and set(dto_a.allowed_edges) == {("A", "B"), ("B", "C")}
    assert dto_b is not None and set(dto_b.allowed_edges) == {("X", "Y"), ("Y", "Z")}
    assert repository.list_versions("dataset_a") == ["v1"]
    assert repository.list_versions("dataset_b") == ["v1"]
