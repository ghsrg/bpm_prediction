from __future__ import annotations

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700000000 + idx),
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity},
        activity_instance_id=f"ai_{idx}",
    )


def test_raw_trace_and_prefix_slice_default_process_version():
    events = [_event(0, "A"), _event(1, "B")]
    trace = RawTrace(case_id="c1", events=events, trace_attributes={})
    assert trace.process_version == "1"

    prefix = PrefixSlice(case_id="c1", prefix_events=[events[0]], target_event=events[1])
    assert prefix.process_version == "1"


def test_process_structure_dto_accepts_required_and_optional_fields():
    dto = ProcessStructureDTO(
        version="2",
        allowed_edges=[("A", "B"), ("B", "C")],
        edge_statistics={("A", "B"): {"count": 10.0, "prob": 0.5}},
        proc_def_id="def_2",
        proc_def_key="proc_key",
        call_bindings={"call_1": {"status": "unresolved", "inference_fallback_strategy": "use_aggregated_stats"}},
    )

    assert dto.version == "2"
    assert ("A", "B") in dto.allowed_edges
    assert dto.edge_statistics is not None
    assert dto.edge_statistics[("A", "B")]["count"] == 10.0
    assert dto.proc_def_id == "def_2"
    assert dto.call_bindings is not None
