import pytest

from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.prefix_policy import PrefixPolicy


def _event(idx: int) -> EventRecord:
    return EventRecord(
        activity_id=f"A{idx}",
        timestamp=1700000000.0 + idx,
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": f"A{idx}"},
        activity_instance_id=f"ai_{idx}",
    )


@pytest.mark.parametrize(("num_events", "expected_slices"), [(1, 0), (3, 2)])
def test_generate_slices_count(num_events, expected_slices):
    trace = RawTrace(
        case_id="case_x",
        process_version="v1",
        events=[_event(i) for i in range(num_events)],
        trace_attributes={},
    )

    slices = PrefixPolicy().generate_slices(trace)
    assert len(slices) == expected_slices

