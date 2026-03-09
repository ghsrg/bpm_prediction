import pytest

from src.domain.entities.event_record import EventRecord
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.raw_trace import RawTrace


@pytest.fixture()
def mock_feature_configs() -> list[FeatureConfig]:
    return [
        FeatureConfig(
            name="concept:name",
            source_key="activity",
            source="event",
            dtype="string",
            fill_na="<UNK>",
            encoding=["embedding"],
            role="activity",
        ),
        FeatureConfig(
            name="org:resource",
            source_key=None,
            source="event",
            dtype="string",
            fill_na="<UNK>",
            encoding=["embedding"],
            role=None,
        ),
        FeatureConfig(
            name="cost",
            source_key="amount",
            source="event",
            dtype="float",
            fill_na=0.0,
            encoding=["z-score"],
            role=None,
        ),
    ]


@pytest.fixture()
def mock_raw_trace() -> RawTrace:
    events = [
        EventRecord(
            activity_id="Start",
            timestamp=1700000000.0,
            resource_id="R1",
            lifecycle="complete",
            position_in_trace=0,
            duration=10.0,
            time_since_case_start=0.0,
            time_since_previous_event=0.0,
            extra={"concept:name": "Start", "org:resource": "R1", "amount": 100.0},
            activity_instance_id="ai_1",
        ),
        EventRecord(
            activity_id="Approve",
            timestamp=1700000100.0,
            resource_id="R2",
            lifecycle="complete",
            position_in_trace=1,
            duration=15.0,
            time_since_case_start=100.0,
            time_since_previous_event=100.0,
            extra={"activity": "Approve", "org:resource": "R2", "amount": 200.0},
            activity_instance_id="ai_2",
        ),
        EventRecord(
            activity_id="End",
            timestamp=1700000200.0,
            resource_id="R1",
            lifecycle="complete",
            position_in_trace=2,
            duration=8.0,
            time_since_case_start=200.0,
            time_since_previous_event=100.0,
            extra={"concept:name": "End", "org:resource": "R1", "amount": 300.0},
            activity_instance_id="ai_3",
        ),
    ]

    return RawTrace(
        case_id="case_1",
        process_version="v1",
        events=events,
        trace_attributes={"customer_segment": "A"},
    )
