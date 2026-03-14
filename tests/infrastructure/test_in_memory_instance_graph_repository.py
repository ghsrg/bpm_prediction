from __future__ import annotations

from datetime import datetime, timedelta

from src.domain.entities.process_event import ProcessEventDTO
from src.infrastructure.repositories.in_memory_instance_graph_repository import InMemoryInstanceGraphRepository


def _event(case_id: str, idx: int) -> ProcessEventDTO:
    start = datetime.fromisoformat("2026-03-10T10:00:00") + timedelta(minutes=idx)
    end = start + timedelta(seconds=30)
    return ProcessEventDTO(
        case_id=case_id,
        activity_def_id=f"Task_{idx}",
        act_inst_id=f"ai_{case_id}_{idx}",
        start_time=start,
        end_time=end,
        duration_ms=30000.0,
    )


def test_repository_saves_and_deduplicates_events():
    repository = InMemoryInstanceGraphRepository()
    event = _event("case_1", 1)
    repository.save_instance_events("procurement", "v1", [event, event])

    loaded = repository.get_instance_events("procurement", "v1")
    assert len(loaded) == 1
    assert loaded[0].activity_def_id == "Task_1"


def test_repository_filters_events_by_time_interval():
    repository = InMemoryInstanceGraphRepository()
    e1 = _event("case_1", 1)
    e2 = _event("case_1", 2)
    repository.save_instance_events("procurement", "v1", [e1, e2])

    loaded = repository.get_instance_events(
        "procurement",
        "v1",
        since=datetime.fromisoformat("2026-03-10T10:02:00"),
    )
    assert len(loaded) == 1
    assert loaded[0].activity_def_id == "Task_2"


def test_repository_persists_graph_projection_and_watermark():
    repository = InMemoryInstanceGraphRepository()
    repository.save_instance_graph("procurement", "v1", {"nodes": [{"id": "A"}], "edges": []})
    repository.save_instance_projection("procurement", "v1", {"node_counts": {"A": 10}})
    watermark = datetime.fromisoformat("2026-03-10T12:00:00")
    repository.set_last_watermark("procurement", "v1", watermark)

    assert repository.get_instance_graph("procurement", "v1") == {"nodes": [{"id": "A"}], "edges": []}
    assert repository.get_instance_projection("procurement", "v1") == {"node_counts": {"A": 10}}
    assert repository.get_last_watermark("procurement", "v1") == watermark

