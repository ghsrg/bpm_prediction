from __future__ import annotations

import matplotlib

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace

matplotlib.use("Agg")


def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700000000 + idx),
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
    events = [_event(i, act) for i, act in enumerate(activities)]
    return RawTrace(case_id=case_id, process_version=version, events=events, trace_attributes={})


def test_extract_from_logs_builds_unique_edges_grouped_by_version_without_leakage():
    service = TopologyExtractorService()

    train_traces = [
        _trace("c1", "v1", ["A", "B", "C"]),
        _trace("c2", "v1", ["A", "B", "D"]),
        _trace("c3", "v1", ["A", "B", "C"]),  # duplicate transitions
        _trace("c4", "v2", ["X", "Y", "Z"]),
    ]
    holdout_trace_not_used = _trace("h1", "v1", ["Q", "W"])  # leakage sentinel

    service.extract_from_logs(train_traces)

    dto_v1 = service.get_process_structure("v1")
    dto_v2 = service.get_process_structure("v2")
    assert dto_v1 is not None
    assert dto_v2 is not None

    assert set(dto_v1.allowed_edges) == {("A", "B"), ("B", "C"), ("B", "D")}
    assert set(dto_v2.allowed_edges) == {("X", "Y"), ("Y", "Z")}

    # Data leakage check: transitions from unseen holdout trace must not appear.
    assert tuple(("Q", "W")) not in set(dto_v1.allowed_edges)
    assert holdout_trace_not_used.process_version == "v1"


def test_get_process_structure_returns_none_for_unknown_version():
    service = TopologyExtractorService()
    service.extract_from_logs([_trace("c1", "v1", ["A", "B"])])

    assert service.get_process_structure("v_unknown") is None


def test_extract_from_bpmn_is_explicit_not_implemented_yet():
    service = TopologyExtractorService()

    try:
        service.extract_from_bpmn({"dummy": "bpmn"})
        assert False, "Expected NotImplementedError for BPMN path in MVP2 Stage 2."
    except NotImplementedError as exc:
        assert str(exc) == "Will be implemented in Enterprise PoC for Camunda integration"


def test_export_to_networkx_returns_digraph_with_expected_edges():
    service = TopologyExtractorService()
    service.fit([_trace("c1", "v1", ["A", "B", "C"])])

    graph = service.export_to_networkx("v1")
    assert graph.is_directed()
    assert set(graph.edges()) == {("A", "B"), ("B", "C")}


def test_plot_topology_saves_png(tmp_path):
    service = TopologyExtractorService()
    service.fit([_trace("c1", "v1", ["A", "B", "C"])])

    output = tmp_path / "topology_v1.png"
    service.plot_topology("v1", save_path=str(output))

    assert output.exists()
    assert output.stat().st_size > 0


def test_extract_from_logs_normalizes_empty_version_to_one_and_exposes_available_versions():
    service = TopologyExtractorService()
    service.fit([_trace("c1", "", ["A", "B"])])

    assert service.available_versions == ["1"]
    dto = service.get_process_structure("1")
    assert dto is not None
    assert set(dto.allowed_edges) == {("A", "B")}
