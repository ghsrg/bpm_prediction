from __future__ import annotations

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace


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
    assert dto_v1.edge_statistics is not None
    assert dto_v1.edge_statistics[("A", "B")]["count"] == 3.0
    assert dto_v1.edge_statistics[("B", "C")]["count"] == 2.0
    assert dto_v1.edge_statistics[("B", "D")]["count"] == 1.0

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


def test_build_dfg_payload_returns_expected_edges_and_start_end_activity_maps():
    service = TopologyExtractorService()
    service.fit([_trace("c1", "v1", ["A", "B", "C"]), _trace("c2", "v1", ["A", "B", "D"])])

    dfg, start_activities, end_activities = service._build_dfg_payload("v1")
    assert dfg == {("A", "B"): 2, ("B", "C"): 1, ("B", "D"): 1}
    assert start_activities == {"A": 2}
    assert end_activities == {"C": 1, "D": 1}


def test_plot_topology_saves_png_with_pm4py_visualizer(tmp_path, monkeypatch):
    service = TopologyExtractorService()
    service.fit([_trace("c1", "v1", ["A", "B", "C"]), _trace("c2", "v1", ["A", "B"])])

    captured = {}

    def _mock_apply(dfg0, log=None, activities_count=None, serv_time=None, parameters=None, variant=None):
        captured["dfg"] = dfg0
        captured["parameters"] = parameters
        captured["variant"] = variant
        return "gviz_obj"

    def _mock_save(gviz, path):
        captured["saved"] = (gviz, path)
        with open(path, "wb") as stream:
            stream.write(b"fake-image")

    monkeypatch.setattr("src.application.services.topology_extractor_service.dfg_visualizer.apply", _mock_apply)
    monkeypatch.setattr("src.application.services.topology_extractor_service.dfg_visualizer.save", _mock_save)

    output = tmp_path / "topology_v1.png"
    service.plot_topology("v1", save_path=str(output), min_edge_freq=2)

    assert captured["dfg"] == {("A", "B"): 2}
    assert captured["saved"] == ("gviz_obj", str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_topology_raises_when_frequency_filter_removes_all_edges():
    service = TopologyExtractorService()
    service.fit([_trace("c1", "v1", ["A", "B", "C"])])

    try:
        service.plot_topology("v1", min_edge_freq=2)
        assert False, "Expected ValueError when all edges are removed by min_edge_freq."
    except ValueError as exc:
        assert "after min_edge_freq=2 filter" in str(exc)


def test_extract_from_logs_normalizes_empty_version_to_one_and_exposes_available_versions():
    service = TopologyExtractorService()
    service.fit([_trace("c1", "", ["A", "B"])])

    assert service.available_versions == ["1"]
    dto = service.get_process_structure("1")
    assert dto is not None
    assert set(dto.allowed_edges) == {("A", "B")}
