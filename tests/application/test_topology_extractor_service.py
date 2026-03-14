from __future__ import annotations

from src.application.services.topology_extractor_service import TopologyExtractorService
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _event(idx: int, activity: str, activity_name: str | None = None, activity_type: str | None = None) -> EventRecord:
    extra = {"concept:name": activity}
    if activity_name is not None:
        extra["activity_name"] = activity_name
    if activity_type is not None:
        extra["activity_type"] = activity_type
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700000000 + idx),
        resource_id="R",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra=extra,
        activity_instance_id=f"ai_{idx}",
    )


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    events = [_event(i, act) for i, act in enumerate(activities)]
    return RawTrace(case_id=case_id, process_version=version, events=events, trace_attributes={})


def _service(process_name: str = "dataset_a") -> TopologyExtractorService:
    return TopologyExtractorService(knowledge_port=InMemoryNetworkXRepository(), process_name=process_name)


def test_extract_from_logs_builds_unique_edges_grouped_by_version_without_leakage():
    service = _service("dataset_a")

    train_traces = [
        _trace("c1", "v1", ["A", "B", "C"]),
        _trace("c2", "v1", ["A", "B", "D"]),
        _trace("c3", "v1", ["A", "B", "C"]),  # duplicate transitions
        _trace("c4", "v2", ["X", "Y", "Z"]),
    ]
    holdout_trace_not_used = _trace("h1", "v1", ["Q", "W"])  # leakage sentinel

    service.extract_from_logs(train_traces, process_name="dataset_a")

    dto_v1 = service.get_process_structure("v1", process_name="dataset_a")
    dto_v2 = service.get_process_structure("v2", process_name="dataset_a")
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
    service = _service("dataset_a")
    service.extract_from_logs([_trace("c1", "v1", ["A", "B"])], process_name="dataset_a")

    assert service.get_process_structure("v_unknown", process_name="dataset_a") is None


def test_extract_from_bpmn_is_explicit_not_implemented_yet():
    service = _service("dataset_a")

    try:
        service.extract_from_bpmn({"dummy": "bpmn"})
        assert False, "Expected NotImplementedError for BPMN path in MVP2 Stage 2."
    except NotImplementedError as exc:
        assert str(exc) == "Will be implemented in Enterprise PoC for Camunda integration"


def test_build_dfg_payload_returns_expected_edges_and_start_end_activity_maps():
    service = _service("dataset_a")
    service.fit(
        [_trace("c1", "v1", ["A", "B", "C"]), _trace("c2", "v1", ["A", "B", "D"])],
        process_name="dataset_a",
    )

    dfg, start_activities, end_activities = service._build_dfg_payload(
        "v1",
        process_name="dataset_a",
        min_edge_frequency=1,
    )
    assert dfg == {("A", "B"): 2, ("B", "C"): 1, ("B", "D"): 1}
    assert start_activities == {"A": 2}
    assert end_activities == {"C": 1, "D": 1}


def test_plot_topology_saves_png_with_pm4py_visualizer(tmp_path, monkeypatch):
    service = _service("dataset_a")
    service.fit(
        [_trace("c1", "v1", ["A", "B", "C"]), _trace("c2", "v1", ["A", "B"])],
        process_name="dataset_a",
    )

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
    service.plot_topology("v1", process_name="dataset_a", save_path=str(output), min_edge_freq=2)

    assert captured["dfg"] == {("A", "B"): 2}
    assert captured["saved"] == ("gviz_obj", str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_topology_raises_when_frequency_filter_removes_all_edges():
    service = _service("dataset_a")
    service.fit([_trace("c1", "v1", ["A", "B", "C"])], process_name="dataset_a")

    try:
        service.plot_topology("v1", process_name="dataset_a", min_edge_freq=2)
        assert False, "Expected ValueError when all edges are removed by min_edge_freq."
    except ValueError as exc:
        assert "after min_edge_freq=2 filter" in str(exc)


def test_extract_from_logs_falls_back_to_process_name_for_empty_version():
    service = _service("dataset_alpha")
    service.fit([_trace("c1", "", ["A", "B"])], process_name="dataset_alpha")

    assert service.available_versions == ["dataset_alpha"]
    dto = service.get_process_structure("dataset_alpha", process_name="dataset_alpha")
    assert dto is not None
    assert set(dto.allowed_edges) == {("A", "B")}


def test_extract_from_logs_persists_node_metadata_for_visualization():
    service = _service("dataset_a")
    trace = RawTrace(
        case_id="c1",
        process_version="v1",
        events=[
            _event(0, "StartEvent_1", activity_name="Start", activity_type="startEvent"),
            _event(1, "Task_Approve", activity_name="Approve request", activity_type="userTask"),
            _event(2, "EndEvent_1", activity_name="End", activity_type="endEvent"),
        ],
        trace_attributes={},
    )
    service.fit([trace], process_name="dataset_a")

    graph = service.knowledge_port.get_graph_for_visualization("dataset_a", "v1", min_edge_frequency=0)
    assert graph.nodes["Task_Approve"]["activity_name"] == "Approve request"
    assert graph.nodes["Task_Approve"]["activity_type"] == "userTask"
    assert graph.nodes["StartEvent_1"]["activity_type"] == "startEvent"


def test_graphviz_category_mapping_respects_activity_type_and_start_end_fallback():
    service = _service("dataset_a")
    # type-based mapping
    assert service._classify_node_category(graph=None, node_id="n1", attrs={"activity_type": "serviceTask"}) == "service_task"
    assert service._classify_node_category(graph=None, node_id="n2", attrs={"activity_type": "exclusiveGateway"}) == "gateway"
    assert service._classify_node_category(graph=None, node_id="n3", attrs={"activity_type": "timerBoundaryEvent"}) == "event_other"

    # fallback mapping by in/out degree when type is absent
    service.fit([_trace("c1", "v1", ["A", "B", "C"])], process_name="dataset_a")
    graph = service.knowledge_port.get_graph_for_visualization("dataset_a", "v1", min_edge_frequency=0)
    assert service._classify_node_category(graph=graph, node_id="A", attrs={}) == "start"
    assert service._classify_node_category(graph=graph, node_id="C", attrs={}) == "end"


def test_extract_from_logs_filters_invalid_edges_into_start_and_out_of_end_events():
    service = _service("dataset_a")
    trace = RawTrace(
        case_id="c1",
        process_version="v1",
        events=[
            _event(0, "StartEvent_1", activity_name="Start", activity_type="startEvent"),
            _event(1, "Task_A", activity_name="Task A", activity_type="userTask"),
            _event(2, "EndEvent_1", activity_name="End", activity_type="endEvent"),
            _event(3, "StartEvent_2", activity_name="Start2", activity_type="startEvent"),
            _event(4, "Task_B", activity_name="Task B", activity_type="serviceTask"),
            _event(5, "EndEvent_2", activity_name="End2", activity_type="endEvent"),
        ],
        trace_attributes={},
    )

    service.fit([trace], process_name="dataset_a")
    dto = service.get_process_structure("v1", process_name="dataset_a")
    assert dto is not None

    edges = set(dto.allowed_edges)
    assert ("StartEvent_1", "Task_A") in edges
    assert ("Task_A", "EndEvent_1") in edges
    assert ("Task_B", "EndEvent_2") in edges
    assert ("EndEvent_1", "StartEvent_2") not in edges
    assert ("EndEvent_1", "Task_B") not in edges
    assert ("Task_A", "StartEvent_2") not in edges

    graph = service.knowledge_port.get_graph_for_visualization("dataset_a", "v1", min_edge_frequency=0)
    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("activity_type", "")).lower()
        if "startevent" in node_type:
            assert graph.in_degree(node_id) == 0
        if "endevent" in node_type:
            assert graph.out_degree(node_id) == 0
