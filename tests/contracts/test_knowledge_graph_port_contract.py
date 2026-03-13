from __future__ import annotations

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _dto(version: str, edges: list[tuple[str, str]], counts: dict[tuple[str, str], float] | None = None) -> ProcessStructureDTO:
    stats = {edge: {"count": float((counts or {}).get(edge, 1.0))} for edge in edges}
    return ProcessStructureDTO(version=version, allowed_edges=edges, edge_statistics=stats)


def test_port_contract_save_and_get_are_process_scoped():
    port = InMemoryNetworkXRepository()
    port.save_process_structure("v1", _dto("v1", [("A", "B")]), process_name="p_camunda")
    port.save_process_structure("v1", _dto("v1", [("X", "Y")]), process_name="p_xes")

    camunda = port.get_process_structure("v1", process_name="p_camunda")
    xes = port.get_process_structure("v1", process_name="p_xes")
    assert camunda is not None and set(camunda.allowed_edges) == {("A", "B")}
    assert xes is not None and set(xes.allowed_edges) == {("X", "Y")}


def test_port_contract_version_only_lookup_is_safe_when_ambiguous():
    port = InMemoryNetworkXRepository()
    port.save_process_structure("v_shared", _dto("v_shared", [("A", "B")]), process_name="p1")
    port.save_process_structure("v_shared", _dto("v_shared", [("X", "Y")]), process_name="p2")

    assert port.get_process_structure("v_shared") is None


def test_port_contract_visualization_graph_filtering_and_weight_preservation():
    port = InMemoryNetworkXRepository()
    dto = _dto(
        "v1",
        [("A", "B"), ("B", "C"), ("C", "D")],
        counts={("A", "B"): 10.0, ("B", "C"): 2.0, ("C", "D"): 1.0},
    )
    port.save_process_structure("v1", dto, process_name="p1")

    graph = port.get_graph_for_visualization("p1", "v1", min_edge_frequency=2)
    assert set(graph.edges()) == {("A", "B"), ("B", "C")}
    assert int(graph["A"]["B"]["weight"]) == 10
    assert int(graph["B"]["C"]["weight"]) == 2


def test_port_contract_visualization_returns_detached_copy():
    port = InMemoryNetworkXRepository()
    port.save_process_structure("v1", _dto("v1", [("A", "B")]), process_name="p1")

    graph = port.get_graph_for_visualization("p1", "v1", min_edge_frequency=0)
    graph.add_edge("B", "C", weight=123)

    fresh = port.get_graph_for_visualization("p1", "v1", min_edge_frequency=0)
    assert ("B", "C") not in fresh.edges()


def test_port_contract_supports_camunda_bpmn_like_labels():
    port = InMemoryNetworkXRepository()
    edges = [
        ("StartEvent_1", "Task: Approve Request"),
        ("Task: Approve Request", "Gateway_XOR_1"),
        ("Gateway_XOR_1", "EndEvent_Approved"),
    ]
    port.save_process_structure("bpmn_v1", _dto("bpmn_v1", edges), process_name="camunda_proc")

    loaded = port.get_process_structure("bpmn_v1", process_name="camunda_proc")
    assert loaded is not None
    assert set(loaded.allowed_edges) == set(edges)


def test_port_contract_supports_cyclic_graph_queries_for_future_neo4j_parity():
    port = InMemoryNetworkXRepository()
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    port.save_process_structure("v_cycle", _dto("v_cycle", edges), process_name="loop_proc")

    graph = port.get_graph_for_visualization("loop_proc", "v_cycle", min_edge_frequency=0)
    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert graph.has_edge("C", "A")
