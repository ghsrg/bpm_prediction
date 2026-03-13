from __future__ import annotations

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _dto(version: str, edges: list[tuple[str, str]], count: int = 1) -> ProcessStructureDTO:
    return ProcessStructureDTO(
        version=version,
        allowed_edges=edges,
        edge_statistics={edge: {"count": float(count)} for edge in edges},
    )


def test_repository_save_get_roundtrip():
    repository = InMemoryNetworkXRepository()
    dto = _dto("v1", [("A", "B"), ("B", "C")], count=3)
    repository.save_process_structure("v1", dto, process_name="p1")

    loaded = repository.get_process_structure("v1", process_name="p1")
    assert loaded is not None
    assert loaded.version == "v1"
    assert set(loaded.allowed_edges) == {("A", "B"), ("B", "C")}


def test_repository_list_versions_by_process():
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure("v1", _dto("v1", [("A", "B")]), process_name="p1")
    repository.save_process_structure("v2", _dto("v2", [("B", "C")]), process_name="p1")
    repository.save_process_structure("v1", _dto("v1", [("X", "Y")]), process_name="p2")

    assert repository.list_versions(process_name="p1") == ["v1", "v2"]
    assert repository.list_versions(process_name="p2") == ["v1"]


def test_repository_get_graph_for_visualization_filters_edges_by_weight():
    repository = InMemoryNetworkXRepository()
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("A", "B"), ("B", "C"), ("B", "D")],
        edge_statistics={
            ("A", "B"): {"count": 10.0},
            ("B", "C"): {"count": 1.0},
            ("B", "D"): {"count": 3.0},
        },
    )
    repository.save_process_structure("v1", dto, process_name="p1")

    graph = repository.get_graph_for_visualization("p1", "v1", min_edge_frequency=3)
    assert set(graph.edges()) == {("A", "B"), ("B", "D")}


def test_repository_version_only_lookup_returns_none_when_ambiguous():
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure("v1", _dto("v1", [("A", "B")]), process_name="p1")
    repository.save_process_structure("v1", _dto("v1", [("X", "Y")]), process_name="p2")

    assert repository.get_process_structure("v1") is None
