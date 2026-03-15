from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)


def _dto(version: str, edges: list[tuple[str, str]], count: int = 1) -> ProcessStructureDTO:
    return ProcessStructureDTO(
        version=version,
        allowed_edges=edges,
        edge_statistics={edge: {"count": float(count)} for edge in edges},
    )


def test_file_repo_save_and_get_roundtrip_with_schema_header(tmp_path: Path):
    repo = FileBasedKnowledgeGraphRepository(base_dir=tmp_path / "kg")
    dto = _dto("v1", [("A", "B"), ("B", "C")], count=3)
    repo.save_process_structure("v1", dto, process_name="proc_a")

    loaded = repo.get_process_structure("v1", process_name="proc_a")
    assert loaded is not None
    assert loaded.version == "v1"
    assert set(loaded.allowed_edges) == {("A", "B"), ("B", "C")}

    payload_path = tmp_path / "kg" / "proc_a" / "v1" / "process_structure.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == FileBasedKnowledgeGraphRepository.SCHEMA_VERSION
    assert payload["repository_backend"] == "file"


def test_file_repo_atomic_write_does_not_leave_tmp_files(tmp_path: Path):
    repo = FileBasedKnowledgeGraphRepository(base_dir=tmp_path / "kg")
    repo.save_process_structure("v1", _dto("v1", [("A", "B")]), process_name="proc_a")
    tmp_files = list((tmp_path / "kg").rglob("*.tmp"))
    assert tmp_files == []


def test_file_repo_version_only_lookup_returns_none_when_ambiguous(tmp_path: Path):
    repo = FileBasedKnowledgeGraphRepository(base_dir=tmp_path / "kg")
    repo.save_process_structure("v1", _dto("v1", [("A", "B")]), process_name="proc_a")
    repo.save_process_structure("v1", _dto("v1", [("X", "Y")]), process_name="proc_b")
    assert repo.get_process_structure("v1") is None


def test_file_repo_get_graph_for_visualization_filters_by_frequency(tmp_path: Path):
    repo = FileBasedKnowledgeGraphRepository(base_dir=tmp_path / "kg")
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("A", "B"), ("B", "C"), ("C", "D")],
        edge_statistics={
            ("A", "B"): {"count": 10.0},
            ("B", "C"): {"count": 2.0},
            ("C", "D"): {"count": 1.0},
        },
    )
    repo.save_process_structure("v1", dto, process_name="proc_a")
    graph = repo.get_graph_for_visualization("proc_a", "v1", min_edge_frequency=2)
    assert set(graph.edges()) == {("A", "B"), ("B", "C")}


def test_file_repo_corrupted_json_returns_none(tmp_path: Path):
    repo = FileBasedKnowledgeGraphRepository(base_dir=tmp_path / "kg")
    payload_path = tmp_path / "kg" / "proc_a" / "v1" / "process_structure.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text("{this-is-not-valid-json", encoding="utf-8")
    assert repo.get_process_structure("v1", process_name="proc_a") is None
