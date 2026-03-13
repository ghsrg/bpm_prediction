"""In-memory knowledge graph repository backed by NetworkX."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


class InMemoryNetworkXRepository(IKnowledgeGraphPort):
    """Store process structures in memory with optional process scoping."""

    def __init__(self) -> None:
        self._structures: Dict[Tuple[str, str], ProcessStructureDTO] = {}
        self._graphs: Dict[Tuple[str, str], nx.DiGraph] = {}
        # Version-only index for legacy consumers that do not pass process_name.
        # None value marks ambiguous version across multiple processes.
        self._version_index: Dict[str, Optional[Tuple[str, str]]] = {}

    def save_process_structure(
        self,
        version: str,
        dto: ProcessStructureDTO,
        process_name: str | None = None,
    ) -> None:
        version_key = self._normalize_version(version)
        process_key = self._normalize_process_name(process_name, version_key)
        storage_key = (process_key, version_key)

        self._structures[storage_key] = dto
        self._graphs[storage_key] = self._dto_to_graph(dto)

        if version_key not in self._version_index:
            self._version_index[version_key] = storage_key
            return

        indexed = self._version_index[version_key]
        if indexed is not None and indexed != storage_key:
            self._version_index[version_key] = None

    def get_process_structure(
        self,
        version: str,
        process_name: str | None = None,
    ) -> Optional[ProcessStructureDTO]:
        version_key = self._normalize_version(version)
        if process_name is not None:
            process_key = self._normalize_process_name(process_name, version_key)
            return self._structures.get((process_key, version_key))

        indexed = self._version_index.get(version_key)
        if indexed is None:
            return None
        return self._structures.get(indexed)

    def list_versions(self, process_name: str | None = None) -> list[str]:
        if process_name is None:
            return sorted({version for _, version in self._structures.keys()})
        process_key = self._normalize_process_name(process_name, "")
        return sorted({version for proc, version in self._structures.keys() if proc == process_key})

    def get_graph_for_visualization(
        self,
        process_name: str,
        version_key: str,
        min_edge_frequency: int = 0,
    ) -> nx.DiGraph:
        if min_edge_frequency < 0:
            raise ValueError("min_edge_frequency must be >= 0.")

        process_key = self._normalize_process_name(process_name, version_key)
        normalized_version = self._normalize_version(version_key)
        stored_graph = self._graphs.get((process_key, normalized_version))
        if stored_graph is None:
            raise ValueError(
                f"No graph found for process '{process_key}' and version '{normalized_version}'."
            )

        filtered = stored_graph.copy()
        if min_edge_frequency > 0:
            drop_edges = [
                (src, dst)
                for src, dst, attrs in filtered.edges(data=True)
                if int(attrs.get("weight", 1)) < min_edge_frequency
            ]
            filtered.remove_edges_from(drop_edges)
        return filtered

    @staticmethod
    def _normalize_process_name(process_name: str | None, fallback: str) -> str:
        raw = (process_name or fallback or "default").strip()
        return raw or "default"

    @staticmethod
    def _normalize_version(version: str) -> str:
        normalized = str(version).strip()
        return normalized or "default"

    @staticmethod
    def _dto_to_graph(dto: ProcessStructureDTO) -> nx.DiGraph:
        graph = nx.DiGraph()
        edge_stats = dto.edge_statistics or {}
        for src, dst in dto.allowed_edges:
            stats = edge_stats.get((src, dst), {})
            weight = int(stats.get("count", 1))
            graph.add_edge(str(src), str(dst), weight=max(weight, 1))
        return graph
