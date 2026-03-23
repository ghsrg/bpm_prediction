"""Domain port contract for knowledge-graph persistence and reads."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO


class IKnowledgeGraphPort(Protocol):
    """Port for storing and reading version-scoped process structures."""

    def save_process_structure(
        self,
        version: str,
        dto: ProcessStructureDTO,
        process_name: str | None = None,
    ) -> None:
        """Persist structure DTO for process/version key."""
        ...

    def get_process_structure(
        self,
        version: str,
        process_name: str | None = None,
    ) -> Optional[ProcessStructureDTO]:
        """Return process structure DTO for process/version key."""
        ...

    def list_versions(self, process_name: str | None = None) -> list[str]:
        """Return available versions (optionally scoped by process name)."""
        ...

    def list_process_names(self) -> list[str]:
        """Return available process namespaces in knowledge storage."""
        ...

    def save_process_structure_snapshot(
        self,
        version: str,
        dto: ProcessStructureDTO,
        process_name: str | None = None,
        *,
        as_of_ts: datetime | None = None,
        knowledge_version: str | None = None,
    ) -> str:
        """Persist immutable snapshot and return assigned knowledge_version."""
        ...

    def get_process_structure_as_of(
        self,
        version: str,
        process_name: str | None = None,
        *,
        as_of_ts: datetime | None = None,
    ) -> Optional[ProcessStructureDTO]:
        """Return latest snapshot up to as_of_ts; fallback to current structure when unavailable.

        Implementations may annotate `dto.metadata` with fallback diagnostics when strict snapshot is missing.
        """
        ...

    def get_graph_for_visualization(
        self,
        process_name: str,
        version_key: str,
        min_edge_frequency: int = 0,
    ) -> nx.DiGraph:
        """Return filtered directed graph ready for visualization."""
        ...
