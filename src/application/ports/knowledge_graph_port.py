"""Application port contract for process-structure knowledge access (MVP2)."""

from __future__ import annotations

from typing import Optional, Protocol

from src.domain.entities.process_structure import ProcessStructureDTO


class IKnowledgeGraphPort(Protocol):
    """Port for retrieving version-scoped normative process structure."""

    def get_process_structure(self, version: str) -> Optional[ProcessStructureDTO]:
        """Return structure DTO for a given process version, or None when absent."""
        ...

