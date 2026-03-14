"""Domain DTO for version-scoped normative process topology (MVP2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


class ProcessStructureDTO(BaseModel):
    """Version-specific allowed transitions extracted from train traces."""

    version: str
    allowed_edges: List[Tuple[str, str]]
    edge_statistics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None
    node_metadata: Optional[Dict[str, Dict[str, str]]] = None
