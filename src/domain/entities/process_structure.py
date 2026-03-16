"""Domain DTO for version-scoped normative process topology."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class ProcessStructureDTO(BaseModel):
    """Version-scoped structural process topology."""

    version: str
    allowed_edges: List[Tuple[str, str]]
    edge_statistics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None
    node_metadata: Optional[Dict[str, Dict[str, str]]] = None
    proc_def_id: Optional[str] = None
    proc_def_key: Optional[str] = None
    deployment_id: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    graph_topology: Optional[Dict[str, Any]] = None
    call_bindings: Optional[Dict[str, Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
