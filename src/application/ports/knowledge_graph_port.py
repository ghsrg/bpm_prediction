"""Compatibility alias for the domain knowledge-graph port.

Deprecated import path:
    src.application.ports.knowledge_graph_port.IKnowledgeGraphPort
Preferred import path:
    src.domain.ports.knowledge_graph_port.IKnowledgeGraphPort
"""

from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort

__all__ = ["IKnowledgeGraphPort"]
