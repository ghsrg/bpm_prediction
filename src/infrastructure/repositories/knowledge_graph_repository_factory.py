"""Factory helpers for knowledge graph repository backend selection."""

from __future__ import annotations

from typing import Any, Dict

from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.infrastructure.repositories.file_based_knowledge_graph_repository import (
    FileBasedKnowledgeGraphRepository,
)
from src.infrastructure.repositories.in_memory_networkx_repository import (
    InMemoryNetworkXRepository,
)


def get_knowledge_graph_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    mapping_cfg = config.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    settings = mapping_cfg.get("knowledge_graph", {})
    if not isinstance(settings, dict):
        settings = {}
    backend = str(settings.get("backend", "in_memory")).strip().lower()
    if backend in {"memory", "in-memory"}:
        backend = "in_memory"
    settings = {
        **settings,
        "backend": backend if backend in {"in_memory", "file"} else "in_memory",
        "path": str(settings.get("path", "data/knowledge_graph")),
        "strict_load": bool(settings.get("strict_load", False)),
        "ingest_split": str(settings.get("ingest_split", "train")).strip().lower() or "train",
    }
    return settings


def build_knowledge_graph_repository(config: Dict[str, Any]) -> IKnowledgeGraphPort:
    settings = get_knowledge_graph_settings(config)
    backend = settings["backend"]
    if backend == "file":
        return FileBasedKnowledgeGraphRepository(base_dir=settings["path"])
    return InMemoryNetworkXRepository()
