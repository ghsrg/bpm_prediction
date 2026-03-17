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


def _is_mvp25_stage3_context(config: Dict[str, Any]) -> bool:
    experiment_cfg = config.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}
    project = str(experiment_cfg.get("project", "")).strip().lower()
    name = str(experiment_cfg.get("name", "")).strip().lower()
    return ("mvp2_5" in project) or ("mvp2_5" in name) or ("stage3" in name)


def _expects_topology(config: Dict[str, Any]) -> bool:
    mapping_cfg = config.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    adapter = str(mapping_cfg.get("adapter", "")).strip().lower()
    model_type = str(model_cfg.get("type", model_cfg.get("model_type", ""))).strip().lower()
    has_camunda_mapping = isinstance(mapping_cfg.get("camunda_adapter"), dict)
    return (adapter == "camunda") or has_camunda_mapping or model_type.startswith("eopkg")


def _require_explicit_knowledge_backend(config: Dict[str, Any]) -> bool:
    return _is_mvp25_stage3_context(config) and _expects_topology(config)


def get_knowledge_graph_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    mapping_cfg = config.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}

    require_explicit = _require_explicit_knowledge_backend(config)
    settings_raw = mapping_cfg.get("knowledge_graph", {})
    if require_explicit and not isinstance(settings_raw, dict):
        raise ValueError(
            "Missing required mapping.knowledge_graph config for MVP2.5 Stage3 topology-aware run. "
            "Set backend explicitly ('file' or 'in_memory')."
        )

    settings = settings_raw if isinstance(settings_raw, dict) else {}
    backend_raw = str(settings.get("backend", "")).strip().lower()
    if require_explicit and not backend_raw:
        raise ValueError(
            "Missing mapping.knowledge_graph.backend for MVP2.5 Stage3 topology-aware run. "
            "Set backend explicitly ('file' or 'in_memory')."
        )

    backend = backend_raw or "in_memory"
    if backend in {"memory", "in-memory"}:
        backend = "in_memory"
    if require_explicit and backend not in {"in_memory", "file"}:
        raise ValueError(
            "Unsupported mapping.knowledge_graph.backend value for MVP2.5 Stage3 run. "
            "Use 'file' or 'in_memory'."
        )
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
