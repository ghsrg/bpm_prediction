"""File-backed knowledge graph repository with atomic JSON writes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


class FileBasedKnowledgeGraphRepository(IKnowledgeGraphPort):
    """Persist process structures as JSON artifacts on local filesystem."""

    SCHEMA_VERSION = "1.0"
    FILE_NAME = "process_structure.json"

    def __init__(self, base_dir: str | Path = "data/knowledge_graph") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_process_structure(
        self,
        version: str,
        dto: ProcessStructureDTO,
        process_name: str | None = None,
    ) -> None:
        version_key = self._normalize_version(version)
        process_key = self._normalize_process_name(process_name, version_key)
        target_path = self._payload_path(process_name=process_key, version=version_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "repository_backend": "file",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_name": process_key,
            "version": version_key,
            "dto": self._serialize_dto(dto),
        }
        self._atomic_write_json(target_path, payload)

    def get_process_structure(
        self,
        version: str,
        process_name: str | None = None,
    ) -> Optional[ProcessStructureDTO]:
        version_key = self._normalize_version(version)

        if process_name is not None:
            process_key = self._normalize_process_name(process_name, version_key)
            payload_path = self._payload_path(process_name=process_key, version=version_key)
            payload = self._safe_read_json(payload_path)
            if payload is None:
                return None
            return self._deserialize_dto(payload.get("dto", {}))

        candidates = self._find_payload_candidates(version=version_key)
        if len(candidates) != 1:
            return None
        payload = self._safe_read_json(candidates[0])
        if payload is None:
            return None
        return self._deserialize_dto(payload.get("dto", {}))

    def list_versions(self, process_name: str | None = None) -> list[str]:
        if process_name is None:
            versions = set()
            for process_dir in self.base_dir.iterdir():
                if not process_dir.is_dir():
                    continue
                for version_dir in process_dir.iterdir():
                    if not version_dir.is_dir():
                        continue
                    if (version_dir / self.FILE_NAME).exists():
                        versions.add(version_dir.name)
            return sorted(versions)

        process_key = self._normalize_process_name(process_name, "")
        process_dir = self.base_dir / process_key
        if not process_dir.exists():
            return []
        versions = [
            version_dir.name
            for version_dir in process_dir.iterdir()
            if version_dir.is_dir() and (version_dir / self.FILE_NAME).exists()
        ]
        return sorted(versions)

    def get_graph_for_visualization(
        self,
        process_name: str,
        version_key: str,
        min_edge_frequency: int = 0,
    ) -> nx.DiGraph:
        if min_edge_frequency < 0:
            raise ValueError("min_edge_frequency must be >= 0.")
        dto = self.get_process_structure(version=version_key, process_name=process_name)
        if dto is None:
            raise ValueError(
                f"No graph found for process '{process_name}' and version '{version_key}'."
            )
        graph = self._dto_to_graph(dto)
        if min_edge_frequency > 0:
            drop_edges = [
                (src, dst)
                for src, dst, attrs in graph.edges(data=True)
                if int(attrs.get("weight", 1)) < min_edge_frequency
            ]
            graph.remove_edges_from(drop_edges)
        return graph

    def _payload_path(self, *, process_name: str, version: str) -> Path:
        return self.base_dir / process_name / version / self.FILE_NAME

    def _find_payload_candidates(self, *, version: str) -> list[Path]:
        matches: list[Path] = []
        if not self.base_dir.exists():
            return matches
        for process_dir in self.base_dir.iterdir():
            if not process_dir.is_dir():
                continue
            candidate = process_dir / version / self.FILE_NAME
            if candidate.exists():
                matches.append(candidate)
        return sorted(matches)

    @classmethod
    def _serialize_dto(cls, dto: ProcessStructureDTO) -> Dict[str, Any]:
        edge_stats_list = []
        for (src, dst), stats in sorted((dto.edge_statistics or {}).items()):
            edge_stats_list.append(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "stats": dict(stats or {}),
                }
            )
        return {
            "version": dto.version,
            "proc_def_id": dto.proc_def_id,
            "proc_def_key": dto.proc_def_key,
            "deployment_id": dto.deployment_id,
            "allowed_edges": [[str(src), str(dst)] for src, dst in dto.allowed_edges],
            "edge_statistics": edge_stats_list,
            "node_metadata": dto.node_metadata or {},
            "nodes": dto.nodes or [],
            "edges": dto.edges or [],
            "graph_topology": dto.graph_topology or {},
            "call_bindings": dto.call_bindings or {},
            "metadata": dto.metadata or {},
        }

    @classmethod
    def _deserialize_dto(cls, payload: Dict[str, Any]) -> ProcessStructureDTO:
        allowed_edges_raw = payload.get("allowed_edges", [])
        allowed_edges: list[Tuple[str, str]] = []
        for edge in allowed_edges_raw:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            allowed_edges.append((str(edge[0]), str(edge[1])))

        edge_stats_map: Dict[Tuple[str, str], Dict[str, float]] = {}
        edge_stats_raw = payload.get("edge_statistics", [])
        if isinstance(edge_stats_raw, list):
            for item in edge_stats_raw:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("src", "")).strip()
                dst = str(item.get("dst", "")).strip()
                if not src or not dst:
                    continue
                raw_stats = item.get("stats", {})
                if not isinstance(raw_stats, dict):
                    raw_stats = {}
                normalized_stats = {}
                for key, value in raw_stats.items():
                    try:
                        normalized_stats[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
                edge_stats_map[(src, dst)] = normalized_stats

        node_metadata = payload.get("node_metadata")
        if not isinstance(node_metadata, dict):
            node_metadata = {}
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            nodes = []
        edges = payload.get("edges")
        if not isinstance(edges, list):
            edges = []
        graph_topology = payload.get("graph_topology")
        if not isinstance(graph_topology, dict):
            graph_topology = {}
        call_bindings = payload.get("call_bindings")
        if not isinstance(call_bindings, dict):
            call_bindings = {}
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return ProcessStructureDTO(
            version=str(payload.get("version", "")).strip() or "default",
            allowed_edges=allowed_edges,
            edge_statistics=edge_stats_map or None,
            node_metadata=node_metadata or None,
            proc_def_id=str(payload.get("proc_def_id", "")).strip() or None,
            proc_def_key=str(payload.get("proc_def_key", "")).strip() or None,
            deployment_id=str(payload.get("deployment_id", "")).strip() or None,
            nodes=nodes or None,
            edges=edges or None,
            graph_topology=graph_topology or None,
            call_bindings=call_bindings or None,
            metadata=metadata or None,
        )

    def _safe_read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if not path.exists():
                return None
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return None
            return payload
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(tmp_path, path)

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
        node_meta = dto.node_metadata or {}

        for node_id, meta in node_meta.items():
            attrs: Dict[str, str] = {}
            name = str(meta.get("activity_name", "")).strip()
            activity_type = str(meta.get("activity_type", "")).strip()
            if name:
                attrs["activity_name"] = name
            if activity_type:
                attrs["activity_type"] = activity_type
            graph.add_node(str(node_id), **attrs)

        for src, dst in dto.allowed_edges:
            stats = edge_stats.get((src, dst), {})
            weight = int(stats.get("count", 1))
            graph.add_edge(str(src), str(dst), weight=max(weight, 1))
        return graph
