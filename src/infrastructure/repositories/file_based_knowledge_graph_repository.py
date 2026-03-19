"""File-backed knowledge graph repository with atomic JSON writes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


logger = logging.getLogger(__name__)


class FileBasedKnowledgeGraphRepository(IKnowledgeGraphPort):
    """Persist process structures as JSON artifacts on local filesystem."""

    SCHEMA_VERSION = "1.1"
    SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1"}
    FILE_NAME = "process_structure.json"
    SNAPSHOT_DIR_NAME = "snapshots"

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
            self._validate_schema_header(payload, payload_path)
            return self._deserialize_dto(payload.get("dto", {}))

        candidates = self._find_payload_candidates(version=version_key)
        if len(candidates) != 1:
            return None
        payload = self._safe_read_json(candidates[0])
        if payload is None:
            return None
        self._validate_schema_header(payload, candidates[0])
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
                    if (version_dir / self.FILE_NAME).exists() or any(
                        (version_dir / self.SNAPSHOT_DIR_NAME).glob("*.json")
                    ):
                        versions.add(version_dir.name)
            return sorted(versions)

        process_key = self._normalize_process_name(process_name, "")
        process_dir = self.base_dir / process_key
        if not process_dir.exists():
            return []
        versions = [
            version_dir.name
            for version_dir in process_dir.iterdir()
            if version_dir.is_dir()
            and (
                (version_dir / self.FILE_NAME).exists()
                or any((version_dir / self.SNAPSHOT_DIR_NAME).glob("*.json"))
            )
        ]
        return sorted(versions)

    def list_process_names(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            process_dir.name
            for process_dir in self.base_dir.iterdir()
            if process_dir.is_dir()
        )

    def save_process_structure_snapshot(
        self,
        version: str,
        dto: ProcessStructureDTO,
        process_name: str | None = None,
        *,
        as_of_ts: datetime | None = None,
        knowledge_version: str | None = None,
    ) -> str:
        version_key = self._normalize_version(version)
        process_key = self._normalize_process_name(process_name, version_key)
        dt = self._to_utc(as_of_ts) or datetime.now(timezone.utc)

        snapshot_dir = self._snapshot_dir(process_name=process_key, version=version_key)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if knowledge_version is None:
            knowledge_version = self._next_knowledge_version(snapshot_dir)
        safe_ts = dt.strftime("%Y%m%dT%H%M%SZ")
        snapshot_path = snapshot_dir / f"{knowledge_version}__{safe_ts}.json"

        dto_copy = dto.model_copy(deep=True)
        metadata = dict(dto_copy.metadata or {})
        metadata["knowledge_version"] = knowledge_version
        metadata["as_of_ts"] = dt.isoformat()
        dto_copy.metadata = metadata

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "repository_backend": "file",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_name": process_key,
            "version": version_key,
            "knowledge_version": knowledge_version,
            "as_of_ts": dt.isoformat(),
            "dto": self._serialize_dto(dto_copy),
        }
        self._atomic_write_json(snapshot_path, payload)
        # Keep latest structure file in sync for backward-compatible consumers.
        self.save_process_structure(version=version_key, dto=dto_copy, process_name=process_key)
        return knowledge_version

    def get_process_structure_as_of(
        self,
        version: str,
        process_name: str | None = None,
        *,
        as_of_ts: datetime | None = None,
    ) -> Optional[ProcessStructureDTO]:
        version_key = self._normalize_version(version)
        target_ts = self._to_utc(as_of_ts)

        if process_name is not None:
            process_key = self._normalize_process_name(process_name, version_key)
            dto = self._load_snapshot_as_of(process_name=process_key, version=version_key, target_ts=target_ts)
            if dto is not None:
                return dto
            return self.get_process_structure(version=version_key, process_name=process_key)

        candidates = self._find_snapshot_candidates(version=version_key)
        if len(candidates) != 1:
            return self.get_process_structure(version=version_key, process_name=process_name)
        process_key = candidates[0]
        dto = self._load_snapshot_as_of(process_name=process_key, version=version_key, target_ts=target_ts)
        if dto is not None:
            return dto
        return self.get_process_structure(version=version_key, process_name=process_key)

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

    def _snapshot_dir(self, *, process_name: str, version: str) -> Path:
        return self.base_dir / process_name / version / self.SNAPSHOT_DIR_NAME

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

    def _find_snapshot_candidates(self, *, version: str) -> list[str]:
        matches: list[str] = []
        if not self.base_dir.exists():
            return matches
        for process_dir in self.base_dir.iterdir():
            if not process_dir.is_dir():
                continue
            snapshot_dir = process_dir / version / self.SNAPSHOT_DIR_NAME
            if snapshot_dir.exists() and any(snapshot_dir.glob("*.json")):
                matches.append(process_dir.name)
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
            "nodes": sorted(
                (dto.nodes or []),
                key=lambda item: (
                    int(item.get("scope_level", 0)),
                    str(item.get("parent_scope_id", "") or ""),
                    str(item.get("id", "")),
                ),
            ),
            "edges": sorted(
                (dto.edges or []),
                key=lambda item: (
                    str(item.get("source", "")),
                    str(item.get("target", "")),
                    str(item.get("edge_type", "")),
                    str(item.get("id", "")),
                ),
            ),
            "graph_topology": dto.graph_topology or {},
            "call_bindings": dict(sorted((dto.call_bindings or {}).items())),
            "node_stats": dto.node_stats or {},
            "edge_stats": dto.edge_stats or {},
            "gnn_features": dto.gnn_features or {},
            "stats_diagnostics": dto.stats_diagnostics or {},
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
        node_stats = payload.get("node_stats")
        if not isinstance(node_stats, dict):
            node_stats = {}
        edge_stats = payload.get("edge_stats")
        if not isinstance(edge_stats, dict):
            edge_stats = {}
        gnn_features = payload.get("gnn_features")
        if not isinstance(gnn_features, dict):
            gnn_features = {}
        stats_diagnostics = payload.get("stats_diagnostics")
        if not isinstance(stats_diagnostics, dict):
            stats_diagnostics = {}
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
            node_stats=node_stats or None,
            edge_stats=edge_stats or None,
            gnn_features=gnn_features or None,
            stats_diagnostics=stats_diagnostics or None,
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

    def _load_snapshot_as_of(
        self,
        *,
        process_name: str,
        version: str,
        target_ts: datetime | None,
    ) -> Optional[ProcessStructureDTO]:
        snapshot_dir = self._snapshot_dir(process_name=process_name, version=version)
        if not snapshot_dir.exists():
            return None

        candidates: list[tuple[datetime, str, Path]] = []
        for path in snapshot_dir.glob("*.json"):
            payload = self._safe_read_json(path)
            if payload is None:
                continue
            as_of_text = str(payload.get("as_of_ts", "")).strip()
            as_of_dt = self._parse_datetime(as_of_text)
            if as_of_dt is None:
                continue
            kv = str(payload.get("knowledge_version", "")).strip() or "k000000"
            candidates.append((as_of_dt, kv, path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))

        if target_ts is None:
            selected_path = candidates[-1][2]
            selected_payload = self._safe_read_json(selected_path)
            if selected_payload is None:
                return None
            self._validate_schema_header(selected_payload, selected_path)
            return self._deserialize_dto(selected_payload.get("dto", {}))

        selected_path: Optional[Path] = None
        for snap_ts, _, snap_path in candidates:
            if snap_ts <= target_ts:
                selected_path = snap_path
            else:
                break
        if selected_path is None:
            return None

        selected_payload = self._safe_read_json(selected_path)
        if selected_payload is None:
            return None
        self._validate_schema_header(selected_payload, selected_path)
        return self._deserialize_dto(selected_payload.get("dto", {}))

    def _validate_schema_header(self, payload: Dict[str, Any], path: Path) -> None:
        schema_version = str(payload.get("schema_version", "")).strip()
        if not schema_version:
            logger.warning("Knowledge artifact '%s' has no schema_version header.", path)
            return
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            logger.warning(
                "Knowledge artifact '%s' schema_version=%s is outside supported set %s.",
                path,
                schema_version,
                sorted(self.SUPPORTED_SCHEMA_VERSIONS),
            )

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(tmp_path, path)

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return FileBasedKnowledgeGraphRepository._to_utc(dt)

    @staticmethod
    def _to_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _next_knowledge_version(snapshot_dir: Path) -> str:
        max_seq = 0
        for path in snapshot_dir.glob("k*.json"):
            prefix = path.stem.split("__", 1)[0]
            if not prefix.startswith("k"):
                continue
            try:
                seq = int(prefix[1:])
            except ValueError:
                continue
            max_seq = max(max_seq, seq)
        return f"k{max_seq + 1:06d}"

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
