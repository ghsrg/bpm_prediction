"""In-memory knowledge graph repository backed by NetworkX."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


class InMemoryNetworkXRepository(IKnowledgeGraphPort):
    """Store process structures in memory with optional process scoping."""

    def __init__(self) -> None:
        self._structures: Dict[Tuple[str, str], ProcessStructureDTO] = {}
        self._graphs: Dict[Tuple[str, str], nx.DiGraph] = {}
        self._snapshots: Dict[Tuple[str, str], list[Tuple[datetime, str, ProcessStructureDTO]]] = {}
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

    def list_process_names(self) -> list[str]:
        return sorted({process_name for process_name, _ in self._structures.keys()})

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
        storage_key = (process_key, version_key)

        dt = self._to_utc(as_of_ts) or datetime.now(timezone.utc)
        if knowledge_version is None:
            seq = len(self._snapshots.get(storage_key, [])) + 1
            knowledge_version = f"k{seq:06d}"

        payload_dto = dto.model_copy(deep=True)
        metadata = dict(payload_dto.metadata or {})
        metadata["knowledge_version"] = knowledge_version
        metadata["as_of_ts"] = dt.isoformat()
        payload_dto.metadata = metadata

        self._snapshots.setdefault(storage_key, []).append((dt, knowledge_version, payload_dto))
        self._snapshots[storage_key].sort(key=lambda item: (item[0], item[1]))
        self.save_process_structure(version=version_key, dto=payload_dto, process_name=process_key)
        return knowledge_version

    def get_process_structure_as_of(
        self,
        version: str,
        process_name: str | None = None,
        *,
        as_of_ts: datetime | None = None,
    ) -> Optional[ProcessStructureDTO]:
        version_key = self._normalize_version(version)
        if process_name is None:
            indexed = self._version_index.get(version_key)
            if indexed is None:
                return None
            storage_key = indexed
        else:
            process_key = self._normalize_process_name(process_name, version_key)
            storage_key = (process_key, version_key)

        target = self._to_utc(as_of_ts)
        snapshots = self._snapshots.get(storage_key, [])
        if not snapshots:
            base = self._structures.get(storage_key)
            return self._mark_missing_asof_snapshot(base, target)

        if target is None:
            return snapshots[-1][2]

        selected: Optional[ProcessStructureDTO] = None
        for snap_ts, _, snap_dto in snapshots:
            if snap_ts <= target:
                selected = snap_dto
            else:
                break
        if selected is not None:
            return selected
        base = self._structures.get(storage_key)
        return self._mark_missing_asof_snapshot(base, target)

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
    def _to_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _mark_missing_asof_snapshot(dto: Optional[ProcessStructureDTO], target: datetime | None) -> Optional[ProcessStructureDTO]:
        if dto is None:
            return None
        dto_copy = dto.model_copy(deep=True)
        metadata = dict(dto_copy.metadata or {})
        metadata["asof_snapshot_found"] = False
        metadata["asof_resolution"] = "missing_snapshot_fallback_base"
        if isinstance(target, datetime):
            metadata["asof_lookup_ts"] = target.isoformat()
        dto_copy.metadata = metadata
        return dto_copy

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
