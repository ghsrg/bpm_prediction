"""Neo4j-backed knowledge graph repository (APOC-free Cypher)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


class Neo4jKnowledgeGraphRepository(IKnowledgeGraphPort):
    """Persist and read process structures via Neo4j using pure Cypher only."""

    _RESERVED_REL_TYPES = {"HAS_VERSION", "CONTAINS_NODE"}
    _LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_]")

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        verify_connectivity: bool = True,
        driver: Any | None = None,
    ) -> None:
        if driver is not None:
            self._driver = driver
        else:
            try:
                from neo4j import GraphDatabase  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Neo4j backend requires 'neo4j' package. Install dependency or use "
                    "mapping.knowledge_graph.backend=file/in_memory."
                ) from exc
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            if verify_connectivity:
                self._driver.verify_connectivity()
        self._database = str(database).strip() or "neo4j"

    def close(self) -> None:
        if hasattr(self._driver, "close"):
            self._driver.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup only
        try:
            self.close()
        except Exception:
            pass

    def save_process_structure(
        self,
        version: str,
        dto: ProcessStructureDTO,
        process_name: str | None = None,
    ) -> None:
        version_key = self._normalize_version(version)
        process_key = self._normalize_process_name(process_name, version_key)
        dto_payload = dto.model_copy(deep=True)
        metadata = dict(dto_payload.metadata or {})
        if dto_payload.node_stats is not None:
            metadata["node_stats"] = dto_payload.node_stats
        if dto_payload.edge_stats is not None:
            metadata["edge_stats"] = dto_payload.edge_stats
        if dto_payload.gnn_features is not None:
            metadata["gnn_features"] = dto_payload.gnn_features
        if dto_payload.stats_diagnostics is not None:
            metadata["stats_diagnostics"] = dto_payload.stats_diagnostics
        dto_payload.metadata = metadata or None

        self._upsert_process_version(process_name=process_key, version_key=version_key, dto=dto_payload)
        for node in dto_payload.nodes or []:
            self._upsert_node(
                process_name=process_key,
                version_key=version_key,
                node=node,
                node_metadata=(dto_payload.node_metadata or {}).get(str(node.get("id", "")), {}),
            )
        for edge in dto_payload.edges or self._edges_from_allowed(dto_payload.allowed_edges):
            self._upsert_edge(
                process_name=process_key,
                version_key=version_key,
                edge=edge,
                edge_stats=(dto_payload.edge_statistics or {}).get(
                    (str(edge.get("source", "")), str(edge.get("target", ""))),
                    {},
                ),
            )

    def get_process_structure(
        self,
        version: str,
        process_name: str | None = None,
    ) -> Optional[ProcessStructureDTO]:
        version_key = self._normalize_version(version)
        resolved_process_name = self._resolve_process_name_for_version(
            version_key=version_key,
            process_name=process_name,
        )
        if resolved_process_name is None:
            return None

        meta_rows = self._run_read(
            operation="load_process_version",
            query="""
            MATCH (pv:ProcessVersion {process_name: $process_name, version_key: $version_key})
            RETURN
              pv.proc_def_id AS proc_def_id,
              pv.proc_def_key AS proc_def_key,
              pv.deployment_id AS deployment_id,
              pv.call_bindings_json AS call_bindings_json,
              pv.graph_topology_json AS graph_topology_json,
              pv.metadata_json AS metadata_json
            LIMIT 1
            """,
            params={"process_name": resolved_process_name, "version_key": version_key},
        )
        if not meta_rows:
            return None
        meta = meta_rows[0]

        node_rows = self._run_read(
            operation="load_nodes_for_version",
            query="""
            MATCH (:ProcessVersion {process_name: $process_name, version_key: $version_key})
                  -[:CONTAINS_NODE]->
                  (n:ProcessNode {process_name: $process_name})
            WHERE $version_key IN coalesce(n.versions, [])
            RETURN
              n.node_id AS node_id,
              n.name AS name,
              n.bpmn_tag AS bpmn_tag,
              n.camunda_type AS camunda_type,
              n.logical_type AS logical_type,
              n.activity_type AS activity_type,
              n.scope_level AS scope_level,
              n.parent_scope_id AS parent_scope_id,
              n.is_event_subprocess AS is_event_subprocess,
              n.is_multi_instance AS is_multi_instance,
              n.is_sequential_mi AS is_sequential_mi,
              n.loop_cardinality_expr AS loop_cardinality_expr,
              n.attached_to AS attached_to,
              n.extensions_json AS extensions_json,
              n.extra_json AS extra_json
            ORDER BY coalesce(n.scope_level, 0), coalesce(n.parent_scope_id, ''), n.node_id
            """,
            params={"process_name": resolved_process_name, "version_key": version_key},
        )

        edge_rows = self._run_read(
            operation="load_edges_for_version",
            query="""
            MATCH (src:ProcessNode {process_name: $process_name})-[r]->(dst:ProcessNode {process_name: $process_name})
            WHERE NOT type(r) IN $reserved_types
              AND $version_key IN coalesce(r.versions, [])
            RETURN
              src.node_id AS source_id,
              dst.node_id AS target_id,
              coalesce(r.edge_id, r.edge_key) AS edge_id,
              coalesce(r.edge_type, toLower(type(r))) AS edge_type,
              coalesce(r.is_default, false) AS is_default,
              coalesce(r.condition_expr, '') AS condition_expr,
              coalesce(r.condition_complexity, 0.0) AS condition_complexity,
              coalesce(r.scope_level, 0) AS scope_level,
              coalesce(r.is_interrupting, false) AS is_interrupting,
              coalesce(r.frequency_count, 1.0) AS frequency_count,
              r.stats_json AS stats_json,
              r.extra_json AS extra_json
            ORDER BY source_id, target_id, edge_type, edge_id
            """,
            params={
                "process_name": resolved_process_name,
                "version_key": version_key,
                "reserved_types": sorted(self._RESERVED_REL_TYPES),
            },
        )

        nodes: List[Dict[str, Any]] = []
        node_metadata: Dict[str, Dict[str, str]] = {}
        for row in node_rows:
            node_id = str(row.get("node_id", "")).strip()
            if not node_id:
                continue
            extensions = self._safe_json_object(row.get("extensions_json"))
            extra = self._safe_json_object(row.get("extra_json"))
            node = {
                "id": node_id,
                "name": str(row.get("name", "") or "").strip(),
                "bpmn_tag": str(row.get("bpmn_tag", "") or "").strip(),
                "camunda_type": str(row.get("camunda_type", "") or "").strip() or None,
                "type": str(row.get("logical_type", "") or "").strip() or None,
                "activity_type": str(row.get("activity_type", "") or "").strip() or None,
                "scope_level": int(row.get("scope_level") or 0),
                "parent_scope_id": str(row.get("parent_scope_id", "") or "").strip() or None,
                "is_event_subprocess": bool(row.get("is_event_subprocess", False)),
                "is_multi_instance": bool(row.get("is_multi_instance", False)),
                "is_sequential_mi": bool(row.get("is_sequential_mi", False)),
                "loop_cardinality_expr": str(row.get("loop_cardinality_expr", "") or "").strip() or None,
                "attached_to": str(row.get("attached_to", "") or "").strip() or None,
                "extensions": extensions,
                **extra,
            }
            nodes.append(node)
            node_metadata[node_id] = {
                "activity_name": str(node.get("name", "") or "").strip() or node_id,
                "activity_type": str(node.get("activity_type", "") or "").strip(),
            }

        edges: List[Dict[str, Any]] = []
        allowed_edges: List[tuple[str, str]] = []
        edge_statistics: Dict[tuple[str, str], Dict[str, float]] = {}
        for row in edge_rows:
            source = str(row.get("source_id", "")).strip()
            target = str(row.get("target_id", "")).strip()
            if not source or not target:
                continue
            edge_type = str(row.get("edge_type", "")).strip() or "sequence"
            edge = {
                "id": str(row.get("edge_id", "")).strip() or f"{edge_type}::{source}->{target}",
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "is_default": bool(row.get("is_default", False)),
                "condition_expr": str(row.get("condition_expr", "") or "").strip() or None,
                "condition_complexity": float(row.get("condition_complexity", 0.0) or 0.0),
                "scope_level": int(row.get("scope_level") or 0),
                "is_interrupting": bool(row.get("is_interrupting", False)),
                **self._safe_json_object(row.get("extra_json")),
            }
            edges.append(edge)
            allowed_edges.append((source, target))
            edge_statistics[(source, target)] = {"count": float(row.get("frequency_count", 1.0) or 1.0)}

        metadata_obj = self._safe_json_object(meta.get("metadata_json")) or None
        node_stats = metadata_obj.get("node_stats") if isinstance(metadata_obj, dict) else None
        edge_stats = metadata_obj.get("edge_stats") if isinstance(metadata_obj, dict) else None
        gnn_features = metadata_obj.get("gnn_features") if isinstance(metadata_obj, dict) else None
        stats_diagnostics = metadata_obj.get("stats_diagnostics") if isinstance(metadata_obj, dict) else None

        return ProcessStructureDTO(
            version=version_key,
            allowed_edges=allowed_edges,
            edge_statistics=edge_statistics or None,
            node_metadata=node_metadata or None,
            proc_def_id=self._optional_text(meta.get("proc_def_id")),
            proc_def_key=self._optional_text(meta.get("proc_def_key")),
            deployment_id=self._optional_text(meta.get("deployment_id")),
            nodes=nodes or None,
            edges=edges or None,
            graph_topology=self._safe_json_object(meta.get("graph_topology_json")) or None,
            call_bindings=self._safe_json_object(meta.get("call_bindings_json")) or None,
            node_stats=node_stats if isinstance(node_stats, dict) else None,
            edge_stats=edge_stats if isinstance(edge_stats, dict) else None,
            gnn_features=gnn_features if isinstance(gnn_features, dict) else None,
            stats_diagnostics=stats_diagnostics if isinstance(stats_diagnostics, dict) else None,
            metadata=metadata_obj,
        )

    def list_versions(self, process_name: str | None = None) -> list[str]:
        if process_name is not None:
            process_key = self._normalize_process_name(process_name, "")
            rows = self._run_read(
                operation="list_versions_for_process",
                query="""
                MATCH (pv:ProcessVersion {process_name: $process_name})
                RETURN pv.version_key AS version_key
                ORDER BY version_key
                """,
                params={"process_name": process_key},
            )
            return sorted(
                {str(row.get("version_key", "")).strip() for row in rows if str(row.get("version_key", "")).strip()}
            )

        rows = self._run_read(
            operation="list_versions_global",
            query="""
            MATCH (pv:ProcessVersion)
            RETURN DISTINCT pv.version_key AS version_key
            ORDER BY version_key
            """,
            params={},
        )
        return sorted(
            {str(row.get("version_key", "")).strip() for row in rows if str(row.get("version_key", "")).strip()}
        )

    def list_process_names(self) -> list[str]:
        rows = self._run_read(
            operation="list_process_names",
            query="""
            MATCH (p:Process)
            RETURN DISTINCT p.process_name AS process_name
            ORDER BY process_name
            """,
            params={},
        )
        return sorted(
            {
                str(row.get("process_name", "")).strip()
                for row in rows
                if str(row.get("process_name", "")).strip()
            }
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
        tenant_id = self._resolve_tenant_id(process_name=process_key, dto=dto)
        proc_def_id = self._resolve_proc_def_id(
            process_name=process_key,
            version_key=version_key,
            dto=dto,
        )
        as_of_utc = self._to_utc(as_of_ts) or datetime.now(timezone.utc)

        # Keep latest structure up-to-date for backward-compatible readers.
        self.save_process_structure(version=version_key, dto=dto, process_name=process_key)

        if knowledge_version is None:
            rows = self._run_read(
                operation="resolve_next_stats_kv_seq",
                query="""
                MATCH (ss:StatsSnapshot {
                  process_name: $process_name,
                  tenant_id: $tenant_id,
                  version_key: $version_key,
                  proc_def_id: $proc_def_id
                })
                RETURN coalesce(max(ss.kv_seq), 0) AS max_seq
                """,
                params={
                    "process_name": process_key,
                    "tenant_id": tenant_id,
                    "version_key": version_key,
                    "proc_def_id": proc_def_id,
                },
            )
            max_seq = int(rows[0].get("max_seq", 0) or 0) if rows else 0
            seq = max_seq + 1
            knowledge_version = f"k{seq:06d}"
        else:
            text = str(knowledge_version).strip()
            knowledge_version = text or "k000001"
            seq = self._knowledge_version_seq(knowledge_version)

        stats_payload = {
            "node_stats_json": self._to_json(dto.node_stats),
            "edge_stats_json": self._to_json(dto.edge_stats),
            "gnn_features_json": self._to_json(dto.gnn_features),
            "stats_diagnostics_json": self._to_json(dto.stats_diagnostics),
            "metadata_json": self._to_json(dto.metadata),
        }
        params = {
            "process_name": process_key,
            "tenant_id": tenant_id,
            "version_key": version_key,
            "proc_def_id": proc_def_id,
            "knowledge_version": knowledge_version,
            "kv_seq": int(seq),
            "as_of_iso": as_of_utc.isoformat(),
            **stats_payload,
        }
        self._run_write(
            operation="upsert_stats_snapshot",
            query="""
            MERGE (p:Process {process_name: $process_name})
            SET p.tenant_id = $tenant_id
            MERGE (pv:ProcessVersion {process_name: $process_name, version_key: $version_key})
            SET pv.proc_def_id = coalesce(pv.proc_def_id, $proc_def_id),
                pv.tenant_id = $tenant_id
            MERGE (p)-[:HAS_VERSION]->(pv)
            MERGE (ss:StatsSnapshot {
              process_name: $process_name,
              tenant_id: $tenant_id,
              version_key: $version_key,
              proc_def_id: $proc_def_id,
              knowledge_version: $knowledge_version
            })
            SET ss.kv_seq = $kv_seq,
                ss.as_of_ts = datetime($as_of_iso),
                ss.created_at_utc = datetime(),
                ss.node_stats_json = $node_stats_json,
                ss.edge_stats_json = $edge_stats_json,
                ss.gnn_features_json = $gnn_features_json,
                ss.stats_diagnostics_json = $stats_diagnostics_json,
                ss.metadata_json = $metadata_json
            MERGE (pv)-[:HAS_STATS_SNAPSHOT]->(ss)
            """,
            params=params,
        )
        return knowledge_version

    def get_process_structure_as_of(
        self,
        version: str,
        process_name: str | None = None,
        *,
        as_of_ts: datetime | None = None,
    ) -> Optional[ProcessStructureDTO]:
        version_key = self._normalize_version(version)
        resolved_process_name = self._resolve_process_name_for_version(
            version_key=version_key,
            process_name=process_name,
        )
        if resolved_process_name is None:
            return None
        base = self.get_process_structure(version=version_key, process_name=resolved_process_name)
        if base is None:
            return None

        tenant_id = self._resolve_tenant_id(process_name=resolved_process_name, dto=base)
        proc_def_id = self._resolve_proc_def_id(
            process_name=resolved_process_name,
            version_key=version_key,
            dto=base,
        )
        as_of_utc = self._to_utc(as_of_ts)

        if as_of_utc is not None:
            rows = self._run_read(
                operation="load_stats_snapshot_as_of",
                query="""
                MATCH (ss:StatsSnapshot {
                  process_name: $process_name,
                  tenant_id: $tenant_id,
                  version_key: $version_key,
                  proc_def_id: $proc_def_id
                })
                WHERE ss.as_of_ts <= datetime($as_of_iso)
                RETURN
                  ss.knowledge_version AS knowledge_version,
                  toString(ss.as_of_ts) AS as_of_ts,
                  ss.node_stats_json AS node_stats_json,
                  ss.edge_stats_json AS edge_stats_json,
                  ss.gnn_features_json AS gnn_features_json,
                  ss.stats_diagnostics_json AS stats_diagnostics_json,
                  ss.metadata_json AS metadata_json
                ORDER BY ss.as_of_ts DESC, ss.kv_seq DESC
                LIMIT 1
                """,
                params={
                    "process_name": resolved_process_name,
                    "tenant_id": tenant_id,
                    "version_key": version_key,
                    "proc_def_id": proc_def_id,
                    "as_of_iso": as_of_utc.isoformat(),
                },
            )
        else:
            rows = self._run_read(
                operation="load_latest_stats_snapshot",
                query="""
                MATCH (ss:StatsSnapshot {
                  process_name: $process_name,
                  tenant_id: $tenant_id,
                  version_key: $version_key,
                  proc_def_id: $proc_def_id
                })
                RETURN
                  ss.knowledge_version AS knowledge_version,
                  toString(ss.as_of_ts) AS as_of_ts,
                  ss.node_stats_json AS node_stats_json,
                  ss.edge_stats_json AS edge_stats_json,
                  ss.gnn_features_json AS gnn_features_json,
                  ss.stats_diagnostics_json AS stats_diagnostics_json,
                  ss.metadata_json AS metadata_json
                ORDER BY ss.as_of_ts DESC, ss.kv_seq DESC
                LIMIT 1
                """,
                params={
                    "process_name": resolved_process_name,
                    "tenant_id": tenant_id,
                    "version_key": version_key,
                    "proc_def_id": proc_def_id,
                },
            )
        if not rows:
            return base

        row = rows[0]
        base.node_stats = self._safe_json_object(row.get("node_stats_json")) or None
        base.edge_stats = self._safe_json_object(row.get("edge_stats_json")) or None
        base.gnn_features = self._safe_json_object(row.get("gnn_features_json")) or None
        base.stats_diagnostics = self._safe_json_object(row.get("stats_diagnostics_json")) or None

        snapshot_meta = self._safe_json_object(row.get("metadata_json"))
        merged_meta = dict(base.metadata or {})
        merged_meta.update(snapshot_meta)
        knowledge_version = self._optional_text(row.get("knowledge_version"))
        if knowledge_version is not None:
            merged_meta["knowledge_version"] = knowledge_version
        as_of_text = self._optional_text(row.get("as_of_ts"))
        if as_of_text is not None:
            merged_meta["as_of_ts"] = as_of_text
        base.metadata = merged_meta or None
        return base

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

    def _upsert_process_version(
        self,
        *,
        process_name: str,
        version_key: str,
        dto: ProcessStructureDTO,
    ) -> None:
        tenant_id = self._resolve_tenant_id(process_name=process_name, dto=dto)
        params = {
            "process_name": process_name,
            "tenant_id": tenant_id,
            "version_key": version_key,
            "proc_def_id": dto.proc_def_id,
            "proc_def_key": dto.proc_def_key,
            "deployment_id": dto.deployment_id,
            "call_bindings_json": self._to_json(dto.call_bindings),
            "graph_topology_json": self._to_json(dto.graph_topology),
            "metadata_json": self._to_json(dto.metadata),
        }
        self._run_write(
            operation="upsert_process_version",
            query="""
            MERGE (p:Process {process_name: $process_name})
            SET p.tenant_id = $tenant_id
            MERGE (pv:ProcessVersion {process_name: $process_name, version_key: $version_key})
            MERGE (p)-[:HAS_VERSION]->(pv)
            SET
              pv.tenant_id = $tenant_id,
              pv.proc_def_id = $proc_def_id,
              pv.proc_def_key = $proc_def_key,
              pv.deployment_id = $deployment_id,
              pv.call_bindings_json = $call_bindings_json,
              pv.graph_topology_json = $graph_topology_json,
              pv.metadata_json = $metadata_json,
              pv.updated_at_utc = datetime()
            """,
            params=params,
        )

    def _upsert_node(
        self,
        *,
        process_name: str,
        version_key: str,
        node: Dict[str, Any],
        node_metadata: Dict[str, str],
    ) -> None:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            return
        labels = self._resolve_node_labels(node)
        labels_clause = "".join(f":{label}" for label in labels if label != "ProcessNode")

        logical_type = self._optional_text(node.get("type")) or self._optional_text(node.get("bpmn_tag")) or "Other"
        activity_type = self._optional_text(node.get("activity_type")) or self._optional_text(node.get("bpmn_tag"))
        name = self._optional_text(node.get("name")) or self._optional_text(node_metadata.get("activity_name")) or node_id
        bpmn_tag = self._optional_text(node.get("bpmn_tag")) or self._optional_text(node.get("activity_type")) or "unknown"

        params = {
            "process_name": process_name,
            "version_key": version_key,
            "node_id": node_id,
            "name": name,
            "bpmn_tag": bpmn_tag,
            "camunda_type": self._optional_text(node.get("camunda_type")),
            "logical_type": logical_type,
            "activity_type": activity_type,
            "scope_level": int(node.get("scope_level", 0) or 0),
            "parent_scope_id": self._optional_text(node.get("parent_scope_id")),
            "is_event_subprocess": bool(node.get("is_event_subprocess", False)),
            "is_multi_instance": bool(node.get("is_multi_instance", False)),
            "is_sequential_mi": bool(node.get("is_sequential_mi", False)),
            "loop_cardinality_expr": self._optional_text(node.get("loop_cardinality_expr")),
            "attached_to": self._optional_text(node.get("attached_to")),
            "extensions_json": self._to_json(node.get("extensions", {})),
            "extra_json": self._to_json(self._node_extra_payload(node)),
        }
        self._run_write(
            operation="upsert_node",
            query=f"""
            MATCH (pv:ProcessVersion {{process_name: $process_name, version_key: $version_key}})
            MERGE (n:ProcessNode {{process_name: $process_name, node_id: $node_id}})
            SET n{labels_clause}
            SET
              n.name = $name,
              n.bpmn_tag = $bpmn_tag,
              n.camunda_type = $camunda_type,
              n.logical_type = $logical_type,
              n.activity_type = $activity_type,
              n.scope_level = $scope_level,
              n.parent_scope_id = $parent_scope_id,
              n.is_event_subprocess = $is_event_subprocess,
              n.is_multi_instance = $is_multi_instance,
              n.is_sequential_mi = $is_sequential_mi,
              n.loop_cardinality_expr = $loop_cardinality_expr,
              n.attached_to = $attached_to,
              n.extensions_json = $extensions_json,
              n.extra_json = $extra_json
            SET n.versions = CASE
              WHEN n.versions IS NULL THEN [$version_key]
              WHEN $version_key IN n.versions THEN n.versions
              ELSE n.versions + $version_key
            END
            MERGE (pv)-[cn:CONTAINS_NODE {{process_name: $process_name, node_id: $node_id}}]->(n)
            SET cn.versions = CASE
              WHEN cn.versions IS NULL THEN [$version_key]
              WHEN $version_key IN cn.versions THEN cn.versions
              ELSE cn.versions + $version_key
            END
            """,
            params=params,
        )

    def _upsert_edge(
        self,
        *,
        process_name: str,
        version_key: str,
        edge: Dict[str, Any],
        edge_stats: Dict[str, float],
    ) -> None:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            return
        edge_type = str(edge.get("edge_type", "sequence")).strip() or "sequence"
        rel_type = self._resolve_relationship_type(edge_type)
        edge_id = str(edge.get("id", "")).strip() or f"{edge_type}::{source}->{target}"
        edge_key = f"{rel_type}|{edge_id}|{source}|{target}"

        params = {
            "process_name": process_name,
            "version_key": version_key,
            "source_id": source,
            "target_id": target,
            "edge_key": edge_key,
            "edge_id": edge_id,
            "edge_type": edge_type,
            "is_default": bool(edge.get("is_default", False)),
            "condition_expr": self._optional_text(edge.get("condition_expr")),
            "condition_complexity": float(edge.get("condition_complexity", 0.0) or 0.0),
            "scope_level": int(edge.get("scope_level", 0) or 0),
            "is_interrupting": bool(edge.get("is_interrupting", False)),
            "frequency_count": float(edge_stats.get("count", 1.0) or 1.0),
            "stats_json": self._to_json(edge_stats),
            "extra_json": self._to_json(self._edge_extra_payload(edge)),
        }
        self._run_write(
            operation="upsert_edge",
            query=f"""
            MATCH (:ProcessVersion {{process_name: $process_name, version_key: $version_key}})
            MATCH (src:ProcessNode {{process_name: $process_name, node_id: $source_id}})
            MATCH (dst:ProcessNode {{process_name: $process_name, node_id: $target_id}})
            MERGE (src)-[r:{rel_type} {{process_name: $process_name, edge_key: $edge_key}}]->(dst)
            SET
              r.edge_id = $edge_id,
              r.edge_type = $edge_type,
              r.is_default = $is_default,
              r.condition_expr = $condition_expr,
              r.condition_complexity = $condition_complexity,
              r.scope_level = $scope_level,
              r.is_interrupting = $is_interrupting,
              r.frequency_count = $frequency_count,
              r.stats_json = $stats_json,
              r.extra_json = $extra_json
            SET r.versions = CASE
              WHEN r.versions IS NULL THEN [$version_key]
              WHEN $version_key IN r.versions THEN r.versions
              ELSE r.versions + $version_key
            END
            """,
            params=params,
        )

    def _resolve_process_name_for_version(
        self,
        *,
        version_key: str,
        process_name: str | None,
    ) -> Optional[str]:
        if process_name is not None:
            return self._normalize_process_name(process_name, version_key)
        rows = self._run_read(
            operation="resolve_process_for_version",
            query="""
            MATCH (pv:ProcessVersion {version_key: $version_key})
            RETURN DISTINCT pv.process_name AS process_name
            ORDER BY process_name
            LIMIT 2
            """,
            params={"version_key": version_key},
        )
        names = sorted(
            {str(row.get("process_name", "")).strip() for row in rows if str(row.get("process_name", "")).strip()}
        )
        if len(names) != 1:
            return None
        return names[0]

    def _resolve_proc_def_id(
        self,
        *,
        process_name: str,
        version_key: str,
        dto: ProcessStructureDTO,
    ) -> str:
        text = self._optional_text(dto.proc_def_id)
        if text is not None:
            return text
        rows = self._run_read(
            operation="resolve_proc_def_id_for_version",
            query="""
            MATCH (pv:ProcessVersion {process_name: $process_name, version_key: $version_key})
            RETURN pv.proc_def_id AS proc_def_id
            LIMIT 1
            """,
            params={"process_name": process_name, "version_key": version_key},
        )
        if rows:
            resolved = self._optional_text(rows[0].get("proc_def_id"))
            if resolved is not None:
                return resolved
        return "__unknown_proc_def__"

    def _run_write(self, *, operation: str, query: str, params: Dict[str, Any]) -> None:
        del operation
        with self._driver.session(database=self._database) as session:
            session.run(query, params)

    def _run_read(self, *, operation: str, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        del operation
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            return [dict(row) for row in result]

    @classmethod
    def _resolve_node_labels(cls, node: Dict[str, Any]) -> List[str]:
        raw = (
            cls._optional_text(node.get("bpmn_tag"))
            or cls._optional_text(node.get("activity_type"))
            or cls._optional_text(node.get("type"))
            or "other"
        ).lower()
        labels = ["ProcessNode"]
        if "start" in raw and "event" in raw:
            labels.append("StartEvent")
        elif "end" in raw and "event" in raw:
            labels.append("EndEvent")
        elif "gateway" in raw:
            labels.append("Gateway")
        elif "callactivity" in raw:
            labels.append("CallActivity")
        elif "usertask" in raw or ("user" in raw and "task" in raw):
            labels.extend(["Task", "UserTask"])
        elif "servicetask" in raw or ("service" in raw and "task" in raw):
            labels.extend(["Task", "ServiceTask"])
        elif "task" in raw:
            labels.append("Task")
        elif "event" in raw:
            labels.append("Event")
        else:
            labels.append("Other")
        return [cls._sanitize_label(label) for label in dict.fromkeys(labels)]

    @classmethod
    def _resolve_relationship_type(cls, edge_type: str) -> str:
        normalized = cls._LABEL_PATTERN.sub("_", str(edge_type).strip().upper())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            return "SEQUENCE_FLOW"
        if normalized[0].isdigit():
            return f"E_{normalized}"
        return normalized

    @classmethod
    def _sanitize_label(cls, label: str) -> str:
        normalized = cls._LABEL_PATTERN.sub("_", str(label).strip())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            return "Other"
        if normalized[0].isdigit():
            return f"L_{normalized}"
        return normalized

    @staticmethod
    def _normalize_process_name(process_name: str | None, fallback: str) -> str:
        raw = (process_name or fallback or "default").strip()
        return raw or "default"

    @staticmethod
    def _normalize_version(version: str) -> str:
        normalized = str(version).strip()
        return normalized or "default"

    @staticmethod
    def _edges_from_allowed(allowed_edges: Iterable[tuple[str, str]]) -> List[Dict[str, str]]:
        edges: List[Dict[str, str]] = []
        for src, dst in allowed_edges:
            s = str(src).strip()
            d = str(dst).strip()
            if not s or not d:
                continue
            edges.append(
                {
                    "id": f"sequence::{s}->{d}",
                    "source": s,
                    "target": d,
                    "edge_type": "sequence",
                }
            )
        return edges

    @staticmethod
    def _node_extra_payload(node: Dict[str, Any]) -> Dict[str, Any]:
        known = {
            "id",
            "name",
            "bpmn_tag",
            "camunda_type",
            "type",
            "activity_type",
            "scope_level",
            "parent_scope_id",
            "is_event_subprocess",
            "is_multi_instance",
            "is_sequential_mi",
            "loop_cardinality_expr",
            "attached_to",
            "extensions",
        }
        return {str(k): v for k, v in node.items() if k not in known}

    @staticmethod
    def _edge_extra_payload(edge: Dict[str, Any]) -> Dict[str, Any]:
        known = {
            "id",
            "source",
            "target",
            "edge_type",
            "is_default",
            "condition_expr",
            "condition_complexity",
            "scope_level",
            "is_interrupting",
        }
        return {str(k): v for k, v in edge.items() if k not in known}

    @staticmethod
    def _optional_text(value: Any) -> Optional[str]:
        text = str(value).strip() if value is not None else ""
        return text or None

    @staticmethod
    def _to_json(value: Any) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _safe_json_object(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if value is None:
            return {}
        text = str(value).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
        return {}

    @staticmethod
    def _resolve_tenant_id(*, process_name: str, dto: ProcessStructureDTO) -> str:
        if "@" in process_name:
            _, tenant = process_name.rsplit("@", 1)
            tenant_text = tenant.strip()
            if tenant_text:
                return tenant_text
        if isinstance(dto.metadata, dict):
            tenant_meta = dto.metadata.get("tenant_id")
            if tenant_meta is not None:
                text = str(tenant_meta).strip()
                if text:
                    return text
        return "default"

    @staticmethod
    def _knowledge_version_seq(knowledge_version: str) -> int:
        text = str(knowledge_version).strip()
        if not text.startswith("k"):
            return 0
        digits = text[1:]
        if digits.isdigit():
            return int(digits)
        return 0

    @staticmethod
    def _to_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _dto_to_graph(dto: ProcessStructureDTO) -> nx.DiGraph:
        graph = nx.DiGraph()
        node_meta = dto.node_metadata or {}
        for node in dto.nodes or []:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            attrs: Dict[str, str] = {}
            name = str(node.get("name", "")).strip() or str(
                node_meta.get(node_id, {}).get("activity_name", "")
            ).strip()
            activity_type = str(node.get("activity_type", "")).strip() or str(
                node_meta.get(node_id, {}).get("activity_type", "")
            ).strip()
            if name:
                attrs["activity_name"] = name
            if activity_type:
                attrs["activity_type"] = activity_type
            graph.add_node(node_id, **attrs)

        edge_stats = dto.edge_statistics or {}
        for src, dst in dto.allowed_edges:
            stats = edge_stats.get((src, dst), {})
            weight = int(stats.get("count", 1))
            graph.add_edge(str(src), str(dst), weight=max(weight, 1))
        return graph
