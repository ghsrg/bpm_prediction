"""Domain DTO for version-scoped normative process topology."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ProcessStructureDTO(BaseModel):
    """Version-scoped structural process topology.

    Contract note:
    - Nodes/edges/call_bindings intentionally remain dict-based for backward compatibility.
    - Validators normalize ordering and enforce required keys to keep payload deterministic.
    """

    model_config = ConfigDict(extra="ignore")

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

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("version must be a non-empty string.")
        return normalized

    @field_validator("allowed_edges")
    @classmethod
    def _normalize_allowed_edges(
        cls, value: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        normalized: set[Tuple[str, str]] = set()
        for edge in value:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            src = str(edge[0]).strip()
            dst = str(edge[1]).strip()
            if not src or not dst:
                continue
            normalized.add((src, dst))
        return sorted(normalized)

    @field_validator("edge_statistics")
    @classmethod
    def _normalize_edge_statistics(
        cls, value: Optional[Dict[Tuple[str, str], Dict[str, float]]]
    ) -> Optional[Dict[Tuple[str, str], Dict[str, float]]]:
        if value is None:
            return None
        normalized: Dict[Tuple[str, str], Dict[str, float]] = {}
        for edge, stats in value.items():
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            src = str(edge[0]).strip()
            dst = str(edge[1]).strip()
            if not src or not dst:
                continue
            raw_stats = stats if isinstance(stats, dict) else {}
            clean_stats: Dict[str, float] = {}
            for key, stat_value in raw_stats.items():
                try:
                    clean_stats[str(key)] = float(stat_value)
                except (TypeError, ValueError):
                    continue
            normalized[(src, dst)] = clean_stats
        return normalized or None

    @field_validator("nodes")
    @classmethod
    def _normalize_nodes(
        cls, value: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if value is None:
            return None
        normalized: List[Dict[str, Any]] = []
        for idx, node in enumerate(value):
            if not isinstance(node, dict):
                raise ValueError(f"nodes[{idx}] must be a dict.")
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                raise ValueError(f"nodes[{idx}] must contain non-empty 'id'.")
            bpmn_tag = str(node.get("bpmn_tag", "")).strip() or str(
                node.get("activity_type", "")
            ).strip()
            if not bpmn_tag:
                raise ValueError(f"nodes[{idx}] must contain non-empty 'bpmn_tag'.")
            node_type = str(node.get("type", "")).strip() or bpmn_tag
            if not node_type:
                raise ValueError(f"nodes[{idx}] must contain non-empty 'type'.")
            activity_type = str(node.get("activity_type", "")).strip() or bpmn_tag

            normalized_node = dict(node)
            normalized_node["id"] = node_id
            normalized_node["bpmn_tag"] = bpmn_tag
            normalized_node["type"] = node_type
            normalized_node["activity_type"] = activity_type
            normalized.append(normalized_node)
        normalized.sort(
            key=lambda item: (
                int(item.get("scope_level", 0)),
                str(item.get("parent_scope_id", "") or ""),
                str(item.get("id", "")),
            )
        )
        return normalized

    @field_validator("edges")
    @classmethod
    def _normalize_edges(
        cls, value: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if value is None:
            return None
        normalized: List[Dict[str, Any]] = []
        for idx, edge in enumerate(value):
            if not isinstance(edge, dict):
                raise ValueError(f"edges[{idx}] must be a dict.")
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                raise ValueError(f"edges[{idx}] must contain non-empty source/target.")
            edge_type = str(edge.get("edge_type", "")).strip() or "sequence"

            normalized_edge = dict(edge)
            normalized_edge["source"] = source
            normalized_edge["target"] = target
            normalized_edge["edge_type"] = edge_type
            if "id" not in normalized_edge or not str(normalized_edge.get("id", "")).strip():
                normalized_edge["id"] = f"{edge_type}::{source}->{target}"
            try:
                normalized_edge["condition_complexity"] = float(
                    normalized_edge.get("condition_complexity", 0.0)
                )
            except (TypeError, ValueError):
                normalized_edge["condition_complexity"] = 0.0
            normalized_edge["condition_complexity"] = min(
                1.0, max(0.0, normalized_edge["condition_complexity"])
            )
            normalized_edge["is_default"] = bool(normalized_edge.get("is_default", False))
            normalized.append(normalized_edge)
        normalized.sort(
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("target", "")),
                str(item.get("edge_type", "")),
                str(item.get("id", "")),
            )
        )
        return normalized

    @field_validator("call_bindings")
    @classmethod
    def _normalize_call_bindings(
        cls, value: Optional[Dict[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        if value is None:
            return None
        normalized: Dict[str, Dict[str, Any]] = {}
        for node_id, payload in sorted(value.items(), key=lambda item: str(item[0])):
            call_node_id = str(node_id).strip()
            if not call_node_id:
                continue
            binding = dict(payload or {})
            status = str(binding.get("status", "unresolved")).strip() or "unresolved"
            fallback = str(
                binding.get("inference_fallback_strategy", "use_aggregated_stats")
            ).strip() or "use_aggregated_stats"

            normalized[call_node_id] = {
                "called_element": binding.get("called_element"),
                "binding_type": str(binding.get("binding_type", "latest")).strip() or "latest",
                "version_tag": binding.get("version_tag"),
                "version_number": binding.get("version_number"),
                "resolved_child_proc_def_id": binding.get("resolved_child_proc_def_id"),
                "status": status,
                "reason": binding.get("reason"),
                "requires_separate_inference": bool(
                    binding.get("requires_separate_inference", True)
                ),
                "inference_trigger": str(
                    binding.get("inference_trigger", "child_process_root")
                ).strip()
                or "child_process_root",
                "child_process_key": binding.get("child_process_key"),
                "child_version_tag": binding.get("child_version_tag"),
                "inference_fallback_strategy": fallback,
            }
        return normalized or None

    @model_validator(mode="after")
    def _normalize_topology(self) -> "ProcessStructureDTO":
        topology = dict(self.graph_topology or {})
        topology["allowed_edges"] = [[src, dst] for src, dst in self.allowed_edges]
        if "cycles_detected" not in topology:
            topology["cycles_detected"] = False
        if "cycle_count" not in topology:
            topology["cycle_count"] = 0
        self.graph_topology = topology
        return self
