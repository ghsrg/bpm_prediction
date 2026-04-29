"""Topology projection and class-space alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.domain.entities.process_structure import ProcessStructureDTO


VALID_GATEWAY_MODES = {"preserve", "collapse_for_prediction"}


@dataclass(frozen=True)
class TopologyProjectionDiagnostics:
    gateway_mode: str
    node_count: int
    edge_count: int
    prediction_node_count: int
    transparent_node_count: int
    projected_edge_count: int
    source_path_count: int
    skipped_projected_edges: list[dict[str, str]] = field(default_factory=list)
    missing_vocab_nodes: list[str] = field(default_factory=list)
    duplicate_activity_labels: dict[str, list[str]] = field(default_factory=dict)
    missing_node_metadata: bool = False
    struct_x_rows: int | None = None
    num_classes: int = 0
    structural_edge_index_max: int | None = None
    is_aligned: bool = True
    failure_reasons: list[str] = field(default_factory=list)

    def with_structural_payload(
        self,
        *,
        struct_x_rows: int | None,
        structural_edge_index_max: int | None,
    ) -> "TopologyProjectionDiagnostics":
        reasons = list(self.failure_reasons)
        is_aligned = bool(self.is_aligned)
        if structural_edge_index_max is not None and structural_edge_index_max >= int(self.num_classes):
            if "structural_edge_index_out_of_bounds" not in reasons:
                reasons.append("structural_edge_index_out_of_bounds")
            is_aligned = False
        if struct_x_rows is not None and int(struct_x_rows) != int(self.num_classes):
            if "struct_x_row_count_mismatch" not in reasons:
                reasons.append("struct_x_row_count_mismatch")
            is_aligned = False
        return TopologyProjectionDiagnostics(
            gateway_mode=self.gateway_mode,
            node_count=self.node_count,
            edge_count=self.edge_count,
            prediction_node_count=self.prediction_node_count,
            transparent_node_count=self.transparent_node_count,
            projected_edge_count=self.projected_edge_count,
            source_path_count=self.source_path_count,
            skipped_projected_edges=list(self.skipped_projected_edges),
            missing_vocab_nodes=list(self.missing_vocab_nodes),
            duplicate_activity_labels={key: list(value) for key, value in self.duplicate_activity_labels.items()},
            missing_node_metadata=self.missing_node_metadata,
            struct_x_rows=struct_x_rows,
            num_classes=self.num_classes,
            structural_edge_index_max=structural_edge_index_max,
            is_aligned=is_aligned,
            failure_reasons=reasons,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gateway_mode": self.gateway_mode,
            "node_count": int(self.node_count),
            "edge_count": int(self.edge_count),
            "prediction_node_count": int(self.prediction_node_count),
            "transparent_node_count": int(self.transparent_node_count),
            "projected_edge_count": int(self.projected_edge_count),
            "source_path_count": int(self.source_path_count),
            "skipped_projected_edges": [dict(item) for item in self.skipped_projected_edges],
            "missing_vocab_nodes": list(self.missing_vocab_nodes),
            "duplicate_activity_labels": {key: list(value) for key, value in self.duplicate_activity_labels.items()},
            "missing_node_metadata": bool(self.missing_node_metadata),
            "struct_x_rows": self.struct_x_rows,
            "num_classes": int(self.num_classes),
            "structural_edge_index_max": self.structural_edge_index_max,
            "is_aligned": bool(self.is_aligned),
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopologyProjectionDiagnostics":
        duplicate_raw = payload.get("duplicate_activity_labels", {})
        duplicates: dict[str, list[str]] = {}
        if isinstance(duplicate_raw, dict):
            for label, node_ids in duplicate_raw.items():
                if isinstance(node_ids, list):
                    duplicates[str(label)] = [str(item) for item in node_ids]
        return cls(
            gateway_mode=str(payload.get("gateway_mode", "preserve")),
            node_count=int(payload.get("node_count", 0) or 0),
            edge_count=int(payload.get("edge_count", 0) or 0),
            prediction_node_count=int(payload.get("prediction_node_count", 0) or 0),
            transparent_node_count=int(payload.get("transparent_node_count", 0) or 0),
            projected_edge_count=int(payload.get("projected_edge_count", 0) or 0),
            source_path_count=int(payload.get("source_path_count", 0) or 0),
            skipped_projected_edges=[
                {"src": str(item.get("src", "")), "dst": str(item.get("dst", "")), "reason": str(item.get("reason", ""))}
                for item in payload.get("skipped_projected_edges", [])
                if isinstance(item, dict)
            ],
            missing_vocab_nodes=[str(item) for item in payload.get("missing_vocab_nodes", [])],
            duplicate_activity_labels=duplicates,
            missing_node_metadata=bool(payload.get("missing_node_metadata", False)),
            struct_x_rows=payload.get("struct_x_rows"),
            num_classes=int(payload.get("num_classes", 0) or 0),
            structural_edge_index_max=payload.get("structural_edge_index_max"),
            is_aligned=bool(payload.get("is_aligned", True)),
            failure_reasons=[str(item) for item in payload.get("failure_reasons", [])],
        )


@dataclass(frozen=True)
class TopologyProjectionResult:
    projected_edge_paths: dict[tuple[str, str], list[list[str]]]
    prediction_nodes: set[str]
    transparent_nodes: set[str]
    diagnostics: TopologyProjectionDiagnostics


class TopologyProjectionCompiler:
    """Build prediction-space topology projection and audit class-space alignment."""

    def __init__(self, *, gateway_mode: str = "preserve") -> None:
        mode = str(gateway_mode or "preserve").strip().lower()
        self.gateway_mode = mode if mode in VALID_GATEWAY_MODES else "preserve"

    def project(self, *, dto: ProcessStructureDTO, activity_vocab: Dict[str, int]) -> TopologyProjectionResult:
        original_paths = self._original_edge_paths(dto)
        nodes = list(dto.nodes or [])
        missing_node_metadata = self.gateway_mode == "collapse_for_prediction" and not nodes
        prediction_nodes: set[str] = set()
        transparent_nodes: set[str] = set()

        if self.gateway_mode != "collapse_for_prediction" or not nodes:
            projected_edge_paths = original_paths
            if nodes:
                roles = self.classify_nodes(nodes)
                prediction_nodes = {node_id for node_id, role in roles.items() if role == "prediction"}
                transparent_nodes = set(roles) - prediction_nodes
        else:
            roles = self.classify_nodes(nodes)
            prediction_nodes = {node_id for node_id, role in roles.items() if role == "prediction"}
            transparent_nodes = set(roles) - prediction_nodes
            projected_edge_paths = self._collapse_prediction_paths(dto=dto, node_roles=roles)
            if not projected_edge_paths:
                projected_edge_paths = original_paths

        missing_vocab_nodes, skipped_edges = self._vocab_alignment(projected_edge_paths, activity_vocab)
        duplicate_labels = self._duplicate_activity_labels(nodes)
        failure_reasons: list[str] = []
        if missing_node_metadata:
            failure_reasons.append("missing_node_metadata")
        if missing_vocab_nodes:
            failure_reasons.append("missing_vocab_nodes")
        diagnostics = TopologyProjectionDiagnostics(
            gateway_mode=self.gateway_mode,
            node_count=len(nodes),
            edge_count=len(original_paths),
            prediction_node_count=len(prediction_nodes),
            transparent_node_count=len(transparent_nodes),
            projected_edge_count=len(projected_edge_paths),
            source_path_count=sum(len(paths) for paths in projected_edge_paths.values()),
            skipped_projected_edges=skipped_edges,
            missing_vocab_nodes=missing_vocab_nodes,
            duplicate_activity_labels=duplicate_labels,
            missing_node_metadata=missing_node_metadata,
            num_classes=len(activity_vocab),
            is_aligned=not failure_reasons,
            failure_reasons=failure_reasons,
        )
        return TopologyProjectionResult(
            projected_edge_paths=dict(sorted(projected_edge_paths.items(), key=lambda item: item[0])),
            prediction_nodes=prediction_nodes,
            transparent_nodes=transparent_nodes,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _original_edge_paths(dto: ProcessStructureDTO) -> dict[tuple[str, str], list[list[str]]]:
        return {
            (str(src).strip(), str(dst).strip()): [[str(src).strip(), str(dst).strip()]]
            for src, dst in dto.allowed_edges
            if str(src).strip() and str(dst).strip()
        }

    @classmethod
    def classify_nodes(cls, nodes: list[Dict[str, Any]]) -> dict[str, str]:
        roles: dict[str, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            tags = " ".join(
                str(node.get(key, "")).strip().lower()
                for key in ("bpmn_tag", "type", "activity_type", "logical_type")
            )
            normalized = tags.replace(" ", "").replace("_", "").replace("-", "")
            roles[node_id] = "prediction" if cls.is_prediction_node_type(normalized) else "transparent"
        return roles

    @staticmethod
    def is_prediction_node_type(normalized_type: str) -> bool:
        if not normalized_type:
            return False
        if "gateway" in normalized_type:
            return False
        if "event" in normalized_type:
            return False
        return any(
            token in normalized_type
            for token in (
                "task",
                "usertask",
                "servicetask",
                "scripttask",
                "manualtask",
                "businessruletask",
                "sendtask",
                "receivetask",
                "callactivity",
                "subprocess",
            )
        )

    def _collapse_prediction_paths(
        self,
        *,
        dto: ProcessStructureDTO,
        node_roles: dict[str, str],
    ) -> dict[tuple[str, str], list[list[str]]]:
        outgoing: dict[str, list[str]] = {}
        for src, dst in dto.allowed_edges:
            src_token = str(src).strip()
            dst_token = str(dst).strip()
            if src_token and dst_token:
                outgoing.setdefault(src_token, []).append(dst_token)

        projected: dict[tuple[str, str], list[list[str]]] = {}
        for src, role in node_roles.items():
            if role != "prediction":
                continue
            for dst, paths in self.reachable_prediction_node_paths_through_transparent(
                source=src,
                outgoing=outgoing,
                node_roles=node_roles,
            ).items():
                if dst != src:
                    projected.setdefault((src, dst), []).extend(paths)
        return dict(sorted(projected.items(), key=lambda item: item[0]))

    @staticmethod
    def reachable_prediction_node_paths_through_transparent(
        *,
        source: str,
        outgoing: dict[str, list[str]],
        node_roles: dict[str, str],
    ) -> dict[str, list[list[str]]]:
        found: dict[str, list[list[str]]] = {}
        visited: set[tuple[str, ...]] = set()
        queue: list[tuple[str, list[str]]] = [(node, [source, node]) for node in outgoing.get(source, [])]
        while queue:
            current, path = queue.pop(0)
            path_key = tuple(path)
            if path_key in visited:
                continue
            visited.add(path_key)
            role = node_roles.get(current, "prediction")
            if role == "prediction":
                found.setdefault(current, []).append(path)
                continue
            for next_node in outgoing.get(current, []):
                if next_node in path:
                    continue
                queue.append((next_node, [*path, next_node]))
        return found

    @staticmethod
    def _vocab_alignment(
        projected_edge_paths: dict[tuple[str, str], list[list[str]]],
        activity_vocab: Dict[str, int],
    ) -> tuple[list[str], list[dict[str, str]]]:
        missing: set[str] = set()
        skipped: list[dict[str, str]] = []
        for src, dst in projected_edge_paths:
            src_missing = src not in activity_vocab
            dst_missing = dst not in activity_vocab
            if src_missing:
                missing.add(src)
            if dst_missing:
                missing.add(dst)
            if src_missing or dst_missing:
                if src_missing and dst_missing:
                    reason = "missing_src_dst_vocab"
                elif src_missing:
                    reason = "missing_src_vocab"
                else:
                    reason = "missing_dst_vocab"
                skipped.append({"src": src, "dst": dst, "reason": reason})
        return sorted(missing), skipped

    @staticmethod
    def _duplicate_activity_labels(nodes: list[Dict[str, Any]]) -> dict[str, list[str]]:
        by_label: dict[str, list[str]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            label = ""
            for key in ("activity_label", "label", "name"):
                label = str(node.get(key, "")).strip()
                if label:
                    break
            if not label:
                label = node_id
            by_label.setdefault(label, []).append(node_id)
        return {label: sorted(ids) for label, ids in sorted(by_label.items()) if len(ids) > 1}
