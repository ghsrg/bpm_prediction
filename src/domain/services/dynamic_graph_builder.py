"""Dynamic graph builder with MVP2 topology-to-tensor injection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder


class DynamicGraphBuilder(BaselineGraphBuilder):
    """Extends baseline tensors with version-scoped allowed-target mask."""

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        knowledge_port: IKnowledgeGraphPort,
        process_name: str | None = None,
        graph_feature_mapping: Optional[Dict[str, Any]] = None,
        stats_time_policy: str = "latest",
    ) -> None:
        super().__init__(feature_encoder=feature_encoder)
        self.knowledge_port = knowledge_port
        self.process_name = str(process_name).strip() if process_name is not None else None
        self.graph_feature_mapping = dict(graph_feature_mapping or {})
        self.stats_time_policy = str(stats_time_policy).strip().lower() or "latest"
        if self.stats_time_policy not in {"latest", "strict_asof"}:
            self.stats_time_policy = "latest"

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Build baseline contract and inject OOS mask when topology is available."""
        contract = super().build_graph(prefix)

        raw_version = str(prefix.process_version).strip() or "default"
        candidate_versions = [raw_version]
        if raw_version.isdigit():
            candidate_versions.append(f"v{int(raw_version)}")
        elif raw_version.lower().startswith("v") and raw_version[1:].isdigit():
            candidate_versions.append(str(int(raw_version[1:])))

        dto = self._resolve_dto(prefix=prefix, candidate_versions=candidate_versions)
        if dto is None or not prefix.prefix_events:
            contract["allowed_target_mask"] = None
            return contract

        target_feature = self.feature_encoder.activity_feature_name
        activity_vocab = self.feature_encoder.categorical_vocabs.get(target_feature, {"<UNK>": 0})
        num_classes = len(activity_vocab)
        allowed_mask = torch.zeros(num_classes, dtype=torch.bool)
        structural_src: list[int] = []
        structural_dst: list[int] = []
        structural_weight: list[float] = []
        edge_stats = dto.edge_statistics or {}
        edge_weight_spec = self._edge_weight_spec()
        edge_weight_index = self._edge_stats_index(dto)

        last_activity = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=prefix.prefix_events[-1].extra,
                feature_name=target_feature,
                default=prefix.prefix_events[-1].activity_id,
            )
        )
        for src, dst in dto.allowed_edges:
            src_token = str(src).strip()
            dst_token = str(dst).strip()
            src_idx = activity_vocab.get(src_token)
            dst_idx = activity_vocab.get(dst_token)
            if src_idx is not None and dst_idx is not None:
                structural_src.append(int(src_idx))
                structural_dst.append(int(dst_idx))
                if edge_weight_spec is not None:
                    edge_key = f"{src_token}|||{dst_token}"
                    structural_weight.append(float(edge_weight_index.get(edge_key, edge_weight_spec.get("default", 1.0))))
                else:
                    stats = edge_stats.get((src, dst), {})
                    structural_weight.append(float(stats.get("count", 1.0)))

            if str(src) != last_activity:
                continue
            dst_token = str(dst)
            dst_idx = activity_vocab.get(dst_token)
            if dst_idx is None:
                dst_idx = activity_vocab.get(dst_token.strip())
            if dst_idx is not None:
                allowed_mask[dst_idx] = True

        if structural_src:
            contract["structural_edge_index"] = torch.tensor([structural_src, structural_dst], dtype=torch.long)
            contract["structural_edge_weight"] = torch.tensor(structural_weight, dtype=torch.float32)
        else:
            contract["structural_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
            contract["structural_edge_weight"] = torch.zeros((0,), dtype=torch.float32)

        struct_x = self._build_struct_x(dto=dto, activity_vocab=activity_vocab)
        if struct_x is not None:
            contract["struct_x"] = struct_x
        contract["allowed_target_mask"] = allowed_mask
        return contract

    def _resolve_dto(
        self,
        *,
        prefix: PrefixSlice,
        candidate_versions: list[str],
    ) -> ProcessStructureDTO | None:
        as_of_ts: datetime | None = None
        if self.stats_time_policy == "strict_asof" and prefix.prefix_events:
            as_of_ts = datetime.fromtimestamp(float(prefix.prefix_events[-1].timestamp), tz=timezone.utc)

        for candidate in candidate_versions:
            if hasattr(self.knowledge_port, "get_process_structure_as_of") and as_of_ts is not None:
                getter = getattr(self.knowledge_port, "get_process_structure_as_of")
                dto = getter(
                    candidate,
                    process_name=self.process_name,
                    as_of_ts=as_of_ts,
                )
            else:
                dto = self.knowledge_port.get_process_structure(
                    candidate,
                    process_name=self.process_name,
                )
            if dto is not None:
                return dto
            if self.process_name is not None:
                if hasattr(self.knowledge_port, "get_process_structure_as_of") and as_of_ts is not None:
                    getter = getattr(self.knowledge_port, "get_process_structure_as_of")
                    dto = getter(candidate, as_of_ts=as_of_ts)
                else:
                    # Legacy compatibility for repositories where version-only lookup is still used.
                    dto = self.knowledge_port.get_process_structure(candidate)
                if dto is not None:
                    return dto
        return None

    def _node_specs(self) -> list[Dict[str, Any]]:
        block = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
        if not bool(block.get("enabled", False)):
            return []
        raw_specs = block.get("node_numeric", [])
        if not isinstance(raw_specs, list):
            return []
        specs: list[Dict[str, Any]] = []
        for item in raw_specs:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", "")).strip()
            window = str(item.get("window", "last_30d")).strip() or "last_30d"
            scope = str(item.get("scope", "version")).strip() or "version"
            if not metric:
                continue
            specs.append(
                {
                    "name": str(item.get("name", metric)).strip() or metric,
                    "metric": metric,
                    "window": window,
                    "scope": scope,
                    "default": float(item.get("default", 0.0) or 0.0),
                }
            )
        return specs

    def _edge_weight_spec(self) -> Dict[str, Any] | None:
        block = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
        if not bool(block.get("enabled", False)):
            return None
        raw = block.get("edge_weight")
        if not isinstance(raw, dict):
            return None
        metric = str(raw.get("metric", "")).strip()
        if not metric:
            return None
        return {
            "metric": metric,
            "window": str(raw.get("window", "last_30d")).strip() or "last_30d",
            "scope": str(raw.get("scope", "version")).strip() or "version",
            "default": float(raw.get("default", 1.0) or 1.0),
        }

    @staticmethod
    def _stats_index(dto: ProcessStructureDTO, section: str) -> Dict[str, Dict[str, float]]:
        metadata = dto.metadata if isinstance(dto.metadata, dict) else {}
        stats_index = metadata.get("stats_index", {}) if isinstance(metadata, dict) else {}
        if not isinstance(stats_index, dict):
            return {}
        section_map = stats_index.get(section, {})
        if not isinstance(section_map, dict):
            return {}
        normalized: Dict[str, Dict[str, float]] = {}
        for key, payload in section_map.items():
            if not isinstance(payload, dict):
                continue
            values: Dict[str, float] = {}
            for item_key, item_value in payload.items():
                try:
                    values[str(item_key)] = float(item_value)
                except (TypeError, ValueError):
                    continue
            normalized[str(key)] = values
        return normalized

    def _edge_stats_index(self, dto: ProcessStructureDTO) -> Dict[str, float]:
        spec = self._edge_weight_spec()
        if spec is None:
            return {}
        key = f"{spec['window']}.{spec['scope']}.{spec['metric']}"
        return self._stats_index(dto, "edge").get(key, {})

    def _build_struct_x(
        self,
        *,
        dto: ProcessStructureDTO,
        activity_vocab: Dict[str, int],
    ) -> torch.Tensor | None:
        specs = self._node_specs()
        if not specs:
            return None
        node_index = self._stats_index(dto, "node")
        num_classes = len(activity_vocab)
        out = torch.zeros((num_classes, len(specs)), dtype=torch.float32)
        for col, spec in enumerate(specs):
            key = f"{spec['window']}.{spec['scope']}.{spec['metric']}"
            values = node_index.get(key, {})
            default = float(spec["default"])
            for token, idx in activity_vocab.items():
                out[int(idx), col] = float(values.get(str(token), default))
        return out
