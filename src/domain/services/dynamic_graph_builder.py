"""Dynamic graph builder with MVP2 topology-to-tensor injection."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional
import logging

import torch

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder


logger = logging.getLogger(__name__)


class DynamicGraphBuilder(BaselineGraphBuilder):
    """Extends baseline tensors with version-scoped allowed-target mask."""

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        knowledge_port: IKnowledgeGraphPort,
        process_name: str | None = None,
        graph_feature_mapping: Optional[Dict[str, Any]] = None,
        stats_time_policy: str = "latest",
        stats_quality_gate: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(feature_encoder=feature_encoder)
        self.knowledge_port = knowledge_port
        self.process_name = str(process_name).strip() if process_name is not None else None
        self.graph_feature_mapping = dict(graph_feature_mapping or {})
        self.stats_time_policy = str(stats_time_policy).strip().lower() or "latest"
        if self.stats_time_policy not in {"latest", "strict_asof"}:
            self.stats_time_policy = "latest"
        self.stats_quality_gate = self._resolve_quality_gate_config(stats_quality_gate)
        self._quality_warned_keys: set[tuple[str, str, str]] = set()

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

        stats_allowed, quality_reason = self._should_use_stats(dto=dto, version_key=raw_version)
        snapshot_meta = self._stats_snapshot_metadata(dto)
        contract["stats_snapshot_version_seq"] = self._snapshot_version_seq(snapshot_meta.get("knowledge_version"))
        contract["stats_snapshot_as_of_epoch"] = self._snapshot_as_of_epoch(snapshot_meta.get("as_of_ts"))
        contract["stats_allowed"] = bool(stats_allowed)
        target_feature = self.feature_encoder.activity_feature_name
        activity_vocab = self.feature_encoder.categorical_vocabs.get(target_feature, {"<UNK>": 0})
        num_classes = len(activity_vocab)
        allowed_mask = torch.zeros(num_classes, dtype=torch.bool)
        structural_src: list[int] = []
        structural_dst: list[int] = []
        structural_weight: list[float] = []
        edge_stats = dto.edge_statistics or {}
        edge_weight_spec = self._edge_weight_spec() if stats_allowed else None
        edge_weight_index = self._edge_stats_index(dto) if stats_allowed else {}
        if quality_reason not in {"ok", "quality_gate_disabled", "quality_metadata_not_required"}:
            self._emit_quality_warning(version_key=raw_version, reason=quality_reason, stats_allowed=stats_allowed)
        if not stats_allowed:
            edge_stats = {}

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
            weight_tensor = torch.tensor(structural_weight, dtype=torch.float32)
            if edge_weight_spec is not None:
                weight_tensor = self._apply_numeric_encodings(weight_tensor, list(edge_weight_spec.get("encoding", ["identity"])))
            contract["structural_edge_weight"] = weight_tensor
        else:
            contract["structural_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
            contract["structural_edge_weight"] = torch.zeros((0,), dtype=torch.float32)

        struct_x = self._build_struct_x(dto=dto, activity_vocab=activity_vocab) if stats_allowed else None
        if struct_x is not None:
            contract["struct_x"] = struct_x
        contract["allowed_target_mask"] = allowed_mask
        return contract

    @staticmethod
    def _stats_snapshot_metadata(dto: ProcessStructureDTO) -> Dict[str, str | None]:
        metadata = dto.metadata if isinstance(dto.metadata, dict) else {}
        knowledge_version = str(metadata.get("knowledge_version", "")).strip() if isinstance(metadata, dict) else ""
        as_of_ts = str(metadata.get("as_of_ts", "")).strip() if isinstance(metadata, dict) else ""
        if not knowledge_version or not as_of_ts:
            stats_contract = metadata.get("stats_contract", {}) if isinstance(metadata, dict) else {}
            identity = stats_contract.get("identity", {}) if isinstance(stats_contract, dict) else {}
            if not knowledge_version:
                knowledge_version = str(identity.get("knowledge_version", "")).strip() if isinstance(identity, dict) else ""
            if not as_of_ts:
                as_of_ts = str(identity.get("as_of_ts", "")).strip() if isinstance(identity, dict) else ""
        return {
            "knowledge_version": knowledge_version or None,
            "as_of_ts": as_of_ts or None,
        }

    @staticmethod
    def _snapshot_version_seq(raw: str | None) -> int | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        if text.lower().startswith("k") and text[1:].isdigit():
            return int(text[1:])
        if text.isdigit():
            return int(text)
        return None

    @staticmethod
    def _snapshot_as_of_epoch(raw: str | None) -> float | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())

    def _resolve_quality_gate_config(self, gate_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        graph_cfg = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
        raw = gate_cfg
        if raw is None:
            nested = graph_cfg.get("stats_quality_gate")
            raw = dict(nested) if isinstance(nested, dict) else {}
        cfg = dict(raw) if isinstance(raw, dict) else {}

        enabled = bool(cfg.get("enabled", True))
        require_quality_metadata = bool(cfg.get("require_quality_metadata", False))
        warn_on_fail = bool(cfg.get("warn_on_fail", True))

        zero_dominant_threshold = float(cfg.get("zero_dominant_threshold", 0.95) or 0.95)
        min_non_zero_ratio_overall = float(cfg.get("min_non_zero_ratio_overall", 0.0) or 0.0)
        min_history_coverage_percent = float(cfg.get("min_history_coverage_percent", 0.0) or 0.0)
        on_fail = str(cfg.get("on_fail", "ignore_with_warning")).strip().lower() or "ignore_with_warning"
        if on_fail not in {"ignore_with_warning", "allow_with_warning", "raise"}:
            on_fail = "ignore_with_warning"

        return {
            "enabled": enabled,
            "require_quality_metadata": require_quality_metadata,
            "warn_on_fail": warn_on_fail,
            "zero_dominant_threshold": float(min(max(zero_dominant_threshold, 0.0), 1.0)),
            "min_non_zero_ratio_overall": float(min(max(min_non_zero_ratio_overall, 0.0), 1.0)),
            "min_history_coverage_percent": float(min(max(min_history_coverage_percent, 0.0), 100.0)),
            "on_fail": on_fail,
        }

    def _should_use_stats(self, *, dto: ProcessStructureDTO, version_key: str) -> tuple[bool, str]:
        gate = self.stats_quality_gate
        if not bool(gate.get("enabled", True)):
            return True, "quality_gate_disabled"

        metadata = dto.metadata if isinstance(dto.metadata, dict) else {}
        contract = metadata.get("stats_contract", {}) if isinstance(metadata, dict) else {}
        quality = contract.get("quality", {}) if isinstance(contract, dict) else {}
        if not isinstance(quality, dict) or not quality:
            if bool(gate.get("require_quality_metadata", False)):
                reason = "missing_quality_metadata"
                action = str(gate.get("on_fail", "ignore_with_warning"))
                if action == "raise":
                    raise ValueError(f"Stats quality metadata is required but missing for version '{version_key}'.")
                if action == "allow_with_warning":
                    return True, reason
                return False, reason
            return True, "quality_metadata_not_required"

        reasons: list[str] = []
        try:
            coverage = float(quality.get("history_coverage_percent", 0.0) or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        if coverage < float(gate.get("min_history_coverage_percent", 0.0)):
            reasons.append("below_min_coverage_threshold")

        try:
            non_zero_ratio = float(quality.get("non_zero_ratio_overall", 0.0) or 0.0)
        except (TypeError, ValueError):
            non_zero_ratio = 0.0
        if non_zero_ratio < float(gate.get("min_non_zero_ratio_overall", 0.0)):
            reasons.append("below_min_non_zero_ratio_threshold")

        zero_dominant = bool(quality.get("zero_dominant", False))
        if zero_dominant:
            reasons.append("zero_dominant")

        if not bool(quality.get("is_usable_for_training", True)):
            reasons.append(str(quality.get("quality_reason", "producer_marked_unusable")))

        if not reasons:
            return True, "ok"

        reason = reasons[0]
        action = str(gate.get("on_fail", "ignore_with_warning"))
        if action == "raise":
            raise ValueError(f"Stats quality gate rejected version '{version_key}': {reason}")
        if action == "allow_with_warning":
            return True, reason
        return False, reason

    def _emit_quality_warning(self, *, version_key: str, reason: str, stats_allowed: bool) -> None:
        if not bool(self.stats_quality_gate.get("warn_on_fail", True)):
            return
        process = self.process_name or "__auto__"
        key = (process, str(version_key), str(reason))
        if key in self._quality_warned_keys:
            return
        self._quality_warned_keys.add(key)
        logger.warning(
            "Stats quality gate: process=%s version=%s reason=%s action=%s stats_allowed=%s.",
            process,
            version_key,
            reason,
            self.stats_quality_gate.get("on_fail", "ignore_with_warning"),
            bool(stats_allowed),
        )

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
                    "encoding": self._normalize_encodings(item.get("encoding", ["identity"])),
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
            "encoding": self._normalize_encodings(raw.get("encoding", ["identity"])),
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
            col_values = torch.full((num_classes,), default, dtype=torch.float32)
            for token, idx in activity_vocab.items():
                col_values[int(idx)] = float(values.get(str(token), default))
            out[:, col] = self._apply_numeric_encodings(col_values, spec["encoding"])
        return out

    @staticmethod
    def _normalize_encodings(raw: Any) -> list[str]:
        values: list[str] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                text = str(item).strip().lower()
                if text:
                    values.append(text)
        elif raw is not None:
            text = str(raw).strip().lower()
            if text:
                values.append(text)
        if not values:
            return ["identity"]
        return values

    @staticmethod
    def _apply_numeric_encodings(column: torch.Tensor, encodings: list[str]) -> torch.Tensor:
        out = column.clone().to(dtype=torch.float32)
        if out.numel() == 0:
            return out

        for enc in encodings:
            mode = str(enc).strip().lower()
            if mode in {"", "identity", "none", "embedding"}:
                continue

            if mode == "log1p":
                out = torch.sign(out) * torch.log1p(torch.abs(out))
                continue

            if mode == "z-score":
                finite_mask = torch.isfinite(out)
                if not bool(torch.any(finite_mask)):
                    out = torch.zeros_like(out)
                    continue
                finite = out[finite_mask]
                mean = torch.mean(finite)
                std = torch.std(finite, unbiased=False)
                if float(std) <= 1e-12 or not math.isfinite(float(std)):
                    out = torch.zeros_like(out)
                else:
                    normalized = (out - mean) / std
                    normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
                    out = normalized
                continue

            # Unknown encoding mode is treated as no-op for backward compatibility.
            continue

        return out
