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
        on_missing_asof_snapshot: str = "disable_stats",
        cache_policy: str = "full",
    ) -> None:
        super().__init__(feature_encoder=feature_encoder)
        self.knowledge_port = knowledge_port
        self.process_name = str(process_name).strip() if process_name is not None else None
        self.graph_feature_mapping = dict(graph_feature_mapping or {})
        self.stats_time_policy = str(stats_time_policy).strip().lower() or "latest"
        if self.stats_time_policy not in {"latest", "strict_asof"}:
            self.stats_time_policy = "latest"
        self.on_missing_asof_snapshot = self._resolve_missing_asof_policy(on_missing_asof_snapshot)
        self.stats_quality_gate = self._resolve_quality_gate_config(stats_quality_gate)
        self._quality_warned_keys: set[tuple[str, str, str]] = set()
        self._missing_asof_warned_keys: set[tuple[str, str, str]] = set()
        self.cache_policy = self._resolve_cache_policy(cache_policy)
        self._dto_cache: dict[tuple[str, tuple[str, ...], str | None], ProcessStructureDTO | None] = {}
        self._topology_cache: dict[tuple[Any, ...], Dict[str, Any]] = {}
        self._dto_cache_max_entries = 32768
        self._topology_cache_max_entries = 512

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Build baseline contract and inject OOS mask when topology is available."""
        contract = super().build_graph(prefix)
        as_of_ts = self._resolve_as_of_timestamp(prefix) if self.stats_time_policy == "strict_asof" else None

        raw_version = str(prefix.process_version).strip() or "default"
        candidate_versions = [raw_version]
        if raw_version.isdigit():
            candidate_versions.append(f"v{int(raw_version)}")
        elif raw_version.lower().startswith("v") and raw_version[1:].isdigit():
            candidate_versions.append(str(int(raw_version[1:])))

        dto = self._resolve_dto(as_of_ts=as_of_ts, candidate_versions=candidate_versions)
        if dto is None or not prefix.prefix_events:
            contract["allowed_target_mask"] = None
            return contract

        missing_asof_snapshot = self._is_missing_asof_snapshot(dto=dto, as_of_ts=as_of_ts)
        stats_allowed, quality_reason = self._should_use_stats(dto=dto, version_key=raw_version)
        if missing_asof_snapshot:
            self._emit_missing_asof_warning(version_key=raw_version, as_of_ts=as_of_ts)
            if self.on_missing_asof_snapshot == "raise":
                as_of_text = as_of_ts.isoformat() if isinstance(as_of_ts, datetime) else "none"
                raise ValueError(
                    f"Strict as-of snapshot is missing for process='{self.process_name or '__auto__'}' "
                    f"version='{raw_version}' as_of='{as_of_text}'."
                )
            if self.on_missing_asof_snapshot == "disable_stats":
                stats_allowed = False
                quality_reason = "missing_asof_snapshot"
        snapshot_meta = self._stats_snapshot_metadata(dto)
        contract["stats_snapshot_version_seq"] = self._snapshot_version_seq(snapshot_meta.get("knowledge_version"))
        contract["stats_snapshot_as_of_epoch"] = self._snapshot_as_of_epoch(snapshot_meta.get("as_of_ts"))
        contract["stats_allowed"] = bool(stats_allowed)
        contract["stats_missing_asof_snapshot"] = bool(missing_asof_snapshot)
        target_feature = self.feature_encoder.activity_feature_name
        activity_vocab = self.feature_encoder.categorical_vocabs.get(target_feature, {"<UNK>": 0})
        if quality_reason not in {"ok", "quality_gate_disabled", "quality_metadata_not_required"}:
            self._emit_quality_warning(version_key=raw_version, reason=quality_reason, stats_allowed=stats_allowed)
        compiled = self._resolve_compiled_topology(
            dto=dto,
            activity_vocab=activity_vocab,
            stats_allowed=stats_allowed,
        )

        last_activity = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=prefix.prefix_events[-1].extra,
                feature_name=target_feature,
                default=prefix.prefix_events[-1].activity_id,
            )
        )
        num_classes = len(activity_vocab)
        allowed_mask = torch.zeros(num_classes, dtype=torch.bool)
        last_activity_idx = activity_vocab.get(last_activity)
        if last_activity_idx is None:
            last_activity_idx = activity_vocab.get(last_activity.strip())
        if last_activity_idx is not None:
            cached_mask = compiled["allowed_masks_by_src"].get(int(last_activity_idx))
            if isinstance(cached_mask, torch.Tensor):
                allowed_mask = cached_mask.clone()

        contract["structural_edge_index"] = compiled["structural_edge_index"].clone()
        contract["structural_edge_weight"] = compiled["structural_edge_weight"].clone()
        struct_x = compiled.get("struct_x")
        if isinstance(struct_x, torch.Tensor):
            contract["struct_x"] = struct_x.clone()
        contract["allowed_target_mask"] = allowed_mask
        return contract

    @staticmethod
    def _clean_optional_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"", "none", "null", "nan"}:
            return ""
        return text

    @staticmethod
    def _stats_snapshot_metadata(dto: ProcessStructureDTO) -> Dict[str, str | None]:
        metadata = dto.metadata if isinstance(dto.metadata, dict) else {}
        knowledge_version = DynamicGraphBuilder._clean_optional_text(metadata.get("knowledge_version")) if isinstance(metadata, dict) else ""
        as_of_ts = DynamicGraphBuilder._clean_optional_text(metadata.get("as_of_ts")) if isinstance(metadata, dict) else ""
        if not knowledge_version or not as_of_ts:
            stats_contract = metadata.get("stats_contract", {}) if isinstance(metadata, dict) else {}
            identity = stats_contract.get("identity", {}) if isinstance(stats_contract, dict) else {}
            if not knowledge_version:
                knowledge_version = DynamicGraphBuilder._clean_optional_text(identity.get("knowledge_version")) if isinstance(identity, dict) else ""
            if not as_of_ts:
                as_of_ts = DynamicGraphBuilder._clean_optional_text(identity.get("as_of_ts")) if isinstance(identity, dict) else ""
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

    @staticmethod
    def _resolve_missing_asof_policy(raw: Any) -> str:
        value = str(raw).strip().lower() or "disable_stats"
        if value not in {"disable_stats", "use_base", "raise"}:
            return "disable_stats"
        return value

    @staticmethod
    def _resolve_cache_policy(raw: Any) -> str:
        value = str(raw).strip().lower() or "full"
        if value in {"none", "disabled", "false"}:
            return "off"
        if value not in {"off", "dto", "full"}:
            return "full"
        return value

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

    def _emit_missing_asof_warning(self, *, version_key: str, as_of_ts: datetime | None) -> None:
        process = self.process_name or "__auto__"
        as_of_text = as_of_ts.isoformat() if isinstance(as_of_ts, datetime) else "none"
        key = (process, str(version_key), as_of_text)
        if key in self._missing_asof_warned_keys:
            return
        self._missing_asof_warned_keys.add(key)
        logger.warning(
            "Strict as-of snapshot missing: process=%s version=%s as_of=%s policy=%s.",
            process,
            version_key,
            as_of_text,
            self.on_missing_asof_snapshot,
        )

    @staticmethod
    def _resolve_as_of_timestamp(prefix: PrefixSlice) -> datetime | None:
        if not prefix.prefix_events:
            return None
        return datetime.fromtimestamp(float(prefix.prefix_events[-1].timestamp), tz=timezone.utc)

    def _resolve_dto(
        self,
        *,
        as_of_ts: datetime | None,
        candidate_versions: list[str],
    ) -> ProcessStructureDTO | None:
        cache_key: tuple[str, tuple[str, ...], str | None] | None = None
        if self.cache_policy in {"dto", "full"}:
            cache_key = (
                self.process_name or "__auto__",
                tuple(str(item).strip() for item in candidate_versions),
                as_of_ts.isoformat() if isinstance(as_of_ts, datetime) else None,
            )
            if cache_key in self._dto_cache:
                return self._dto_cache[cache_key]

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
                if cache_key is not None:
                    self._cache_put(
                        cache=self._dto_cache,
                        key=cache_key,
                        value=dto,
                        max_entries=self._dto_cache_max_entries,
                    )
                return dto
            if self.process_name is not None:
                if hasattr(self.knowledge_port, "get_process_structure_as_of") and as_of_ts is not None:
                    getter = getattr(self.knowledge_port, "get_process_structure_as_of")
                    dto = getter(candidate, as_of_ts=as_of_ts)
                else:
                    # Legacy compatibility for repositories where version-only lookup is still used.
                    dto = self.knowledge_port.get_process_structure(candidate)
                if dto is not None:
                    if cache_key is not None:
                        self._cache_put(
                            cache=self._dto_cache,
                            key=cache_key,
                            value=dto,
                            max_entries=self._dto_cache_max_entries,
                        )
                    return dto
        if cache_key is not None:
            self._cache_put(
                cache=self._dto_cache,
                key=cache_key,
                value=None,
                max_entries=self._dto_cache_max_entries,
            )
        return None

    @staticmethod
    def _cache_put(cache: Dict[Any, Any], key: Any, value: Any, *, max_entries: int) -> None:
        cache[key] = value
        while len(cache) > max_entries:
            cache.pop(next(iter(cache)))

    def _topology_cache_key(
        self,
        *,
        dto: ProcessStructureDTO,
        activity_vocab: Dict[str, int],
        stats_allowed: bool,
    ) -> tuple[Any, ...]:
        snapshot = self._stats_snapshot_metadata(dto)
        normalized_edges = tuple((str(src).strip(), str(dst).strip()) for src, dst in dto.allowed_edges)
        return (
            self.process_name or "__auto__",
            str(dto.version),
            self._clean_optional_text(snapshot.get("knowledge_version")),
            self._clean_optional_text(snapshot.get("as_of_ts")),
            bool(stats_allowed),
            int(len(activity_vocab)),
            normalized_edges,
        )

    def _resolve_compiled_topology(
        self,
        *,
        dto: ProcessStructureDTO,
        activity_vocab: Dict[str, int],
        stats_allowed: bool,
    ) -> Dict[str, Any]:
        cache_key: tuple[Any, ...] | None = None
        if self.cache_policy == "full":
            cache_key = self._topology_cache_key(
                dto=dto,
                activity_vocab=activity_vocab,
                stats_allowed=stats_allowed,
            )
            cached = self._topology_cache.get(cache_key)
            if isinstance(cached, dict):
                return cached

        num_classes = len(activity_vocab)
        allowed_masks_by_src: Dict[int, torch.Tensor] = {}
        structural_src: list[int] = []
        structural_dst: list[int] = []
        structural_weight: list[float] = []
        edge_stats = dto.edge_statistics or {}
        edge_weight_spec = self._edge_weight_spec() if stats_allowed else None
        edge_weight_index = self._edge_stats_index(dto) if stats_allowed else {}
        if not stats_allowed:
            edge_stats = {}

        for src, dst in dto.allowed_edges:
            src_token = str(src).strip()
            dst_token = str(dst).strip()
            src_idx = activity_vocab.get(src_token)
            dst_idx = activity_vocab.get(dst_token)
            if src_idx is None or dst_idx is None:
                continue
            src_idx_int = int(src_idx)
            dst_idx_int = int(dst_idx)
            structural_src.append(src_idx_int)
            structural_dst.append(dst_idx_int)
            if edge_weight_spec is not None:
                edge_key = f"{src_token}|||{dst_token}"
                structural_weight.append(float(edge_weight_index.get(edge_key, edge_weight_spec.get("default", 1.0))))
            else:
                stats = edge_stats.get((src, dst), {})
                structural_weight.append(float(stats.get("count", 1.0)))
            row_mask = allowed_masks_by_src.get(src_idx_int)
            if row_mask is None:
                row_mask = torch.zeros(num_classes, dtype=torch.bool)
                allowed_masks_by_src[src_idx_int] = row_mask
            row_mask[dst_idx_int] = True

        if structural_src:
            structural_edge_index = torch.tensor([structural_src, structural_dst], dtype=torch.long)
            structural_edge_weight = torch.tensor(structural_weight, dtype=torch.float32)
            if edge_weight_spec is not None:
                structural_edge_weight = self._apply_numeric_encodings(
                    structural_edge_weight,
                    list(edge_weight_spec.get("encoding", ["identity"])),
                )
        else:
            structural_edge_index = torch.zeros((2, 0), dtype=torch.long)
            structural_edge_weight = torch.zeros((0,), dtype=torch.float32)

        struct_x = self._build_struct_x(dto=dto, activity_vocab=activity_vocab) if stats_allowed else None
        compiled = {
            "allowed_masks_by_src": allowed_masks_by_src,
            "structural_edge_index": structural_edge_index,
            "structural_edge_weight": structural_edge_weight,
            "struct_x": struct_x,
        }
        if cache_key is not None:
            self._cache_put(
                cache=self._topology_cache,
                key=cache_key,
                value=compiled,
                max_entries=self._topology_cache_max_entries,
            )
        return compiled

    def _is_missing_asof_snapshot(self, *, dto: ProcessStructureDTO, as_of_ts: datetime | None) -> bool:
        if self.stats_time_policy != "strict_asof" or as_of_ts is None:
            return False
        metadata = dto.metadata if isinstance(dto.metadata, dict) else {}
        if isinstance(metadata, dict):
            marker = metadata.get("asof_snapshot_found")
            if marker is False:
                return True
            resolution = str(metadata.get("asof_resolution", "")).strip().lower()
            if resolution in {"missing_snapshot_fallback_base", "fallback_base_no_snapshot"}:
                return True
        snapshot_meta = self._stats_snapshot_metadata(dto)
        knowledge_version = self._clean_optional_text(snapshot_meta.get("knowledge_version"))
        as_of_raw = self._clean_optional_text(snapshot_meta.get("as_of_ts"))
        if not knowledge_version or not as_of_raw:
            return True
        snapshot_epoch = self._snapshot_as_of_epoch(as_of_raw)
        if snapshot_epoch is None:
            return True
        return bool(snapshot_epoch > float(as_of_ts.timestamp()) + 1e-6)

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
