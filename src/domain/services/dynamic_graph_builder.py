"""Dynamic graph builder with MVP2 topology-to-tensor injection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Dict, Optional
import logging
from pathlib import Path

import torch

from src.domain.services.torch_serialization import load_trusted_torch_artifact

from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.tensor_contract import GraphTensorContract
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.topology_projection_alignment import (
    TopologyProjectionCompiler,
    TopologyProjectionDiagnostics,
)


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
        cache_dir: str | None = None,
        candidate_identity_mode: str = "fixed_vocab_bridge",
        process_state_mask_enabled: bool = False,
        process_state_mask_source: str = "lifecycle_active_set",
        process_state_mask_include_direct_successors: bool = True,
        process_state_mask_include_active_candidates: bool = True,
        process_state_mask_relaxed_lookback_events: int = 8,
        process_state_mask_relaxed_max_depth: int = 1,
        process_state_mask_relaxed_max_cardinality_ratio: float = 0.35,
        process_state_mask_relaxed_suppress_completed: bool = True,
        process_state_mask_relaxed_anchor_policy: str = "open_successors",
        process_state_mask_relaxed_loop_policy: str = "keep_direct_successor_repeats",
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
        self.topology_gateway_mode = self._resolve_topology_gateway_mode(self.graph_feature_mapping)
        self.topology_projection_config = self._resolve_topology_projection_config(self.graph_feature_mapping)
        self._quality_warned_keys: set[tuple[str, str, str]] = set()
        self._missing_asof_warned_keys: set[tuple[str, str, str]] = set()
        self._topology_projection_warned_keys: set[tuple[str, str, str]] = set()
        self.cache_policy = self._resolve_cache_policy(cache_policy)
        self._dto_cache: dict[tuple[Any, ...], ProcessStructureDTO | None] = {}
        self._topology_cache: dict[tuple[Any, ...], Dict[str, Any]] = {}
        self._cache_diagnostics: dict[str, int] = {
            "dto_cache_hits": 0,
            "dto_cache_misses": 0,
            "topology_cache_hits": 0,
            "topology_cache_misses": 0,
        }
        self._resolved_snapshot_identities: set[tuple[str, str, str | None, str | None]] = set()
        self._dto_cache_max_entries = 32768
        self._topology_cache_max_entries = 512
        self._topology_disk_cache_schema = 4
        self._topology_disk_cache_dir = self._resolve_topology_disk_cache_dir(cache_dir)
        self.candidate_identity_mode = str(candidate_identity_mode or "fixed_vocab_bridge").strip().lower()
        self.process_state_mask_enabled = bool(process_state_mask_enabled)
        self.process_state_mask_source = str(process_state_mask_source or "lifecycle_active_set").strip().lower()
        if self.process_state_mask_source not in {
            "lifecycle_active_set",
            "event_active_candidates",
            "relaxed_reachability",
        }:
            self.process_state_mask_source = "lifecycle_active_set"
        self.process_state_mask_include_direct_successors = bool(process_state_mask_include_direct_successors)
        self.process_state_mask_include_active_candidates = bool(process_state_mask_include_active_candidates)
        self.process_state_mask_relaxed_lookback_events = max(1, int(process_state_mask_relaxed_lookback_events))
        self.process_state_mask_relaxed_max_depth = max(1, int(process_state_mask_relaxed_max_depth))
        self.process_state_mask_relaxed_max_cardinality_ratio = min(
            1.0,
            max(0.01, float(process_state_mask_relaxed_max_cardinality_ratio)),
        )
        self.process_state_mask_relaxed_suppress_completed = bool(process_state_mask_relaxed_suppress_completed)
        self.process_state_mask_relaxed_anchor_policy = str(
            process_state_mask_relaxed_anchor_policy or "open_successors"
        ).strip().lower()
        if self.process_state_mask_relaxed_anchor_policy not in {"recent_prefix", "open_successors"}:
            self.process_state_mask_relaxed_anchor_policy = "open_successors"
        self.process_state_mask_relaxed_loop_policy = str(
            process_state_mask_relaxed_loop_policy or "keep_direct_successor_repeats"
        ).strip().lower()
        if self.process_state_mask_relaxed_loop_policy not in {"keep_direct_successor_repeats"}:
            self.process_state_mask_relaxed_loop_policy = "keep_direct_successor_repeats"

    def cache_diagnostics(self) -> dict[str, int]:
        return {
            **{key: int(value) for key, value in self._cache_diagnostics.items()},
            "dto_cache_entries": int(len(self._dto_cache)),
            "topology_cache_entries": int(len(self._topology_cache)),
            "unique_snapshot_identities": int(len(self._resolved_snapshot_identities)),
        }

    def build_graph(self, prefix: PrefixSlice) -> GraphTensorContract:
        """Build baseline contract and inject OOS mask when topology is available."""
        contract = super().build_graph(prefix)
        mapping = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
        stats_enabled = bool(mapping.get("enabled", False))
        as_of_ts = (
            self._resolve_as_of_timestamp(prefix)
            if self.stats_time_policy == "strict_asof"
            else None
        )

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
        candidate_allowed_mask = torch.zeros(
            int(compiled.get("candidate_count", 0) or 0),
            dtype=torch.bool,
        )
        active_mask = torch.zeros(num_classes, dtype=torch.bool)
        active_candidates = self._extract_active_candidates(
            prefix.prefix_events[-1].extra if prefix.prefix_events else {},
        )
        relaxed_candidates: set[str] = set()
        relaxed_diagnostics: dict[str, int] = {
            "raw_candidate_count": 0,
            "suppressed_completed_count": 0,
            "final_candidate_count": 0,
            "target_suppressed_by_completed_filter_count": 0,
            "anchor_count": 0,
            "skipped_closed_anchor_count": 0,
        }
        if self.process_state_mask_enabled and self.process_state_mask_include_active_candidates:
            if self.process_state_mask_source == "relaxed_reachability":
                process_state_candidates, relaxed_diagnostics = self._relaxed_reachability_candidates(
                    prefix=prefix,
                    compiled=compiled,
                    activity_vocab=activity_vocab,
                    active_candidates=active_candidates,
                )
                relaxed_candidates = set(process_state_candidates)
            else:
                process_state_candidates = self._process_state_active_candidates(
                    prefix=prefix,
                    compiled=compiled,
                    activity_vocab=activity_vocab,
                )
            active_candidates.update(process_state_candidates)
        for token in active_candidates:
            idx = activity_vocab.get(token)
            if idx is None:
                idx = activity_vocab.get(str(token).strip())
            if idx is not None:
                active_mask[int(idx)] = True
        last_activity_idx = activity_vocab.get(last_activity)
        if last_activity_idx is None:
            last_activity_idx = activity_vocab.get(last_activity.strip())
        contract["prefix_last_activity_idx"] = torch.tensor(
            [int(last_activity_idx) if last_activity_idx is not None else -1],
            dtype=torch.long,
        )
        if last_activity_idx is not None and self.process_state_mask_include_direct_successors:
            cached_mask = compiled["allowed_masks_by_src"].get(int(last_activity_idx))
            if isinstance(cached_mask, torch.Tensor):
                allowed_mask = cached_mask.clone()
        if self.process_state_mask_include_direct_successors:
            for src_idx in self._candidate_indices_for_token(compiled=compiled, token=last_activity):
                cached_candidate_mask = compiled.get("candidate_allowed_masks_by_src", {}).get(int(src_idx))
                if isinstance(cached_candidate_mask, torch.Tensor):
                    candidate_allowed_mask = torch.logical_or(candidate_allowed_mask, cached_candidate_mask)
        allowed_mask = torch.logical_or(allowed_mask, active_mask)
        for token in active_candidates:
            for candidate_idx in self._candidate_indices_for_token(compiled=compiled, token=token):
                if 0 <= int(candidate_idx) < int(candidate_allowed_mask.numel()):
                    candidate_allowed_mask[int(candidate_idx)] = True

        # Reuse cached structural tensors across prefixes to avoid per-prefix RAM duplication.
        contract["structural_edge_index"] = compiled["structural_edge_index"]
        contract["structural_edge_weight"] = compiled["structural_edge_weight"]
        struct_node_to_class_index = compiled.get("struct_node_to_class_index")
        if isinstance(struct_node_to_class_index, torch.Tensor):
            contract["struct_node_to_class_index"] = struct_node_to_class_index
            contract["struct_prefix_state_x"] = self._build_struct_prefix_state_x(
                prefix=prefix,
                activity_vocab=activity_vocab,
                struct_node_to_class_index=struct_node_to_class_index,
                active_candidates=active_candidates,
            )
        struct_node_to_candidate_index = compiled.get("struct_node_to_candidate_index")
        if isinstance(struct_node_to_candidate_index, torch.Tensor):
            contract["struct_node_to_candidate_index"] = struct_node_to_candidate_index
        candidate_class_index = compiled.get("candidate_class_index")
        if isinstance(candidate_class_index, torch.Tensor):
            contract["candidate_class_index"] = candidate_class_index
        candidate_is_unseen = compiled.get("candidate_is_unseen")
        if isinstance(candidate_is_unseen, torch.Tensor):
            contract["candidate_is_unseen"] = candidate_is_unseen
        contract["candidate_ids"] = tuple(compiled.get("candidate_ids", ()))
        contract["candidate_labels"] = tuple(compiled.get("candidate_labels", ()))
        struct_x = compiled.get("struct_x")
        if isinstance(struct_x, torch.Tensor):
            contract["struct_x"] = struct_x
        diagnostics = compiled.get("topology_projection_diagnostics")
        if isinstance(diagnostics, TopologyProjectionDiagnostics):
            self._attach_topology_projection_summary(contract=contract, diagnostics=diagnostics)
        elif isinstance(diagnostics, dict):
            self._attach_topology_projection_summary(
                contract=contract,
                diagnostics=TopologyProjectionDiagnostics.from_dict(diagnostics),
            )
        contract["allowed_target_mask"] = allowed_mask
        contract["candidate_allowed_target_mask"] = candidate_allowed_mask
        contract["process_state_mask_active_candidate_count"] = torch.tensor(
            [int(len(active_candidates))],
            dtype=torch.long,
        )
        contract["process_state_mask_relaxed_candidate_count"] = torch.tensor(
            [int(len(relaxed_candidates))],
            dtype=torch.long,
        )
        contract["process_state_mask_relaxed_raw_candidate_count"] = torch.tensor(
            [int(relaxed_diagnostics.get("raw_candidate_count", 0))],
            dtype=torch.long,
        )
        contract["process_state_mask_relaxed_suppressed_completed_count"] = torch.tensor(
            [int(relaxed_diagnostics.get("suppressed_completed_count", 0))],
            dtype=torch.long,
        )
        contract["process_state_mask_relaxed_final_candidate_count"] = torch.tensor(
            [int(relaxed_diagnostics.get("final_candidate_count", len(relaxed_candidates)))],
            dtype=torch.long,
        )
        contract["process_state_mask_target_suppressed_by_completed_filter_count"] = torch.tensor(
            [int(relaxed_diagnostics.get("target_suppressed_by_completed_filter_count", 0))],
            dtype=torch.long,
        )
        contract["process_state_mask_relaxed_anchor_count"] = torch.tensor(
            [int(relaxed_diagnostics.get("anchor_count", 0))],
            dtype=torch.long,
        )
        contract["process_state_mask_relaxed_skipped_closed_anchor_count"] = torch.tensor(
            [int(relaxed_diagnostics.get("skipped_closed_anchor_count", 0))],
            dtype=torch.long,
        )
        if prefix.target_event is not None:
            contract["target_label"] = str(prefix.target_event.activity_id)
        return contract

    @staticmethod
    def _candidate_indices_for_token(*, compiled: Dict[str, Any], token: str) -> list[int]:
        normalized = str(token).strip()
        if not normalized:
            return []
        by_label = compiled.get("candidate_indices_by_label", {})
        by_id = compiled.get("candidate_indices_by_id", {})
        result: list[int] = []
        if isinstance(by_label, dict):
            result.extend(int(item) for item in by_label.get(normalized, []))
        if isinstance(by_id, dict):
            result.extend(int(item) for item in by_id.get(normalized, []))
        return sorted(set(result))

    def _build_struct_prefix_state_x(
        self,
        *,
        prefix: PrefixSlice,
        activity_vocab: Dict[str, int],
        struct_node_to_class_index: torch.Tensor,
        active_candidates: set[str] | None = None,
    ) -> torch.Tensor:
        """Project observed prefix state onto structural node rows."""
        node_to_class = struct_node_to_class_index.reshape(-1).to(dtype=torch.long)
        state = torch.zeros((int(node_to_class.numel()), 6), dtype=torch.float32)
        prefix_length = int(len(prefix.prefix_events))
        if prefix_length <= 0:
            return state

        target_feature = self.feature_encoder.activity_feature_name
        counts_by_class: dict[int, int] = {}
        last_pos_by_class: dict[int, int] = {}
        for pos, event in enumerate(prefix.prefix_events):
            token = str(
                self.feature_encoder.resolve_event_feature(
                    event_extra=event.extra,
                    feature_name=target_feature,
                    default=event.activity_id,
                )
            )
            class_idx = activity_vocab.get(token)
            if class_idx is None:
                class_idx = activity_vocab.get(token.strip())
            if class_idx is None:
                continue
            class_idx_int = int(class_idx)
            counts_by_class[class_idx_int] = counts_by_class.get(class_idx_int, 0) + 1
            last_pos_by_class[class_idx_int] = int(pos)

        last_event = prefix.prefix_events[-1]
        last_token = str(
            self.feature_encoder.resolve_event_feature(
                event_extra=last_event.extra,
                feature_name=target_feature,
                default=last_event.activity_id,
            )
        )
        last_class_idx = activity_vocab.get(last_token)
        if last_class_idx is None:
            last_class_idx = activity_vocab.get(last_token.strip())

        active_class_indices: set[int] = set()
        active_tokens = set(active_candidates or set())
        active_tokens.update(self._extract_active_candidates(last_event.extra))
        for token in active_tokens:
            active_idx = activity_vocab.get(token)
            if active_idx is None:
                active_idx = activity_vocab.get(str(token).strip())
            if active_idx is not None:
                active_class_indices.add(int(active_idx))

        for row_idx, class_idx in enumerate(node_to_class.tolist()):
            if class_idx < 0:
                continue
            count = int(counts_by_class.get(int(class_idx), 0))
            last_pos = last_pos_by_class.get(int(class_idx))
            if count > 0:
                state[row_idx, 0] = float(math.log1p(count))
                state[row_idx, 1] = 1.0
            if last_class_idx is not None and int(class_idx) == int(last_class_idx):
                state[row_idx, 2] = 1.0
            if last_pos is not None:
                state[row_idx, 3] = float((int(last_pos) + 1) / prefix_length)
                state[row_idx, 4] = float(1.0 / (1.0 + (prefix_length - 1 - int(last_pos))))
            if int(class_idx) in active_class_indices:
                state[row_idx, 5] = 1.0
        return state

    def _process_state_active_candidates(
        self,
        *,
        prefix: PrefixSlice,
        compiled: Dict[str, Any],
        activity_vocab: Dict[str, int],
    ) -> set[str]:
        if self.process_state_mask_source == "event_active_candidates":
            if not prefix.prefix_events:
                return set()
            return self._extract_active_candidates(prefix.prefix_events[-1].extra)
        if self.process_state_mask_source == "relaxed_reachability":
            candidates, _diagnostics = self._relaxed_reachability_candidates(
                prefix=prefix,
                compiled=compiled,
                activity_vocab=activity_vocab,
                active_candidates=set(),
            )
            return candidates
        return self._active_candidates_from_lifecycle(prefix)

    def _recent_prefix_activity_tokens(self, prefix: PrefixSlice, *, limit: int) -> list[str]:
        tokens: list[str] = []
        target_feature = self.feature_encoder.activity_feature_name
        for event in reversed(prefix.prefix_events):
            extra = event.extra if isinstance(event.extra, dict) else {}
            token = str(
                self.feature_encoder.resolve_event_feature(
                    event_extra=extra,
                    feature_name=target_feature,
                    default=event.activity_id,
                )
            ).strip()
            if token:
                tokens.append(token)
            if len(tokens) >= int(limit):
                break
        return tokens

    def _prefix_process_state(self, prefix: PrefixSlice) -> dict[str, set[str]]:
        completed_tokens: set[str] = set()
        active_tokens: set[str] = set()
        completed_instance_ids: set[str] = set()
        active_instance_ids: set[str] = set()
        active_token_by_instance: dict[str, str] = {}
        target_feature = self.feature_encoder.activity_feature_name
        for event in prefix.prefix_events:
            extra = event.extra if isinstance(event.extra, dict) else {}
            lifecycle = str(extra.get("lifecycle:transition") or event.lifecycle or "").strip().lower()
            token = str(
                self.feature_encoder.resolve_event_feature(
                    event_extra=extra,
                    feature_name=target_feature,
                    default=event.activity_id,
                )
            ).strip()
            if not token:
                continue
            instance_id = str(
                extra.get("sim:activity_instance_id")
                or event.activity_instance_id
                or f"{token}:{event.position_in_trace}"
            ).strip()
            if lifecycle in {"assign", "start"}:
                active_tokens.add(token)
                if instance_id:
                    active_instance_ids.add(instance_id)
                    active_token_by_instance[instance_id] = token
            elif not lifecycle or lifecycle in {"complete", "completed"}:
                completed_tokens.add(token)
                if instance_id:
                    completed_instance_ids.add(instance_id)
                    active_instance_ids.discard(instance_id)
                    previous_token = active_token_by_instance.pop(instance_id, None)
                    if previous_token is not None and previous_token not in active_token_by_instance.values():
                        active_tokens.discard(previous_token)
        return {
            "completed_tokens": completed_tokens,
            "active_tokens": active_tokens,
            "completed_instance_ids": completed_instance_ids,
            "active_instance_ids": active_instance_ids,
        }

    def _direct_successor_labels_for_token(self, *, compiled: Dict[str, Any], token: str) -> set[str]:
        candidate_masks_by_src = compiled.get("candidate_allowed_masks_by_src", {})
        candidate_labels = tuple(str(item) for item in compiled.get("candidate_labels", ()))
        result: set[str] = set()
        for src_candidate_idx in self._candidate_indices_for_token(compiled=compiled, token=token):
            mask = (
                candidate_masks_by_src.get(int(src_candidate_idx))
                if isinstance(candidate_masks_by_src, dict)
                else None
            )
            if not isinstance(mask, torch.Tensor):
                continue
            for dst_idx_raw in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
                dst_idx = int(dst_idx_raw)
                if 0 <= dst_idx < len(candidate_labels):
                    result.add(str(candidate_labels[dst_idx]))
        return result

    def _candidate_has_loop_or_rework_evidence(self, *, compiled: Dict[str, Any], token: str) -> bool:
        token = str(token).strip()
        if not token:
            return False
        return token in self._direct_successor_labels_for_token(compiled=compiled, token=token)

    def _relaxed_reachability_candidates(
        self,
        *,
        prefix: PrefixSlice,
        compiled: Dict[str, Any],
        activity_vocab: Dict[str, int],
        active_candidates: set[str],
    ) -> tuple[set[str], dict[str, int]]:
        diagnostics = {
            "raw_candidate_count": 0,
            "suppressed_completed_count": 0,
            "final_candidate_count": 0,
            "target_suppressed_by_completed_filter_count": 0,
            "anchor_count": 0,
            "skipped_closed_anchor_count": 0,
        }
        anchors = self._recent_prefix_activity_tokens(
            prefix,
            limit=self.process_state_mask_relaxed_lookback_events,
        )
        candidate_masks_by_src = compiled.get("candidate_allowed_masks_by_src", {})
        candidate_labels = tuple(str(item) for item in compiled.get("candidate_labels", ()))
        prefix_state = self._prefix_process_state(prefix)
        completed_tokens = set(prefix_state["completed_tokens"])
        active_tokens = set(prefix_state["active_tokens"])
        active_tokens.update(str(item).strip() for item in active_candidates if str(item).strip())
        last_token = anchors[0] if anchors else ""
        last_direct_successors = self._direct_successor_labels_for_token(
            compiled=compiled,
            token=last_token,
        )
        result: set[str] = set()
        for anchor in anchors:
            anchor_direct_successors = self._direct_successor_labels_for_token(
                compiled=compiled,
                token=anchor,
            )
            anchor_has_loop_evidence = str(anchor).strip() in anchor_direct_successors
            if (
                self.process_state_mask_relaxed_anchor_policy == "open_successors"
                and anchor_direct_successors
                and anchor_direct_successors.issubset(completed_tokens)
                and not anchor_direct_successors.intersection(active_tokens)
                and not anchor_has_loop_evidence
            ):
                diagnostics["skipped_closed_anchor_count"] += 1
                continue
            diagnostics["anchor_count"] += 1
            frontier = self._candidate_indices_for_token(compiled=compiled, token=anchor)
            visited = {int(item) for item in frontier}
            for _depth in range(self.process_state_mask_relaxed_max_depth):
                next_frontier: list[int] = []
                for src_candidate_idx in frontier:
                    mask = (
                        candidate_masks_by_src.get(int(src_candidate_idx))
                        if isinstance(candidate_masks_by_src, dict)
                        else None
                    )
                    if not isinstance(mask, torch.Tensor):
                        continue
                    for dst_idx_raw in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
                        dst_idx = int(dst_idx_raw)
                        if dst_idx in visited:
                            continue
                        visited.add(dst_idx)
                        next_frontier.append(dst_idx)
                        if 0 <= dst_idx < len(candidate_labels):
                            result.add(str(candidate_labels[dst_idx]))
                frontier = next_frontier
                if not frontier:
                    break
        raw_result = set(result)
        diagnostics["raw_candidate_count"] = int(len(raw_result))
        if self.process_state_mask_relaxed_suppress_completed:
            keep_completed = set(last_direct_successors)
            keep_completed.update(active_tokens)
            filtered = {
                token
                for token in raw_result
                if token not in completed_tokens
                or token in keep_completed
                or self._candidate_has_loop_or_rework_evidence(compiled=compiled, token=token)
            }
            diagnostics["suppressed_completed_count"] = int(len(raw_result) - len(filtered))
        else:
            filtered = raw_result
        target_token = ""
        if prefix.target_event is not None:
            target_extra = prefix.target_event.extra if isinstance(prefix.target_event.extra, dict) else {}
            target_token = str(
                self.feature_encoder.resolve_event_feature(
                    event_extra=target_extra,
                    feature_name=self.feature_encoder.activity_feature_name,
                    default=prefix.target_event.activity_id,
                )
            ).strip()
        if target_token and target_token in raw_result and target_token not in filtered:
            diagnostics["target_suppressed_by_completed_filter_count"] = 1
        final = self._cap_relaxed_candidates(result=filtered, compiled=compiled)
        diagnostics["final_candidate_count"] = int(len(final))
        return final, diagnostics

    def _cap_relaxed_candidates(self, *, result: set[str], compiled: Dict[str, Any]) -> set[str]:
        candidate_count = int(compiled.get("candidate_count", 0) or 0)
        if candidate_count <= 0:
            candidate_count = len(tuple(compiled.get("candidate_labels", ())))
        if candidate_count <= 0:
            return set(result)
        max_count = max(1, int(math.ceil(candidate_count * self.process_state_mask_relaxed_max_cardinality_ratio)))
        if len(result) <= max_count:
            return set(result)
        return set(sorted(result)[:max_count])

    def _active_candidates_from_lifecycle(self, prefix: PrefixSlice) -> set[str]:
        active_by_instance: dict[str, str] = {}
        target_feature = self.feature_encoder.activity_feature_name
        for event in prefix.prefix_events:
            extra = event.extra if isinstance(event.extra, dict) else {}
            activity = str(
                self.feature_encoder.resolve_event_feature(
                    event_extra=extra,
                    feature_name=target_feature,
                    default=event.activity_id,
                )
            ).strip()
            if not activity:
                continue
            lifecycle = str(extra.get("lifecycle:transition") or event.lifecycle or "").strip().lower()
            instance_id = str(
                extra.get("sim:activity_instance_id")
                or event.activity_instance_id
                or f"{activity}:{event.position_in_trace}"
            ).strip()
            if not instance_id:
                continue
            if lifecycle in {"assign", "start"}:
                active_by_instance[instance_id] = activity
            elif lifecycle == "complete":
                active_by_instance.pop(instance_id, None)
        return {activity for activity in active_by_instance.values() if activity}

    @staticmethod
    def _extract_active_candidates(event_extra: Any) -> set[str]:
        if not isinstance(event_extra, dict):
            return set()
        candidates: set[str] = set()

        raw_counts = event_extra.get("active_activity_counts_after_complete")
        if isinstance(raw_counts, dict):
            for key, value in raw_counts.items():
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    continue
                token = str(key).strip()
                if token and count > 0:
                    candidates.add(token)

        raw_list = event_extra.get("active_activities_after_complete")
        if isinstance(raw_list, (list, tuple, set)):
            for item in raw_list:
                token = str(item).strip()
                if token:
                    candidates.add(token)
        elif isinstance(raw_list, str):
            for item in raw_list.split(","):
                token = str(item).strip()
                if token:
                    candidates.add(token)

        return candidates

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

    def _should_cache_dto_lookup(self, *, as_of_ts: datetime | None) -> bool:
        if self.cache_policy not in {"dto", "full"}:
            return False
        if self.stats_time_policy == "strict_asof" and isinstance(as_of_ts, datetime):
            mapping = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
            if bool(mapping.get("enabled", False)):
                return False
        return True

    def _resolve_dto(
        self,
        *,
        as_of_ts: datetime | None,
        candidate_versions: list[str],
    ) -> ProcessStructureDTO | None:
        cache_key: tuple[Any, ...] | None = None
        if self._should_cache_dto_lookup(as_of_ts=as_of_ts):
            mapping = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
            stats_enabled = bool(mapping.get("enabled", False))
            resolved_as_of = as_of_ts if stats_enabled else None
            cache_key = (
                self.process_name or "__auto__",
                tuple(str(item).strip() for item in candidate_versions),
                resolved_as_of.isoformat() if isinstance(resolved_as_of, datetime) else None,
            )
            if cache_key in self._dto_cache:
                self._cache_diagnostics["dto_cache_hits"] += 1
                return self._dto_cache[cache_key]
        self._cache_diagnostics["dto_cache_misses"] += 1

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
                self._track_snapshot_identity(dto)
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
                    self._track_snapshot_identity(dto)
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

    def _track_snapshot_identity(self, dto: ProcessStructureDTO) -> None:
        snapshot = self._stats_snapshot_metadata(dto)
        self._resolved_snapshot_identities.add(
            (
                self.process_name or "__auto__",
                str(dto.version),
                self._clean_optional_text(snapshot.get("knowledge_version")) or None,
                self._clean_optional_text(snapshot.get("as_of_ts")) or None,
            )
        )

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
        normalized_nodes = self._topology_nodes_fingerprint_payload(dto)
        graph_mapping_fingerprint = self._graph_mapping_fingerprint()
        vocab_fingerprint = self._activity_vocab_fingerprint(activity_vocab)
        return (
            self.process_name or "__auto__",
            "topology_native_candidate_identity_v1",
            str(dto.version),
            self.candidate_identity_mode,
            self._clean_optional_text(snapshot.get("knowledge_version")),
            self._clean_optional_text(snapshot.get("as_of_ts")),
            bool(stats_allowed),
            vocab_fingerprint,
            graph_mapping_fingerprint,
            normalized_edges,
            normalized_nodes,
        )

    @staticmethod
    def _activity_vocab_fingerprint(activity_vocab: Dict[str, int]) -> str:
        pairs = sorted((str(key), int(value)) for key, value in activity_vocab.items())
        payload = json.dumps(pairs, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _topology_nodes_fingerprint_payload(dto: ProcessStructureDTO) -> tuple[tuple[str, str, str, str, str], ...]:
        nodes = dto.nodes or []
        normalized: list[tuple[str, str, str, str, str]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            normalized.append(
                (
                    node_id,
                    str(node.get("bpmn_tag", "")).strip(),
                    str(node.get("type", "")).strip(),
                    str(node.get("activity_type", "")).strip(),
                    str(node.get("logical_type", "")).strip(),
                )
            )
        return tuple(sorted(normalized))

    def _graph_mapping_fingerprint(self) -> str:
        block = self.graph_feature_mapping if isinstance(self.graph_feature_mapping, dict) else {}
        payload = {
            "enabled": bool(block.get("enabled", False)),
            "node_specs": self._node_specs(),
            "edge_weight": self._edge_weight_spec(),
            "quality_gate": self.stats_quality_gate,
            "time_policy": self.stats_time_policy,
            "missing_asof_policy": self.on_missing_asof_snapshot,
            "topology_gateway_mode": self.topology_gateway_mode,
            "topology_projection_config": self.topology_projection_config,
        }
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _resolve_topology_gateway_mode(graph_feature_mapping: Dict[str, Any]) -> str:
        block = graph_feature_mapping if isinstance(graph_feature_mapping, dict) else {}
        projection = block.get("topology_projection", {})
        raw = None
        if isinstance(projection, dict):
            raw = projection.get("gateway_mode")
        if raw is None:
            raw = block.get("gateway_mode")
        mode = str(raw or "preserve").strip().lower()
        if mode not in {"preserve", "collapse_for_prediction"}:
            return "preserve"
        return mode

    @staticmethod
    def _resolve_topology_projection_config(graph_feature_mapping: Dict[str, Any]) -> Dict[str, Any]:
        block = graph_feature_mapping if isinstance(graph_feature_mapping, dict) else {}
        projection = block.get("topology_projection", {})
        cfg = dict(projection) if isinstance(projection, dict) else {}
        raw_diagnostics_enabled = cfg.get("diagnostics_enabled", True)
        if isinstance(raw_diagnostics_enabled, str):
            diagnostics_enabled = raw_diagnostics_enabled.strip().lower() not in {"0", "false", "no", "off"}
        else:
            diagnostics_enabled = bool(raw_diagnostics_enabled)
        raw_on_fail = str(cfg.get("on_fail", "warn")).strip().lower() or "warn"
        alias_map = {
            "warning": "warn",
            "allow_with_warning": "warn",
            "ignore_with_warning": "warn",
            "disabled": "disable_struct",
            "disable_stats": "disable_struct",
            "disable": "disable_struct",
        }
        on_fail = alias_map.get(raw_on_fail, raw_on_fail)
        if on_fail not in {"warn", "raise", "disable_struct"}:
            on_fail = "warn"
        return {
            "diagnostics_enabled": diagnostics_enabled,
            "on_fail": on_fail,
        }

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
                self._cache_diagnostics["topology_cache_hits"] += 1
                return cached
            cached_disk = self._load_compiled_topology_from_disk(cache_key)
            if isinstance(cached_disk, dict):
                self._cache_diagnostics["topology_cache_hits"] += 1
                self._cache_put(
                    cache=self._topology_cache,
                    key=cache_key,
                    value=cached_disk,
                    max_entries=self._topology_cache_max_entries,
                )
                return cached_disk
            self._cache_diagnostics["topology_cache_misses"] += 1

        num_classes = len(activity_vocab)
        if self.candidate_identity_mode == "topology_native":
            candidate_ids, candidate_labels, candidate_class_values = self._topology_native_candidates(
                dto=dto,
                activity_vocab=activity_vocab,
            )
        else:
            ordered_vocab = sorted(activity_vocab.items(), key=lambda item: int(item[1]))
            candidate_ids = tuple(str(label) for label, _idx in ordered_vocab)
            candidate_labels = tuple(str(label) for label, _idx in ordered_vocab)
            candidate_class_values = [int(idx) for _label, idx in ordered_vocab]
        candidate_count = len(candidate_ids)
        candidate_id_to_index = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
        candidate_indices_by_label: dict[str, list[int]] = {}
        candidate_indices_by_id: dict[str, list[int]] = {}
        for idx, (candidate_id, candidate_label) in enumerate(zip(candidate_ids, candidate_labels)):
            candidate_indices_by_id.setdefault(candidate_id, []).append(idx)
            candidate_indices_by_label.setdefault(candidate_label, []).append(idx)
        
        allowed_masks_by_src: Dict[int, torch.Tensor] = {}
        candidate_allowed_masks_by_src: Dict[int, torch.Tensor] = {}
        structural_src: list[int] = []
        structural_dst: list[int] = []
        structural_weight: list[float] = []
        edge_stats = dto.edge_statistics or {}
        edge_weight_spec = self._edge_weight_spec() if stats_allowed else None
        edge_weight_index = self._edge_stats_index(dto) if stats_allowed else {}
        if not stats_allowed:
            edge_stats = {}

        if self.candidate_identity_mode == "topology_native":
            projection_result = TopologyProjectionCompiler(gateway_mode=self.topology_gateway_mode).project(
                dto=dto,
                activity_vocab={candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)},
            )
        else:
            projection_result = TopologyProjectionCompiler(gateway_mode=self.topology_gateway_mode).project(
                dto=dto,
                activity_vocab=activity_vocab,
            )

        projected_edge_paths = projection_result.projected_edge_paths
        for src, dst in projected_edge_paths:
            src_token = str(src).strip()
            dst_token = str(dst).strip()
            src_idx = activity_vocab.get(src_token)
            dst_idx = activity_vocab.get(dst_token)
            src_candidate_idx = candidate_id_to_index.get(src_token)
            dst_candidate_idx = candidate_id_to_index.get(dst_token)

            if self.candidate_identity_mode == "topology_native":
                if src_candidate_idx is None or dst_candidate_idx is None:
                    continue
                src_idx_int = int(src_idx) if src_idx is not None else -1
                dst_idx_int = int(dst_idx) if dst_idx is not None else -1
                structural_src.append(int(src_candidate_idx))
                structural_dst.append(int(dst_candidate_idx))
            else:
                if src_idx is None or dst_idx is None:
                    continue
                src_idx_int = int(src_idx)
                dst_idx_int = int(dst_idx)
                structural_src.append(src_idx_int)
                structural_dst.append(dst_idx_int)

            if edge_weight_spec is not None:
                edge_key = f"{src_token}|||{dst_token}"
                if edge_key in edge_weight_index:
                    structural_weight.append(float(edge_weight_index.get(edge_key, edge_weight_spec.get("default", 1.0))))
                else:
                    structural_weight.append(
                        self._projected_edge_weight(
                            paths=projected_edge_paths.get((src_token, dst_token), []),
                            edge_weight_index=edge_weight_index,
                            edge_weight_spec=edge_weight_spec,
                        )
                    )
            else:
                stats = edge_stats.get((src, dst), {})
                structural_weight.append(float(stats.get("count", 1.0)))

            if src_idx_int >= 0 and dst_idx_int >= 0:
                row_mask = allowed_masks_by_src.get(src_idx_int)
                if row_mask is None:
                    row_mask = torch.zeros(num_classes, dtype=torch.bool)
                    allowed_masks_by_src[src_idx_int] = row_mask
                row_mask[dst_idx_int] = True

            if src_candidate_idx is not None and dst_candidate_idx is not None:
                candidate_row_mask = candidate_allowed_masks_by_src.get(int(src_candidate_idx))
                if candidate_row_mask is None:
                    candidate_row_mask = torch.zeros(candidate_count, dtype=torch.bool)
                    candidate_allowed_masks_by_src[int(src_candidate_idx)] = candidate_row_mask
                candidate_row_mask[int(dst_candidate_idx)] = True

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

        candidate_class_index = torch.tensor(candidate_class_values, dtype=torch.long)
        candidate_is_unseen = candidate_class_index < 0

        if self.candidate_identity_mode == "topology_native":
            struct_node_to_class_index = candidate_class_index.clone()
            struct_node_to_candidate_index = torch.arange(candidate_count, dtype=torch.long)
            struct_x = (
                self._build_candidate_struct_x(
                    dto=dto,
                    candidate_labels=candidate_labels,
                    candidate_class_index=candidate_class_index,
                    activity_vocab=activity_vocab,
                )
                if stats_allowed
                else None
            )
        else:
            struct_node_to_class_index = torch.arange(num_classes, dtype=torch.long)
            struct_node_to_candidate_index = torch.arange(num_classes, dtype=torch.long)
            struct_x = self._build_struct_x(dto=dto, activity_vocab=activity_vocab) if stats_allowed else None

        structural_edge_index_max = (
            int(structural_edge_index.max().item())
            if isinstance(structural_edge_index, torch.Tensor) and structural_edge_index.numel() > 0
            else None
        )
        struct_x_rows = int(struct_x.size(0)) if isinstance(struct_x, torch.Tensor) and struct_x.dim() >= 1 else None
        diagnostics = projection_result.diagnostics.with_structural_payload(
            struct_x_rows=struct_x_rows,
            structural_edge_index_max=structural_edge_index_max,
        )
        if (
            (not diagnostics.is_aligned)
            and (
                self.topology_projection_config.get("on_fail") == "raise"
                or bool(self.topology_projection_config.get("diagnostics_enabled", True))
            )
        ):
            self._handle_topology_projection_diagnostics(diagnostics=diagnostics)
        if self.topology_projection_config.get("on_fail") == "disable_struct" and not diagnostics.is_aligned:
            allowed_masks_by_src = {}
            structural_edge_index = torch.zeros((2, 0), dtype=torch.long)
            structural_edge_weight = torch.zeros((0,), dtype=torch.float32)
            struct_node_to_class_index = torch.arange(num_classes, dtype=torch.long)
            struct_x = None
        compiled = {
            "allowed_masks_by_src": allowed_masks_by_src,
            "candidate_allowed_masks_by_src": candidate_allowed_masks_by_src,
            "structural_edge_index": structural_edge_index,
            "structural_edge_weight": structural_edge_weight,
            "struct_node_to_class_index": struct_node_to_class_index,
            "struct_node_to_candidate_index": struct_node_to_candidate_index,
            "candidate_ids": tuple(candidate_ids),
            "candidate_labels": tuple(candidate_labels),
            "candidate_class_index": candidate_class_index,
            "candidate_is_unseen": candidate_is_unseen,
            "candidate_indices_by_label": candidate_indices_by_label,
            "candidate_indices_by_id": candidate_indices_by_id,
            "candidate_count": int(candidate_count),
            "struct_x": struct_x,
            "topology_projection_diagnostics": diagnostics,
        }
        if cache_key is not None:
            self._cache_put(
                cache=self._topology_cache,
                key=cache_key,
                value=compiled,
                max_entries=self._topology_cache_max_entries,
            )
            self._save_compiled_topology_to_disk(cache_key=cache_key, compiled=compiled)
        return compiled

    @staticmethod
    def _node_label(node: Dict[str, Any]) -> str:
        for key in ("activity_label", "label", "name", "id"):
            value = str(node.get(key, "")).strip()
            if value:
                return value
        return ""

    def _topology_native_candidates(
        self,
        *,
        dto: ProcessStructureDTO,
        activity_vocab: Dict[str, int],
    ) -> tuple[list[str], list[str], list[int]]:
        nodes = list(dto.nodes or [])
        node_by_id = {str(node.get("id", "")).strip(): node for node in nodes if isinstance(node, dict)}
        projection_result = TopologyProjectionCompiler(gateway_mode=self.topology_gateway_mode).project(
            dto=dto,
            activity_vocab={},
        )
        candidate_ids: set[str] = set()
        if self.topology_gateway_mode == "collapse_for_prediction" and projection_result.prediction_nodes:
            candidate_ids.update(projection_result.prediction_nodes)
        else:
            for src, dst in projection_result.projected_edge_paths:
                if str(src).strip():
                    candidate_ids.add(str(src).strip())
                if str(dst).strip():
                    candidate_ids.add(str(dst).strip())
        if not candidate_ids:
            for src, dst in dto.allowed_edges:
                if str(src).strip():
                    candidate_ids.add(str(src).strip())
                if str(dst).strip():
                    candidate_ids.add(str(dst).strip())

        ordered_ids = sorted(candidate_ids)
        labels: list[str] = []
        class_values: list[int] = []
        for candidate_id in ordered_ids:
            label = self._node_label(node_by_id.get(candidate_id, {"id": candidate_id})) or candidate_id
            labels.append(label)
            class_idx = activity_vocab.get(candidate_id, -1)
            if class_idx == -1:
                class_idx = activity_vocab.get(label, -1)
            class_values.append(int(class_idx))
        return ordered_ids, labels, class_values

    def _build_candidate_struct_x(
        self,
        *,
        dto: ProcessStructureDTO,
        candidate_labels: list[str],
        candidate_class_index: torch.Tensor,
        activity_vocab: Dict[str, int],
    ) -> torch.Tensor | None:
        base = self._build_struct_x(dto=dto, activity_vocab=activity_vocab)
        if not isinstance(base, torch.Tensor):
            return None
        out = torch.zeros((len(candidate_labels), int(base.size(1))), dtype=torch.float32)
        for row_idx, class_idx in enumerate(candidate_class_index.view(-1).tolist()):
            if 0 <= int(class_idx) < int(base.size(0)):
                out[row_idx] = base[int(class_idx)]
        return out

    def _handle_topology_projection_diagnostics(self, *, diagnostics: TopologyProjectionDiagnostics) -> None:
        if diagnostics.is_aligned:
            return
        action = str(self.topology_projection_config.get("on_fail", "warn"))
        reasons = ",".join(diagnostics.failure_reasons) or "unknown"
        if action == "raise":
            raise ValueError(
                "Topology projection alignment failed: "
                f"gateway_mode={diagnostics.gateway_mode} reasons={reasons} "
                f"missing_vocab_nodes={diagnostics.missing_vocab_nodes} "
                f"skipped_edges={diagnostics.skipped_projected_edges} "
                f"missing_node_metadata={diagnostics.missing_node_metadata}."
            )
        key = (diagnostics.gateway_mode, action, reasons)
        if key in self._topology_projection_warned_keys:
            return
        self._topology_projection_warned_keys.add(key)
        logger.warning(
            "Topology projection alignment warning: gateway_mode=%s action=%s reasons=%s "
            "missing_vocab_nodes=%s skipped_edges=%d missing_node_metadata=%s.",
            diagnostics.gateway_mode,
            action,
            reasons,
            ",".join(diagnostics.missing_vocab_nodes) or "none",
            len(diagnostics.skipped_projected_edges),
            bool(diagnostics.missing_node_metadata),
        )

    @staticmethod
    def _attach_topology_projection_summary(
        *,
        contract: GraphTensorContract,
        diagnostics: TopologyProjectionDiagnostics,
    ) -> None:
        contract["topology_projection_aligned"] = bool(diagnostics.is_aligned)
        contract["topology_projection_projected_edge_count"] = int(diagnostics.projected_edge_count)
        contract["topology_projection_source_path_count"] = int(diagnostics.source_path_count)
        contract["topology_projection_skipped_edge_count"] = int(len(diagnostics.skipped_projected_edges))
        contract["topology_projection_missing_vocab_count"] = int(len(diagnostics.missing_vocab_nodes))
        contract["topology_projection_duplicate_label_count"] = int(len(diagnostics.duplicate_activity_labels))
        contract["topology_projection_missing_node_metadata"] = bool(diagnostics.missing_node_metadata)

    def _project_allowed_edge_paths_for_prediction(self, dto: ProcessStructureDTO) -> dict[tuple[str, str], list[list[str]]]:
        activity_vocab = {
            str(node.get("id", "")).strip(): idx
            for idx, node in enumerate(dto.nodes or [])
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        if not activity_vocab:
            activity_vocab = {
                token: idx
                for idx, token in enumerate(sorted({str(item).strip() for edge in dto.allowed_edges for item in edge if str(item).strip()}))
            }
        return TopologyProjectionCompiler(gateway_mode=self.topology_gateway_mode).project(
            dto=dto,
            activity_vocab=activity_vocab,
        ).projected_edge_paths

    @classmethod
    def _classify_projection_nodes(cls, nodes: list[Dict[str, Any]]) -> dict[str, str]:
        return TopologyProjectionCompiler.classify_nodes(nodes)

    @staticmethod
    def _is_prediction_node_type(normalized_type: str) -> bool:
        return TopologyProjectionCompiler.is_prediction_node_type(normalized_type)

    @staticmethod
    def _reachable_prediction_node_paths_through_transparent(
        *,
        source: str,
        outgoing: dict[str, list[str]],
        node_roles: dict[str, str],
    ) -> dict[str, list[list[str]]]:
        return TopologyProjectionCompiler.reachable_prediction_node_paths_through_transparent(
            source=source,
            outgoing=outgoing,
            node_roles=node_roles,
        )

    @staticmethod
    def _projected_edge_weight(
        *,
        paths: list[list[str]],
        edge_weight_index: Dict[str, float],
        edge_weight_spec: Dict[str, Any],
    ) -> float:
        default = float(edge_weight_spec.get("default", 1.0) or 1.0)
        metric = str(edge_weight_spec.get("metric", "")).strip().lower()
        if metric != "transition_probability" or not paths:
            return default
        total = 0.0
        for path in paths:
            if len(path) < 2:
                continue
            probability = 1.0
            for src, dst in zip(path, path[1:]):
                probability *= float(edge_weight_index.get(f"{src}|||{dst}", default))
            total += probability
        return float(max(0.0, min(1.0, total if total > 0.0 else default)))

    def _resolve_topology_disk_cache_dir(self, raw_dir: str | None) -> Path | None:
        if self.cache_policy != "full":
            return None
        raw = str(raw_dir).strip() if raw_dir is not None else ""
        if not raw:
            raw = ".cache/dynamic_graph_builder"
        process_token = (self.process_name or "__auto__").strip()
        safe_token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in process_token) or "__auto__"
        target = Path(raw).resolve() / safe_token
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not initialize topology disk cache dir '%s': %s", target, exc)
            return None
        return target

    @staticmethod
    def _topology_cache_key_digest(cache_key: tuple[Any, ...]) -> str:
        text = repr(cache_key).encode("utf-8", errors="ignore")
        return hashlib.sha1(text).hexdigest()

    def _topology_cache_file_path(self, cache_key: tuple[Any, ...]) -> Path | None:
        if self._topology_disk_cache_dir is None:
            return None
        digest = self._topology_cache_key_digest(cache_key)
        return self._topology_disk_cache_dir / f"{digest}.pt"

    def _load_compiled_topology_from_disk(self, cache_key: tuple[Any, ...]) -> Dict[str, Any] | None:
        path = self._topology_cache_file_path(cache_key)
        if path is None or (not path.exists()):
            return None
        try:
            payload = load_trusted_torch_artifact(path, map_location="cpu")
        except Exception as exc:
            logger.warning("Failed to load topology disk cache '%s': %s", path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("schema", -1)) != int(self._topology_disk_cache_schema):
            return None
        compiled = payload.get("compiled")
        if not isinstance(compiled, dict):
            return None
        allowed_masks_by_src = compiled.get("allowed_masks_by_src")
        structural_edge_index = compiled.get("structural_edge_index")
        structural_edge_weight = compiled.get("structural_edge_weight")
        struct_node_to_class_index = compiled.get("struct_node_to_class_index")
        struct_node_to_candidate_index = compiled.get("struct_node_to_candidate_index")
        candidate_class_index = compiled.get("candidate_class_index")
        candidate_is_unseen = compiled.get("candidate_is_unseen")
        struct_x = compiled.get("struct_x")
        diagnostics = compiled.get("topology_projection_diagnostics")
        if not isinstance(allowed_masks_by_src, dict):
            return None
        if not isinstance(structural_edge_index, torch.Tensor):
            return None
        if not isinstance(structural_edge_weight, torch.Tensor):
            return None
        if not isinstance(struct_node_to_class_index, torch.Tensor):
            return None
        if not isinstance(struct_node_to_candidate_index, torch.Tensor):
            return None
        if not isinstance(candidate_class_index, torch.Tensor):
            return None
        if not isinstance(candidate_is_unseen, torch.Tensor):
            return None
        if struct_x is not None and (not isinstance(struct_x, torch.Tensor)):
            return None
        if diagnostics is not None and not isinstance(diagnostics, dict):
            return None
        return {
            "allowed_masks_by_src": allowed_masks_by_src,
            "candidate_allowed_masks_by_src": compiled.get("candidate_allowed_masks_by_src", {}),
            "structural_edge_index": structural_edge_index,
            "structural_edge_weight": structural_edge_weight,
            "struct_node_to_class_index": struct_node_to_class_index,
            "struct_node_to_candidate_index": struct_node_to_candidate_index,
            "candidate_ids": tuple(str(item) for item in compiled.get("candidate_ids", ())),
            "candidate_labels": tuple(str(item) for item in compiled.get("candidate_labels", ())),
            "candidate_class_index": candidate_class_index,
            "candidate_is_unseen": candidate_is_unseen,
            "candidate_indices_by_label": compiled.get("candidate_indices_by_label", {}),
            "candidate_indices_by_id": compiled.get("candidate_indices_by_id", {}),
            "candidate_count": int(compiled.get("candidate_count", int(candidate_class_index.numel()))),
            "struct_x": struct_x,
            "topology_projection_diagnostics": diagnostics,
        }

    def _save_compiled_topology_to_disk(self, *, cache_key: tuple[Any, ...], compiled: Dict[str, Any]) -> None:
        path = self._topology_cache_file_path(cache_key)
        if path is None:
            return
        payload = {
            "schema": int(self._topology_disk_cache_schema),
            "compiled": self._serializable_compiled_topology(compiled),
        }
        tmp_path = path.with_suffix(".tmp")
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
        except Exception as exc:
            logger.warning("Failed to persist topology disk cache '%s': %s", path, exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    @staticmethod
    def _serializable_compiled_topology(compiled: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(compiled)
        diagnostics = payload.get("topology_projection_diagnostics")
        if isinstance(diagnostics, TopologyProjectionDiagnostics):
            payload["topology_projection_diagnostics"] = diagnostics.as_dict()
        return payload

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
