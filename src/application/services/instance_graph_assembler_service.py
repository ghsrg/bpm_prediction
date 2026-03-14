"""Stage 3.1 service that assembles canonical and fallback instance graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO
from src.domain.ports.knowledge_graph_port import IKnowledgeGraphPort


@dataclass
class InstanceGraphBuildResult:
    """Container for assembled graph artifacts and diagnostics."""

    graph: Dict[str, Any]
    projection: Dict[str, Any]
    mode: str
    diagnostics: RuntimeFetchDiagnosticsDTO


class InstanceGraphAssemblerService:
    """Build canonical IG artifacts from runtime rows with robust fallback rules."""

    def __init__(self, knowledge_port: IKnowledgeGraphPort) -> None:
        self.knowledge_port = knowledge_port
        self._low_coverage_streak: Dict[Tuple[str, str], int] = {}

    def build(
        self,
        *,
        process_name: str,
        version_key: str,
        events: Sequence[ProcessEventDTO],
        execution_rows: Sequence[Dict[str, Any]],
        variables_rows: Sequence[Dict[str, Any]],
        identity_rows: Sequence[Dict[str, Any]],
        diagnostics: RuntimeFetchDiagnosticsDTO,
        config: Dict[str, Any],
    ) -> InstanceGraphBuildResult:
        """Assemble canonical graph; fallback to structure-only when required."""
        cfg = self._normalize_config(config)
        key = (process_name, version_key)

        if not events:
            diagnostics.fallback_triggered = True
            diagnostics.fallback_reason = "empty_history"
            fallback = self._build_fallback_graph(process_name, version_key, diagnostics)
            return fallback

        coverage_percent = diagnostics.history_coverage_percent
        low_coverage = False
        if coverage_percent is not None:
            low_coverage = (coverage_percent / 100.0) < cfg["fallback_min_coverage_ratio"]
            if "legacy_removal_time_treated_as_eternal" in diagnostics.warnings:
                low_coverage = False

        if low_coverage:
            self._low_coverage_streak[key] = self._low_coverage_streak.get(key, 0) + 1
        else:
            self._low_coverage_streak[key] = 0

        if self._low_coverage_streak[key] >= cfg["coverage_hysteresis_fetches"]:
            diagnostics.fallback_triggered = True
            diagnostics.fallback_reason = "low_coverage_hysteresis"
            fallback = self._build_fallback_graph(process_name, version_key, diagnostics)
            return fallback

        canonical_mode = str(cfg["canonical_mode"])
        if canonical_mode == "execution-centric":
            canonical_mode = self._resolve_execution_mode(
                process_name=process_name,
                version_key=version_key,
                events=events,
                execution_rows=execution_rows,
                threshold=float(cfg["execution_missing_degrade_threshold"]),
                diagnostics=diagnostics,
            )

        normalized_events = self._attach_identity_links(events=events, identity_rows=identity_rows)
        normalized_events = self._apply_high_mi_guard(normalized_events, cfg)

        graph = self._build_graph_payload(
            events=normalized_events,
            execution_rows=execution_rows,
            variables_rows=variables_rows,
            mode=canonical_mode,
            depth_limit=int(cfg["execution_tree_depth_limit"]),
        )
        projection = self._build_projection_payload(normalized_events)

        diagnostics.fallback_triggered = False
        diagnostics.fallback_reason = None
        diagnostics.meta["canonical_mode"] = canonical_mode
        diagnostics.meta["identity_rows"] = len(identity_rows)
        diagnostics.meta["variables_rows"] = len(variables_rows)

        return InstanceGraphBuildResult(
            graph=graph,
            projection=projection,
            mode=f"canonical_{canonical_mode.replace('-', '_')}",
            diagnostics=diagnostics,
        )

    def _build_fallback_graph(
        self,
        process_name: str,
        version_key: str,
        diagnostics: RuntimeFetchDiagnosticsDTO,
    ) -> InstanceGraphBuildResult:
        dto = self.knowledge_port.get_process_structure(version_key, process_name=process_name)
        edges = list(dto.allowed_edges) if dto is not None else []
        nodes = sorted({node for edge in edges for node in edge})

        graph = {
            "nodes": [{"id": node, "source": "structure"} for node in nodes],
            "edges": [
                {"source": src, "target": dst, "edge_type": "sequence"}
                for src, dst in edges
            ],
            "metadata": {
                "mode": "fallback_structure_only",
                "potentially_stale_child_aggregates": True,
            },
        }
        projection = {
            "node_counts": {node: 0 for node in nodes},
            "transition_counts": {f"{src}->{dst}": 0 for src, dst in edges},
            "mode": "fallback_structure_only",
        }
        diagnostics.meta["canonical_mode"] = "fallback_structure_only"
        return InstanceGraphBuildResult(
            graph=graph,
            projection=projection,
            mode="fallback_structure_only",
            diagnostics=diagnostics,
        )

    def _resolve_execution_mode(
        self,
        *,
        process_name: str,
        version_key: str,
        events: Sequence[ProcessEventDTO],
        execution_rows: Sequence[Dict[str, Any]],
        threshold: float,
        diagnostics: RuntimeFetchDiagnosticsDTO,
    ) -> str:
        if not events:
            return "activity-centric"
        event_case_ids = {event.case_id for event in events}
        execution_case_ids = {
            str(row.get("case_id", "")).strip()
            for row in execution_rows
            if str(row.get("case_id", "")).strip()
        }
        if not event_case_ids:
            return "activity-centric"
        missing_ratio = float(len([case_id for case_id in event_case_ids if case_id not in execution_case_ids]) / len(event_case_ids))
        diagnostics.meta["execution_missing_ratio"] = missing_ratio
        if missing_ratio > threshold:
            diagnostics.warnings.append("execution_centric_degraded_to_activity")
            return "activity-centric"
        return "execution-centric"

    def _apply_high_mi_guard(
        self,
        events: Sequence[ProcessEventDTO],
        config: Dict[str, Any],
    ) -> List[ProcessEventDTO]:
        threshold = int(config["max_execution_nodes_per_case"])
        strategy = str(config["extreme_mi_strategy"]).lower()
        grouped: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
        for event in events:
            grouped[event.case_id].append(event)

        normalized: List[ProcessEventDTO] = []
        for case_id, items in grouped.items():
            if len(items) <= threshold:
                normalized.extend(items)
                continue
            if strategy == "sample":
                normalized.extend(sorted(items, key=lambda item: item.start_time or datetime.min)[:threshold])
                continue
            # aggregate strategy: keep one representative event per activity_def_id for this case.
            by_activity: Dict[str, ProcessEventDTO] = {}
            for event in sorted(items, key=lambda item: item.start_time or datetime.min):
                by_activity[event.activity_def_id] = event
                if len(by_activity) >= threshold:
                    break
            normalized.extend(by_activity.values())
        return normalized

    @staticmethod
    def _attach_identity_links(
        *,
        events: Sequence[ProcessEventDTO],
        identity_rows: Sequence[Dict[str, Any]],
    ) -> List[ProcessEventDTO]:
        by_task: Dict[str, List[str]] = defaultdict(list)
        for row in identity_rows:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            group_id = str(row.get("candidate_group_id", "")).strip()
            if group_id:
                by_task[task_id].append(group_id)

        attached: List[ProcessEventDTO] = []
        for event in events:
            if event.task_id and event.task_id in by_task:
                event = event.model_copy(update={"candidate_groups": sorted(set(by_task[event.task_id]))})
            attached.append(event)
        return attached

    def _build_graph_payload(
        self,
        *,
        events: Sequence[ProcessEventDTO],
        execution_rows: Sequence[Dict[str, Any]],
        variables_rows: Sequence[Dict[str, Any]],
        mode: str,
        depth_limit: int,
    ) -> Dict[str, Any]:
        by_case: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
        for event in events:
            by_case[event.case_id].append(event)

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        for case_id, case_events in by_case.items():
            ordered = sorted(case_events, key=lambda item: item.start_time or datetime.min)
            for idx, event in enumerate(ordered):
                node_id = event.act_inst_id or event.task_id or f"{case_id}:{idx}:{event.activity_def_id}"
                nodes.append(
                    {
                        "id": node_id,
                        "case_id": case_id,
                        "activity_def_id": event.activity_def_id,
                        "activity_type": event.activity_type,
                        "execution_id": event.execution_id,
                    }
                )
                if idx == 0:
                    continue
                prev = ordered[idx - 1]
                prev_id = prev.act_inst_id or prev.task_id or f"{case_id}:{idx-1}:{prev.activity_def_id}"
                edge_type = "sequence"
                if event.activity_type == "boundaryEvent":
                    interrupting = bool(event.extra.get("interrupting", False))
                    edge_type = "cancellation_edge" if interrupting else "fork_edge"
                edges.append({"source": prev_id, "target": node_id, "edge_type": edge_type})

        if mode == "execution-centric" and execution_rows:
            for row in execution_rows:
                depth = self._safe_int(row.get("depth") or row.get("scope_depth"))
                if depth_limit >= 0 and depth > depth_limit:
                    continue
                parent = str(row.get("parent_execution_id") or row.get("parent_id") or "").strip()
                child = str(row.get("execution_id") or row.get("id") or "").strip()
                if parent and child:
                    edges.append({"source": parent, "target": child, "edge_type": "scope"})

        loop_markers = {
            str(row.get("execution_id")): self._safe_int(row.get("loop_counter"))
            for row in variables_rows
            if str(row.get("execution_id", "")).strip()
        }
        for node in nodes:
            execution_id = str(node.get("execution_id") or "").strip()
            if execution_id and execution_id in loop_markers:
                node["loop_counter"] = loop_markers[execution_id]

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "mode": mode,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
            },
        }

    @staticmethod
    def _build_projection_payload(events: Sequence[ProcessEventDTO]) -> Dict[str, Any]:
        by_case: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
        for event in events:
            by_case[event.case_id].append(event)

        node_counts: Dict[str, int] = defaultdict(int)
        transition_counts: Dict[str, int] = defaultdict(int)
        for case_events in by_case.values():
            ordered = sorted(case_events, key=lambda item: item.start_time or datetime.min)
            for idx, event in enumerate(ordered):
                node_counts[event.activity_def_id] += 1
                if idx > 0:
                    prev = ordered[idx - 1]
                    transition_counts[f"{prev.activity_def_id}->{event.activity_def_id}"] += 1
        return {
            "node_counts": dict(node_counts),
            "transition_counts": dict(transition_counts),
            "mode": "collapsed_projection",
        }

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "canonical_mode": str(config.get("canonical_mode", "activity-centric")).strip().lower(),
            "execution_tree_depth_limit": int(config.get("execution_tree_depth_limit", 4)),
            "execution_missing_degrade_threshold": float(config.get("execution_missing_degrade_threshold", 0.5)),
            "max_execution_nodes_per_case": int(config.get("max_execution_nodes_per_case", 5000)),
            "extreme_mi_strategy": str(config.get("extreme_mi_strategy", "aggregate")).strip().lower(),
            "fallback_min_coverage_ratio": float(config.get("fallback_min_coverage_ratio", 0.05)),
            "coverage_hysteresis_fetches": int(config.get("coverage_hysteresis_fetches", 2)),
        }
