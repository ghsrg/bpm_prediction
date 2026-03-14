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
        task_rows: Sequence[Dict[str, Any]] | None = None,
        process_variables_rows: Sequence[Dict[str, Any]] | None = None,
        process_instance_links: Sequence[Dict[str, Any]] | None = None,
    ) -> InstanceGraphBuildResult:
        """Assemble canonical graph; fallback to structure-only when required."""
        cfg = self._normalize_config(config)
        key = (process_name, version_key)
        normalized_task_rows = list(task_rows or [])
        normalized_process_variables_rows = list(process_variables_rows or [])
        normalized_process_instance_links = list(process_instance_links or [])

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

        normalized_events = self._enrich_actor_roles(
            events=events,
            identity_rows=identity_rows,
            task_rows=normalized_task_rows,
        )
        normalized_events = self._apply_high_mi_guard(normalized_events, cfg)

        graph = self._build_graph_payload(
            events=normalized_events,
            execution_rows=execution_rows,
            variables_rows=variables_rows,
            process_variables_rows=normalized_process_variables_rows,
            process_instance_links=normalized_process_instance_links,
            mode=canonical_mode,
            depth_limit=int(cfg["execution_tree_depth_limit"]),
            subprocess_mode=str(cfg["subprocess_graph_mode"]),
        )
        graph.setdefault("metadata", {})["coverage_percent"] = diagnostics.history_coverage_percent
        unresolved_call_count = int(graph.get("metadata", {}).get("unresolved_call_count", 0))
        if unresolved_call_count > 0:
            diagnostics.warnings.append("unresolved_call")
            if bool(cfg.get("fallback_on_unresolved_call", True)):
                diagnostics.fallback_triggered = True
                diagnostics.fallback_reason = "unresolved_call"
                fallback = self._build_fallback_graph(process_name, version_key, diagnostics)
                return fallback
        projection = self._build_projection_payload(normalized_events)

        diagnostics.fallback_triggered = False
        diagnostics.fallback_reason = None
        diagnostics.meta["canonical_mode"] = canonical_mode
        diagnostics.meta["identity_rows"] = len(identity_rows)
        diagnostics.meta["task_rows"] = len(normalized_task_rows)
        diagnostics.meta["variables_rows"] = len(variables_rows)
        diagnostics.meta["process_variables_rows"] = len(normalized_process_variables_rows)
        diagnostics.meta["process_instance_links"] = len(normalized_process_instance_links)

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

    @classmethod
    def _enrich_actor_roles(
        cls,
        *,
        events: Sequence[ProcessEventDTO],
        identity_rows: Sequence[Dict[str, Any]],
        task_rows: Sequence[Dict[str, Any]],
    ) -> List[ProcessEventDTO]:
        candidate_groups_by_task: Dict[str, set[str]] = defaultdict(set)
        candidate_users_by_task: Dict[str, set[str]] = defaultdict(set)
        assignee_by_task: Dict[str, str] = {}
        executed_by_task: Dict[str, str] = {}

        for row in task_rows:
            task_id = cls._norm_text(row.get("task_id"))
            if not task_id:
                continue
            assignee = cls._norm_text(row.get("assignee"))
            executed_by = cls._norm_text(row.get("executed_by") or row.get("completed_by"))
            if assignee:
                assignee_by_task[task_id] = assignee
            if executed_by:
                executed_by_task[task_id] = executed_by

        for row in identity_rows:
            task_id = cls._norm_text(row.get("task_id"))
            if not task_id:
                continue
            link_type = cls._norm_text(row.get("link_type") or row.get("type")) or ""
            user_id = cls._norm_text(
                row.get("candidate_user_id")
                or row.get("user_id")
                or row.get("potential_user_id")
            )
            group_id = cls._norm_text(
                row.get("candidate_group_id")
                or row.get("group_id")
                or row.get("potential_group_id")
            )

            if link_type in {"assignee", "owner"}:
                if user_id:
                    assignee_by_task[task_id] = user_id
                continue

            if link_type in {"candidate", "candidate-user", "candidate-group", "potentialowner", "participant", ""}:
                if user_id:
                    candidate_users_by_task[task_id].add(user_id)
                if group_id:
                    candidate_groups_by_task[task_id].add(group_id)

        enriched: List[ProcessEventDTO] = []
        for event in events:
            task_id = cls._norm_text(event.task_id)
            potential_users = set(event.potential_executor_users or [])
            potential_groups = set(event.potential_executor_groups or event.candidate_groups or [])
            if task_id:
                potential_users.update(candidate_users_by_task.get(task_id, set()))
                potential_groups.update(candidate_groups_by_task.get(task_id, set()))

            assigned = (
                event.assigned_executor
                or (assignee_by_task.get(task_id) if task_id else None)
                or event.assignee
            )
            executed_by = (
                event.executed_by
                or (executed_by_task.get(task_id) if task_id else None)
                or event.assignee
                or assigned
            )
            assignee_alias = event.assignee or executed_by or assigned

            update_payload: Dict[str, Any] = {
                "assigned_executor": assigned,
                "executed_by": executed_by,
                "assignee": assignee_alias,
                "potential_executor_users": sorted(potential_users) if potential_users else None,
                "potential_executor_groups": sorted(potential_groups) if potential_groups else None,
                "candidate_groups": sorted(potential_groups) if potential_groups else None,
            }
            enriched.append(event.model_copy(update=update_payload))
        return enriched

    def _build_graph_payload(
        self,
        *,
        events: Sequence[ProcessEventDTO],
        execution_rows: Sequence[Dict[str, Any]],
        variables_rows: Sequence[Dict[str, Any]],
        process_variables_rows: Sequence[Dict[str, Any]],
        process_instance_links: Sequence[Dict[str, Any]],
        mode: str,
        depth_limit: int,
        subprocess_mode: str,
    ) -> Dict[str, Any]:
        by_case: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
        for event in events:
            by_case[event.case_id].append(event)

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        call_activity_links: List[Dict[str, Any]] = []
        edge_keys: set[Tuple[str, str, str]] = set()
        case_variable_map, exec_variable_map = self._index_process_variables(process_variables_rows)
        mi_metrics = self._index_multi_instance_variables(variables_rows)
        links_by_parent_case = self._index_process_instance_links(process_instance_links)

        parallel_token_max = 0
        cycle_count = 0
        parallel_paths_found = 0
        call_activities_found = 0
        resolved_calls = 0
        requires_inference_count = 0
        unresolved_call_count = 0
        for case_id, case_events in by_case.items():
            ordered = sorted(case_events, key=lambda item: self._event_order_key(item))
            execution_nodes: Dict[str, List[str]] = defaultdict(list)
            execution_parent_map: Dict[str, str] = {}
            event_by_node_id: Dict[str, ProcessEventDTO] = {}
            node_start_ts: Dict[str, datetime] = {}
            node_id_by_act_inst: Dict[str, str] = {}
            node_times: List[datetime] = []
            case_execution_meta = self._index_execution_meta_for_case(
                execution_rows=execution_rows,
                case_id=case_id,
                depth_limit=depth_limit,
            )

            for idx, event in enumerate(ordered):
                if self._is_technical_subprocess_node(event=event, case_id=case_id):
                    # Ignore technical start/end markers that exist only to wire internal subprocess flow.
                    continue
                node_id = event.act_inst_id or event.task_id or f"{case_id}:{idx}:{event.activity_def_id}"
                start_ts = event.start_time or datetime.min
                process_variables = dict(case_variable_map.get(case_id, {}))
                execution_id = self._norm_text(event.execution_id)
                parent_execution_id = self._norm_text(event.parent_execution_id)
                if execution_id:
                    process_variables.update(exec_variable_map.get((case_id, execution_id), {}))
                    execution_nodes[execution_id].append(node_id)
                    if parent_execution_id:
                        execution_parent_map[execution_id] = parent_execution_id
                act_inst_key = self._norm_text(event.act_inst_id)
                if act_inst_key:
                    node_id_by_act_inst[act_inst_key] = node_id
                event_by_node_id[node_id] = event
                node_start_ts[node_id] = start_ts
                node_times.append(event.end_time or event.start_time or datetime.min)

                call_proc_inst_id = self._norm_text(event.call_proc_inst_id)
                call_link = None
                is_call_activity = self._norm_text(event.activity_type) == "callActivity"
                if is_call_activity:
                    call_activities_found += 1
                    requires_inference_count += 1
                if call_proc_inst_id or is_call_activity:
                    call_link = {
                        "parent_case_id": case_id,
                        "parent_activity_instance_id": node_id,
                        "parent_activity_def_id": event.activity_def_id,
                        "parent_execution_id": execution_id,
                        "child_case_id": call_proc_inst_id or None,
                        "called_element": event.called_element,
                        "binding_type": event.binding_type,
                        "version_tag": event.version_tag,
                        "version_number": event.version_number,
                        "resolved_child_proc_def_id": event.resolved_child_proc_def_id,
                        "child_process_key": event.child_process_key,
                        "child_version": event.child_version,
                        "requires_separate_inference": bool(is_call_activity),
                        "inference_entry_point": "child_process_root" if is_call_activity else None,
                    }
                    call_activity_links.append(call_link)
                    if is_call_activity:
                        if self._norm_text(event.resolved_child_proc_def_id):
                            resolved_calls += 1
                        else:
                            unresolved_call_count += 1

                execution_meta = case_execution_meta.get(execution_id, {})
                loop_counter = mi_metrics["loop_counter"].get((case_id, execution_id)) if execution_id else None
                nr_of_instances = mi_metrics["nr_of_instances"].get((case_id, execution_id)) if execution_id else None
                nr_of_completed = mi_metrics["nr_of_completed"].get((case_id, execution_id)) if execution_id else None
                boundary_type = self._resolve_boundary_type(event)
                nodes.append(
                    {
                        "id": node_id,
                        "case_id": case_id,
                        "activity_def_id": event.activity_def_id,
                        "activity_name": event.activity_name,
                        "activity_type": event.activity_type,
                        "type": event.activity_type,
                        "start_time": event.start_time,
                        "end_time": event.end_time,
                        "duration_ms": event.duration_ms,
                        "is_active": event.end_time is None,
                        "execution_id": event.execution_id,
                        "parent_execution_id": parent_execution_id or execution_meta.get("parent_execution_id"),
                        "parent_act_inst_id": event.parent_act_inst_id,
                        "sequence_counter": event.sequence_counter,
                        "scope_depth": event.scope_depth if event.scope_depth is not None else execution_meta.get("scope_depth"),
                        "is_concurrent": event.is_concurrent if event.is_concurrent is not None else execution_meta.get("is_concurrent"),
                        "is_scope": event.is_scope if event.is_scope is not None else execution_meta.get("is_scope"),
                        "is_event_scope": event.is_event_scope if event.is_event_scope is not None else execution_meta.get("is_event_scope"),
                        "execution_rev": event.execution_rev if event.execution_rev is not None else execution_meta.get("execution_rev"),
                        "parallel_token_count": 1,
                        "loop_counter": loop_counter,
                        "nr_of_instances": nr_of_instances,
                        "nr_of_completed_instances": nr_of_completed,
                        "boundary_type": boundary_type,
                        "call_proc_inst_id": call_proc_inst_id or None,
                        "called_element": event.called_element,
                        "binding_type": event.binding_type,
                        "version_tag": event.version_tag,
                        "version_number": event.version_number,
                        "resolved_child_proc_def_id": event.resolved_child_proc_def_id,
                        "child_process_key": event.child_process_key,
                        "child_version": event.child_version,
                        "requires_separate_inference": bool(is_call_activity),
                        "inference_entry_point": "child_process_root" if is_call_activity else None,
                        "assigned_executor": event.assigned_executor,
                        "executed_by": event.executed_by,
                        "potential_executor_users": event.potential_executor_users,
                        "potential_executor_groups": event.potential_executor_groups,
                        "process_variables": process_variables or None,
                        "call_activity_link": call_link,
                    }
                )

            # Build scope hierarchy and sequence inside each scope.
            nodes_by_parent_scope: Dict[str, List[str]] = defaultdict(list)
            for node_id, event in event_by_node_id.items():
                scope_parent = self._norm_text(event.parent_act_inst_id) or case_id
                nodes_by_parent_scope[scope_parent].append(node_id)

            scope_level_cache: Dict[str, int] = {}
            subprocess_parent_cache: Dict[str, str | None] = {}
            for node in nodes:
                if node.get("case_id") != case_id:
                    continue
                node_id = self._norm_text(node.get("id"))
                level, parent_subprocess_id = self._resolve_scope_meta(
                    node_id=node_id,
                    event_by_node_id=event_by_node_id,
                    node_id_by_act_inst=node_id_by_act_inst,
                    case_id=case_id,
                    scope_level_cache=scope_level_cache,
                    subprocess_parent_cache=subprocess_parent_cache,
                )
                node["scope_level"] = level
                node["parent_subprocess_id"] = parent_subprocess_id

            node_scope_level: Dict[str, int] = {}
            for node in nodes:
                if node.get("case_id") != case_id:
                    continue
                node_id = self._norm_text(node.get("id"))
                node_scope_level[node_id] = int(node.get("scope_level") or 0)

            container_entry_exit: Dict[str, Tuple[str, str]] = {}
            scope_visible_children: Dict[str, List[str]] = {}
            for scope_parent, child_node_ids in nodes_by_parent_scope.items():
                sorted_children = sorted(
                    child_node_ids,
                    key=lambda node_id: self._event_order_key(event_by_node_id[node_id]),
                )
                scope_visible_children[scope_parent] = list(sorted_children)
                visible_children = list(sorted_children)

                # For flattened mode remove phantom start/end events from subprocess inner flow.
                if scope_parent != case_id:
                    visible_children = [
                        node_id
                        for node_id in visible_children
                        if not self._is_subprocess_phantom_event(event_by_node_id[node_id])
                    ]
                    scope_visible_children[scope_parent] = list(visible_children)

                container_node_id = node_id_by_act_inst.get(scope_parent)
                if container_node_id and scope_parent != case_id:
                    for node in nodes:
                        if node.get("id") == container_node_id:
                            node["is_subprocess_container"] = True
                            node["subprocess_children"] = [
                                child_node
                                for child_node in sorted_children
                                if child_node in event_by_node_id
                            ]
                            break
                    if visible_children:
                        container_entry_exit[container_node_id] = (visible_children[0], visible_children[-1])
                        entry_edge_type = "sequence" if subprocess_mode == "flattened" else "subprocess_entry"
                        exit_edge_type = "sequence" if subprocess_mode == "flattened" else "subprocess_exit"
                        self._append_edge(
                            edges=edges,
                            edge_keys=edge_keys,
                            source=container_node_id,
                            target=visible_children[0],
                            edge_type=entry_edge_type,
                            transition_type="subprocess_entry",
                        )
                        self._append_edge(
                            edges=edges,
                            edge_keys=edge_keys,
                            source=visible_children[-1],
                            target=container_node_id,
                            edge_type=exit_edge_type,
                            transition_type="subprocess_exit",
                        )

                for idx in range(1, len(visible_children)):
                    prev_id = visible_children[idx - 1]
                    curr_id = visible_children[idx]
                    prev_event = event_by_node_id[prev_id]
                    curr_event = event_by_node_id[curr_id]
                    self._append_edge(
                        edges=edges,
                        edge_keys=edge_keys,
                        source=prev_id,
                        target=curr_id,
                        edge_type=self._resolve_edge_type(curr_event),
                        transition_type="sequence",
                        timestamp_delta=self._timestamp_delta(prev_event, curr_event),
                    )

            # Rewire simple parallel pattern inside each scope:
            # split gateway -> two branch nodes -> join gateway.
            for scope_parent, visible_children in scope_visible_children.items():
                if len(visible_children) < 4:
                    continue
                self._rewire_parallel_subprocess_branches(
                    edges=edges,
                    edge_keys=edge_keys,
                    event_by_node_id=event_by_node_id,
                    ordered_scope_nodes=visible_children,
                    case_id=case_id,
                    scope_key=scope_parent or case_id,
                )

            # 2) Build execution hierarchy and parallel branches.
            first_node_by_execution: Dict[str, Tuple[datetime, str]] = {}
            ordered_by_execution: Dict[str, List[Tuple[datetime, str]]] = {}
            for execution_id, node_ids in execution_nodes.items():
                sorted_nodes = sorted(
                    [(node_start_ts.get(node_id, datetime.min), node_id) for node_id in node_ids],
                    key=lambda item: item[0],
                )
                if not sorted_nodes:
                    continue
                ordered_by_execution[execution_id] = sorted_nodes
                root_scope_nodes = [item for item in sorted_nodes if node_scope_level.get(item[1], 0) == 0]
                first_node_by_execution[execution_id] = root_scope_nodes[0] if root_scope_nodes else sorted_nodes[0]

            parent_to_children: Dict[str, List[str]] = defaultdict(list)
            for execution_id, parent_execution_id in execution_parent_map.items():
                if parent_execution_id:
                    parent_to_children[parent_execution_id].append(execution_id)
            for execution_id, meta in case_execution_meta.items():
                parent_execution_id = self._norm_text(meta.get("parent_execution_id"))
                if parent_execution_id:
                    parent_to_children[parent_execution_id].append(execution_id)

            for parent_execution_id, child_executions in parent_to_children.items():
                unique_children = sorted({child for child in child_executions if child in first_node_by_execution})
                if not unique_children:
                    continue
                is_parallel_group = len(unique_children) > 1 or any(
                    bool(case_execution_meta.get(child, {}).get("is_concurrent")) for child in unique_children
                )
                if not is_parallel_group:
                    continue
                group_id = f"{case_id}:{parent_execution_id}"
                token_count = len(unique_children)
                parallel_token_max = max(parallel_token_max, token_count)
                parallel_paths_found += 1
                child_entry_nodes: List[str] = []
                for child_execution_id in unique_children:
                    child_ts, child_node_id = first_node_by_execution[child_execution_id]
                    source_node_id = self._pick_execution_anchor(
                        ordered_by_execution=ordered_by_execution,
                        parent_execution_id=parent_execution_id,
                        target_ts=child_ts,
                    )
                    if not source_node_id:
                        continue
                    self._append_edge(
                        edges=edges,
                        edge_keys=edge_keys,
                        source=source_node_id,
                        target=child_node_id,
                        edge_type="parallel_branch",
                        parallel_group_id=group_id,
                        token_count_in_group=token_count,
                    )
                    child_entry_nodes.append(child_node_id)

                # Remove forced sequential chain between branch entry nodes.
                for left_idx in range(len(child_entry_nodes)):
                    for right_idx in range(left_idx + 1, len(child_entry_nodes)):
                        left_node = child_entry_nodes[left_idx]
                        right_node = child_entry_nodes[right_idx]
                        self._remove_edge(
                            edges=edges,
                            edge_keys=edge_keys,
                            source=left_node,
                            target=right_node,
                            edge_type="sequence",
                        )
                        self._remove_edge(
                            edges=edges,
                            edge_keys=edge_keys,
                            source=right_node,
                            target=left_node,
                            edge_type="sequence",
                        )

            # 3) Loops/cycles by repeated activity inside each execution stream.
            for execution_id, stream in ordered_by_execution.items():
                activity_seen: Dict[str, str] = {}
                loop_iteration_by_activity: Dict[str, int] = defaultdict(int)
                for _, node_id in stream:
                    event = event_by_node_id.get(node_id)
                    if event is None:
                        continue
                    activity_def_id = self._norm_text(event.activity_def_id)
                    if not activity_def_id:
                        continue
                    loop_iteration_by_activity[activity_def_id] += 1
                    iteration = loop_iteration_by_activity[activity_def_id]
                    if iteration > 1 and activity_def_id in activity_seen:
                        self._append_edge(
                            edges=edges,
                            edge_keys=edge_keys,
                            source=activity_seen[activity_def_id],
                            target=node_id,
                            edge_type="loop_back",
                            is_loop_back_edge=True,
                            loop_iteration=iteration,
                            execution_id=execution_id,
                        )
                        cycle_count += 1
                    activity_seen[activity_def_id] = node_id

            # 4) Call activity hierarchy fallback from ACT_HI_PROCINST links.
            for row in links_by_parent_case.get(case_id, []):
                child_case_id = self._norm_text(row.get("case_id"))
                if not child_case_id:
                    continue
                if any(link.get("child_case_id") == child_case_id for link in call_activity_links):
                    continue
                call_activity_links.append(
                    {
                        "parent_case_id": case_id,
                        "parent_activity_instance_id": None,
                        "parent_activity_def_id": None,
                        "parent_execution_id": None,
                        "child_case_id": child_case_id,
                        "source": "process_instance_links",
                    }
                )

            # 5) Confidence score per node (sample size + freshness).
            activity_counts: Dict[str, int] = defaultdict(int)
            for event in ordered:
                activity_counts[event.activity_def_id] += 1
            min_ts = min(node_times) if node_times else datetime.min
            max_ts = max(node_times) if node_times else datetime.min
            range_seconds = max(1.0, (max_ts - min_ts).total_seconds()) if max_ts >= min_ts else 1.0
            for node in nodes:
                if node.get("case_id") != case_id:
                    continue
                activity_def_id = self._norm_text(node.get("activity_def_id"))
                sample_size = activity_counts.get(activity_def_id, 0)
                node_ts = node.get("end_time") or node.get("start_time") or min_ts
                freshness = 0.0
                if isinstance(node_ts, datetime):
                    freshness = max(0.0, min(1.0, (node_ts - min_ts).total_seconds() / range_seconds))
                node["confidence_score"] = round(
                    max(0.0, min(1.0, 0.6 * min(1.0, sample_size / 5.0) + 0.4 * freshness)),
                    4,
                )
                if self._safe_int(node.get("parallel_token_count")) < parallel_token_max and self._norm_text(node.get("execution_id")):
                    node["parallel_token_count"] = max(1, self._safe_int(node.get("parallel_token_count")))
                    child_count = len(parent_to_children.get(self._norm_text(node.get("execution_id")), []))
                    if child_count > 1:
                        node["parallel_token_count"] = child_count

            if subprocess_mode == "flattened-no-subprocess-node":
                self._collapse_subprocess_containers(
                    nodes=nodes,
                    edges=edges,
                    edge_keys=edge_keys,
                    container_entry_exit=container_entry_exit,
                    case_id=case_id,
                )

        loop_density = float(cycle_count / max(1, len(nodes)))
        parallel_paths_found = sum(1 for edge in edges if edge.get("edge_type") == "parallel_branch")
        parallel_token_max = max(
            parallel_token_max,
            max((self._safe_int(edge.get("token_count_in_group")) for edge in edges if edge.get("edge_type") == "parallel_branch"), default=0),
        )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "mode": "canonical",
                "canonical_mode": mode,
                "subprocesses_flattened": subprocess_mode in {"flattened", "flattened-no-subprocess-node"},
                "hierarchical_mode": subprocess_mode == "hierarchical",
                "subprocess_nodes_removed": subprocess_mode == "flattened-no-subprocess-node",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_call_activity_links": len(call_activity_links),
                "call_activity_links": call_activity_links,
                "call_activities_found": call_activities_found,
                "resolved_calls": resolved_calls,
                "requires_inference_count": requires_inference_count,
                "unresolved_call_count": unresolved_call_count,
                "parallel_paths_found": parallel_paths_found,
                "parallel_token_max": parallel_token_max,
                "cycles_detected": cycle_count > 0,
                "cycle_count": cycle_count,
                "loop_density": loop_density,
            },
        }

    @staticmethod
    def _resolve_edge_type(event: ProcessEventDTO) -> str:
        if event.activity_type == "boundaryEvent":
            interrupting = bool(event.extra.get("interrupting", False))
            return "cancellation_edge" if interrupting else "fork_edge"
        return "sequence"

    @staticmethod
    def _append_edge(
        *,
        edges: List[Dict[str, Any]],
        edge_keys: set[Tuple[str, str, str]],
        source: str,
        target: str,
        edge_type: str,
        **extra: Any,
    ) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, edge_type)
        if key in edge_keys:
            return
        edge_keys.add(key)
        payload = {"source": source, "target": target, "edge_type": edge_type}
        payload.update(extra)
        edges.append(payload)

    @staticmethod
    def _remove_edge(
        *,
        edges: List[Dict[str, Any]],
        edge_keys: set[Tuple[str, str, str]],
        source: str,
        target: str,
        edge_type: str,
    ) -> None:
        key = (source, target, edge_type)
        if key in edge_keys:
            edge_keys.remove(key)
        for idx in range(len(edges) - 1, -1, -1):
            edge = edges[idx]
            if edge.get("source") == source and edge.get("target") == target and edge.get("edge_type") == edge_type:
                edges.pop(idx)

    @classmethod
    def _resolve_scope_meta(
        cls,
        *,
        node_id: str,
        event_by_node_id: Dict[str, ProcessEventDTO],
        node_id_by_act_inst: Dict[str, str],
        case_id: str,
        scope_level_cache: Dict[str, int],
        subprocess_parent_cache: Dict[str, str | None],
    ) -> Tuple[int, str | None]:
        if node_id in scope_level_cache:
            return scope_level_cache[node_id], subprocess_parent_cache.get(node_id)

        event = event_by_node_id.get(node_id)
        if event is None:
            scope_level_cache[node_id] = 0
            subprocess_parent_cache[node_id] = None
            return 0, None

        level = 0
        parent_subprocess_id: str | None = None
        parent_act = cls._norm_text(event.parent_act_inst_id)
        visited: set[str] = set()
        while parent_act and parent_act != case_id and parent_act not in visited:
            visited.add(parent_act)
            parent_node_id = node_id_by_act_inst.get(parent_act)
            if not parent_node_id:
                break
            parent_event = event_by_node_id.get(parent_node_id)
            if parent_event is None:
                break
            level += 1
            if parent_subprocess_id is None and cls._norm_text(parent_event.activity_type) == "subProcess":
                parent_subprocess_id = parent_node_id
            parent_act = cls._norm_text(parent_event.parent_act_inst_id)

        scope_level_cache[node_id] = level
        subprocess_parent_cache[node_id] = parent_subprocess_id
        return level, parent_subprocess_id

    @staticmethod
    def _is_subprocess_phantom_event(event: ProcessEventDTO) -> bool:
        activity_type = str(event.activity_type or "").lower()
        return "startevent" in activity_type or "endevent" in activity_type

    @classmethod
    def _is_technical_subprocess_node(
        cls,
        *,
        event: ProcessEventDTO,
        case_id: str,
    ) -> bool:
        parent_act_id = cls._norm_text(event.parent_act_inst_id)
        if not parent_act_id or parent_act_id == case_id:
            return False
        return cls._is_subprocess_phantom_event(event)

    def _collapse_subprocess_containers(
        self,
        *,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        edge_keys: set[Tuple[str, str, str]],
        container_entry_exit: Dict[str, Tuple[str, str]],
        case_id: str,
    ) -> None:
        subprocess_nodes = [
            node
            for node in nodes
            if node.get("case_id") == case_id and self._norm_text(node.get("activity_type")) == "subProcess"
        ]
        if not subprocess_nodes:
            return

        removed_subprocess_activity: Dict[str, str] = {}
        for container in subprocess_nodes:
            container_id = self._norm_text(container.get("id"))
            if not container_id:
                continue
            removed_subprocess_activity[container_id] = self._norm_text(container.get("activity_def_id"))
            entry_node_id, exit_node_id = container_entry_exit.get(container_id, ("", ""))

            incoming = [
                dict(edge)
                for edge in edges
                if self._norm_text(edge.get("target")) == container_id and self._norm_text(edge.get("source")) != exit_node_id
            ]
            outgoing = [
                dict(edge)
                for edge in edges
                if self._norm_text(edge.get("source")) == container_id and self._norm_text(edge.get("target")) != entry_node_id
            ]

            if not entry_node_id and not exit_node_id:
                # No internal children detected: direct passthrough from incoming to outgoing.
                for incoming_edge in incoming:
                    source = self._norm_text(incoming_edge.get("source"))
                    if not source:
                        continue
                    for outgoing_edge in outgoing:
                        target = self._norm_text(outgoing_edge.get("target"))
                        if not target:
                            continue
                        edge_type = self._norm_text(outgoing_edge.get("edge_type")) or self._norm_text(incoming_edge.get("edge_type")) or "sequence"
                        extra = {
                            key: value
                            for key, value in outgoing_edge.items()
                            if key not in {"source", "target", "edge_type"}
                        }
                        self._append_edge(
                            edges=edges,
                            edge_keys=edge_keys,
                            source=source,
                            target=target,
                            edge_type=edge_type,
                            **extra,
                        )
            else:
                for incoming_edge in incoming:
                    source = self._norm_text(incoming_edge.get("source"))
                    if not source or not entry_node_id:
                        continue
                    edge_type = self._norm_text(incoming_edge.get("edge_type")) or "sequence"
                    extra = {
                        key: value
                        for key, value in incoming_edge.items()
                        if key not in {"source", "target", "edge_type"}
                    }
                    self._append_edge(
                        edges=edges,
                        edge_keys=edge_keys,
                        source=source,
                        target=entry_node_id,
                        edge_type=edge_type,
                        **extra,
                    )

                for outgoing_edge in outgoing:
                    target = self._norm_text(outgoing_edge.get("target"))
                    if not target or not exit_node_id:
                        continue
                    edge_type = self._norm_text(outgoing_edge.get("edge_type")) or "sequence"
                    extra = {
                        key: value
                        for key, value in outgoing_edge.items()
                        if key not in {"source", "target", "edge_type"}
                    }
                    self._append_edge(
                        edges=edges,
                        edge_keys=edge_keys,
                        source=exit_node_id,
                        target=target,
                        edge_type=edge_type,
                        **extra,
                    )

            # Remove all edges touching subprocess container.
            for edge in list(edges):
                source = self._norm_text(edge.get("source"))
                target = self._norm_text(edge.get("target"))
                if source != container_id and target != container_id:
                    continue
                self._remove_edge(
                    edges=edges,
                    edge_keys=edge_keys,
                    source=source,
                    target=target,
                    edge_type=self._norm_text(edge.get("edge_type")) or "sequence",
                )

        removed_ids = set(removed_subprocess_activity.keys())
        if not removed_ids:
            return
        nodes[:] = [node for node in nodes if self._norm_text(node.get("id")) not in removed_ids]
        for node in nodes:
            parent_subprocess_id = self._norm_text(node.get("parent_subprocess_id"))
            if parent_subprocess_id in removed_subprocess_activity:
                node["parent_subprocess_id"] = None
                node["parent_subprocess_activity_def_id"] = removed_subprocess_activity[parent_subprocess_id]
                node["scope_level"] = max(0, self._safe_int(node.get("scope_level")) - 1)

    @classmethod
    def _rewire_parallel_subprocess_branches(
        cls,
        *,
        edges: List[Dict[str, Any]],
        edge_keys: set[Tuple[str, str, str]],
        event_by_node_id: Dict[str, ProcessEventDTO],
        ordered_scope_nodes: Sequence[str],
        case_id: str,
        scope_key: str,
    ) -> None:
        if len(ordered_scope_nodes) < 4:
            return
        ordered_scope = [
            node_id
            for node_id in ordered_scope_nodes
            if node_id in event_by_node_id
        ]
        for idx in range(0, len(ordered_scope) - 3):
            split_id = ordered_scope[idx]
            a_id = ordered_scope[idx + 1]
            b_id = ordered_scope[idx + 2]
            join_id = ordered_scope[idx + 3]

            split_ev = event_by_node_id.get(split_id)
            a_ev = event_by_node_id.get(a_id)
            b_ev = event_by_node_id.get(b_id)
            join_ev = event_by_node_id.get(join_id)
            if not split_ev or not a_ev or not b_ev or not join_ev:
                continue

            split_type = str(split_ev.activity_type or "").lower()
            join_type = str(join_ev.activity_type or "").lower()
            if "parallelgateway" not in split_type:
                continue
            if "gateway" not in join_type:
                continue

            a_start = a_ev.start_time or datetime.min
            b_start = b_ev.start_time or datetime.min
            a_end = a_ev.end_time or a_ev.start_time or datetime.max
            b_end = b_ev.end_time or b_ev.start_time or datetime.max
            overlap = not (a_end < b_start or b_end < a_start)
            same_parent_exec = bool(
                cls._norm_text(a_ev.parent_execution_id)
                and cls._norm_text(a_ev.parent_execution_id) == cls._norm_text(b_ev.parent_execution_id)
            )
            if not overlap and not same_parent_exec:
                continue

            # Remove sequential chain between split -> branch A -> branch B -> join.
            cls._remove_edge(edges=edges, edge_keys=edge_keys, source=split_id, target=a_id, edge_type="sequence")
            cls._remove_edge(edges=edges, edge_keys=edge_keys, source=a_id, target=b_id, edge_type="sequence")
            cls._remove_edge(edges=edges, edge_keys=edge_keys, source=b_id, target=join_id, edge_type="sequence")
            cls._remove_edge(edges=edges, edge_keys=edge_keys, source=b_id, target=a_id, edge_type="sequence")

            # Add parallel edges.
            cls._append_edge(
                edges=edges,
                edge_keys=edge_keys,
                source=split_id,
                target=a_id,
                edge_type="parallel_branch",
                parallel_group_id=f"{case_id}:{scope_key}:{split_id}",
                token_count_in_group=2,
            )
            cls._append_edge(
                edges=edges,
                edge_keys=edge_keys,
                source=split_id,
                target=b_id,
                edge_type="parallel_branch",
                parallel_group_id=f"{case_id}:{scope_key}:{split_id}",
                token_count_in_group=2,
            )
            cls._append_edge(
                edges=edges,
                edge_keys=edge_keys,
                source=a_id,
                target=join_id,
                edge_type="parallel_branch",
                parallel_group_id=f"{case_id}:{scope_key}:{split_id}",
                token_count_in_group=2,
            )
            cls._append_edge(
                edges=edges,
                edge_keys=edge_keys,
                source=b_id,
                target=join_id,
                edge_type="parallel_branch",
                parallel_group_id=f"{case_id}:{scope_key}:{split_id}",
                token_count_in_group=2,
            )

    def _pick_execution_anchor(
        self,
        *,
        ordered_by_execution: Dict[str, List[Tuple[datetime, str]]],
        parent_execution_id: str,
        target_ts: datetime,
    ) -> str:
        parent_items = ordered_by_execution.get(parent_execution_id, [])
        if not parent_items:
            return ""
        last_before = ""
        for timestamp, node_id in parent_items:
            if timestamp <= target_ts:
                last_before = node_id
            else:
                break
        return last_before or parent_items[-1][1]

    @staticmethod
    def _event_order_key(event: ProcessEventDTO) -> Tuple[int, int, datetime, datetime, int, str]:
        sequence_counter = event.sequence_counter if event.sequence_counter is not None else 0
        has_sequence = 0 if event.sequence_counter is not None else 1
        start_ts = event.start_time or datetime.min
        # Active events without end timestamp should not jump ahead of completed siblings
        # with the same start time.
        end_ts = event.end_time or datetime.max
        activity_type = str(event.activity_type or "").lower()
        if "start" in activity_type:
            activity_priority = 0
        elif "gateway" in activity_type:
            activity_priority = 1
        elif "subprocess" in activity_type:
            activity_priority = 2
        elif "end" in activity_type:
            activity_priority = 4
        else:
            activity_priority = 3
        act_inst = event.act_inst_id or event.task_id or event.activity_def_id or ""
        return has_sequence, sequence_counter, start_ts, end_ts, activity_priority, act_inst

    @classmethod
    def _is_sequence_pair(cls, prev: ProcessEventDTO, curr: ProcessEventDTO) -> bool:
        prev_exec = cls._norm_text(prev.execution_id)
        curr_exec = cls._norm_text(curr.execution_id)
        same_execution = bool(prev_exec and prev_exec == curr_exec)

        curr_parent_act = cls._norm_text(curr.parent_act_inst_id)
        prev_act = cls._norm_text(prev.act_inst_id)
        parent_chain = bool(curr_parent_act and prev_act and curr_parent_act == prev_act)

        prev_parent_act = cls._norm_text(prev.parent_act_inst_id)
        same_parent_act = bool(prev_parent_act and prev_parent_act == curr_parent_act)

        return same_execution or parent_chain or same_parent_act

    @classmethod
    def _is_sequence_pair_for_case(
        cls,
        prev: ProcessEventDTO,
        curr: ProcessEventDTO,
        execution_meta: Dict[str, Dict[str, Any]],
    ) -> bool:
        if cls._is_sequence_pair(prev, curr):
            return True

        prev_exec = cls._norm_text(prev.execution_id)
        curr_exec = cls._norm_text(curr.execution_id)
        curr_parent_exec = cls._norm_text(curr.parent_execution_id) or cls._norm_text(
            execution_meta.get(curr_exec, {}).get("parent_execution_id")
        )

        parent_exec_chain = bool(prev_exec and curr_parent_exec and prev_exec == curr_parent_exec)
        return parent_exec_chain

    @classmethod
    def _resolve_boundary_type(cls, event: ProcessEventDTO) -> str | None:
        if cls._norm_text(event.activity_type) != "boundaryEvent":
            return None
        interrupting = bool(event.extra.get("interrupting", False))
        return "interrupting" if interrupting else "non_interrupting"

    @staticmethod
    def _timestamp_delta(prev: ProcessEventDTO, curr: ProcessEventDTO) -> float | None:
        prev_ts = prev.end_time or prev.start_time
        curr_ts = curr.start_time or curr.end_time
        if prev_ts is None or curr_ts is None:
            return None
        return float((curr_ts - prev_ts).total_seconds())

    @classmethod
    def _index_execution_meta_for_case(
        cls,
        *,
        execution_rows: Sequence[Dict[str, Any]],
        case_id: str,
        depth_limit: int,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for row in execution_rows:
            row_case = cls._norm_text(row.get("case_id"))
            if row_case and row_case != case_id:
                continue
            depth = cls._safe_int(row.get("depth") or row.get("scope_depth"))
            if depth_limit >= 0 and depth > depth_limit:
                continue
            execution_id = cls._norm_text(row.get("execution_id") or row.get("id"))
            if not execution_id:
                continue
            result[execution_id] = {
                "parent_execution_id": cls._norm_text(row.get("parent_execution_id") or row.get("parent_id")),
                "is_concurrent": cls._safe_bool(row.get("is_concurrent")),
                "is_scope": cls._safe_bool(row.get("is_scope")),
                "is_event_scope": cls._safe_bool(row.get("is_event_scope")),
                "scope_depth": depth,
                "execution_rev": cls._safe_int_or_none(row.get("rev") or row.get("execution_rev")),
            }
        return result

    @classmethod
    def _index_multi_instance_variables(
        cls,
        rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[Tuple[str, str], int]]:
        result: Dict[str, Dict[Tuple[str, str], int]] = {
            "loop_counter": {},
            "nr_of_instances": {},
            "nr_of_completed": {},
        }
        for row in rows:
            case_id = cls._norm_text(row.get("case_id"))
            execution_id = cls._norm_text(row.get("execution_id"))
            if not case_id or not execution_id:
                continue
            var_name = cls._norm_text(row.get("var_name") or row.get("name")).lower()
            value = cls._safe_int_or_none(row.get("long_value") or row.get("value"))
            if value is None:
                continue
            key = (case_id, execution_id)
            if var_name == "loopcounter":
                result["loop_counter"][key] = value
            elif var_name == "nrofinstances":
                result["nr_of_instances"][key] = value
            elif var_name == "nrofcompletedinstances":
                result["nr_of_completed"][key] = value
        return result

    @classmethod
    def _index_process_instance_links(
        cls,
        rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            parent_case = cls._norm_text(
                row.get("super_proc_inst_id")
                or row.get("super_process_instance_id")
                or row.get("SUPER_PROCESS_INSTANCE_ID_")
            )
            if not parent_case:
                continue
            result[parent_case].append(dict(row))
        return dict(result)

    @staticmethod
    def _build_projection_payload(events: Sequence[ProcessEventDTO]) -> Dict[str, Any]:
        by_case: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
        for event in events:
            if InstanceGraphAssemblerService._is_technical_subprocess_node(event=event, case_id=event.case_id):
                continue
            by_case[event.case_id].append(event)

        node_counts: Dict[str, int] = defaultdict(int)
        transition_counts: Dict[str, int] = defaultdict(int)
        subprocess_collapsed: Dict[str, Dict[str, Any]] = {}
        for case_events in by_case.values():
            ordered = sorted(case_events, key=lambda item: item.start_time or datetime.min)
            children_by_parent: Dict[str, List[ProcessEventDTO]] = defaultdict(list)
            for event in ordered:
                parent_act = str(event.parent_act_inst_id or "").strip()
                if parent_act:
                    children_by_parent[parent_act].append(event)
            for idx, event in enumerate(ordered):
                node_counts[event.activity_def_id] += 1
                if idx > 0:
                    prev = ordered[idx - 1]
                    transition_counts[f"{prev.activity_def_id}->{event.activity_def_id}"] += 1
                if str(event.activity_type or "") == "subProcess":
                    children = children_by_parent.get(str(event.act_inst_id or "").strip(), [])
                    durations = [float(child.duration_ms or 0.0) for child in children if child.duration_ms is not None]
                    activity_counts = defaultdict(int)
                    for child in children:
                        activity_counts[str(child.activity_def_id)] += 1
                    has_cycles = any(count > 1 for count in activity_counts.values())
                    max_parallel = 0
                    grouped_by_start: Dict[float, int] = defaultdict(int)
                    for child in children:
                        if child.start_time is None:
                            continue
                        grouped_by_start[child.start_time.timestamp()] += 1
                    if grouped_by_start:
                        max_parallel = max(grouped_by_start.values())
                    subprocess_collapsed[str(event.act_inst_id)] = {
                        "internal_node_count": len(children),
                        "internal_duration_avg": float(sum(durations) / len(durations)) if durations else 0.0,
                        "has_internal_cycles": bool(has_cycles),
                        "internal_parallel_max": int(max_parallel),
                    }
        return {
            "node_counts": dict(node_counts),
            "transition_counts": dict(transition_counts),
            "subprocess_collapsed": subprocess_collapsed,
            "mode": "collapsed_projection",
        }

    @classmethod
    def _index_process_variables(
        cls,
        rows: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
        by_case: Dict[str, Dict[str, Any]] = defaultdict(dict)
        by_case_execution: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)
        for row in rows:
            case_id = cls._norm_text(row.get("case_id"))
            var_name = cls._norm_text(row.get("var_name") or row.get("name"))
            if not case_id or not var_name:
                continue
            value = cls._resolve_variable_value(row)
            execution_id = cls._norm_text(row.get("execution_id"))
            if execution_id:
                by_case_execution[(case_id, execution_id)][var_name] = value
            else:
                by_case[case_id][var_name] = value
        return dict(by_case), dict(by_case_execution)

    @classmethod
    def _resolve_variable_value(cls, row: Dict[str, Any]) -> Any:
        value = row.get("typed_value")
        if value is not None and value != "":
            return value
        for key in ("double_value", "long_value", "text_value", "text2_value", "value"):
            value = row.get(key)
            if value is not None and value != "":
                if key in {"double_value", "long_value"}:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return value
                return value
        return None

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_int_or_none(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_bool(value: Any) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None

    @staticmethod
    def _norm_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"", "nan", "nat", "none", "null", "<na>", "na", "n/a"}:
            return ""
        return text

    @staticmethod
    def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "canonical_mode": str(config.get("canonical_mode", "activity-centric")).strip().lower(),
            "subprocess_graph_mode": str(config.get("subprocess_graph_mode", "flattened-no-subprocess-node")).strip().lower(),
            "fallback_on_unresolved_call": bool(config.get("fallback_on_unresolved_call", True)),
            "execution_tree_depth_limit": int(config.get("execution_tree_depth_limit", 4)),
            "execution_missing_degrade_threshold": float(config.get("execution_missing_degrade_threshold", 0.5)),
            "max_execution_nodes_per_case": int(config.get("max_execution_nodes_per_case", 5000)),
            "extreme_mi_strategy": str(config.get("extreme_mi_strategy", "aggregate")).strip().lower(),
            "fallback_min_coverage_ratio": float(config.get("fallback_min_coverage_ratio", 0.05)),
            "coverage_hysteresis_fetches": int(config.get("coverage_hysteresis_fetches", 2)),
        }
