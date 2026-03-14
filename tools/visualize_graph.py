"""CLI utility to visualize a single instance graph (IG) for sanity-checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from graphviz import Digraph
from graphviz.backend import ExecutableNotFound
import matplotlib.pyplot as plt
import networkx as nx

from src.adapters.ingestion.camunda_runtime_adapter import CamundaRuntimeAdapter
from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.services.instance_graph_assembler_service import InstanceGraphAssemblerService
from src.cli import load_yaml_config
from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO
from src.domain.entities.raw_trace import RawTrace
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


@dataclass
class _CaseSummary:
    case_id: str
    num_events: int
    start_ts: float
    end_ts: float
    has_call_activity: bool


_NULLISH_TEXT = {"", "nan", "nat", "none", "null", "<na>", "na", "n/a"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in _NULLISH_TEXT:
        return ""
    return text


def _parse_iso(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = _clean_text(value)
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _resolve_dataset_name(data_cfg: Mapping[str, Any], fallback_path: str) -> str:
    candidates = [
        data_cfg.get("dataset_name"),
        data_cfg.get("dataset_label"),
        data_cfg.get("process_name"),
    ]
    fallback = _clean_text(fallback_path)
    if fallback:
        candidates.append(Path(fallback).stem)
    for candidate in candidates:
        value = _clean_text(candidate)
        if value:
            return value
    return "default_dataset"


def _build_mapping_with_dataset(mapping_cfg: Mapping[str, Any], dataset_name: str) -> Dict[str, Any]:
    result = dict(mapping_cfg)
    xes_cfg_raw = result.get("xes_adapter", {})
    xes_cfg = dict(xes_cfg_raw) if isinstance(xes_cfg_raw, dict) else {}
    xes_cfg.setdefault("dataset_name", dataset_name)
    result["xes_adapter"] = xes_cfg

    camunda_cfg_raw = result.get("camunda_adapter", {})
    camunda_cfg = dict(camunda_cfg_raw) if isinstance(camunda_cfg_raw, dict) else {}
    camunda_cfg.setdefault("dataset_name", dataset_name)
    camunda_cfg.setdefault("process_name", dataset_name)
    result["camunda_adapter"] = camunda_cfg
    return result


def _resolve_adapter_kind(mapping_cfg: Mapping[str, Any]) -> str:
    adapter = _clean_text(mapping_cfg.get("adapter", "")).lower()
    if adapter:
        return adapter
    if isinstance(mapping_cfg.get("camunda_adapter"), dict):
        return "camunda"
    return "xes"


def _build_camunda_mapping_from_legacy(cfg: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    camunda_cfg = cfg.get("camunda", {})
    if not isinstance(camunda_cfg, dict):
        return {}
    process_name = _clean_text(
        data_cfg.get("dataset_name")
        or data_cfg.get("dataset_label")
        or data_cfg.get("process_name")
        or camunda_cfg.get("process_name")
        or "default_process"
    ) or "default_process"
    camunda_adapter_cfg = dict(camunda_cfg)
    camunda_adapter_cfg.setdefault("process_name", process_name)
    runtime_cfg = camunda_cfg.get("runtime", {})
    if isinstance(runtime_cfg, dict):
        camunda_adapter_cfg["runtime"] = dict(runtime_cfg)
    return {
        "adapter": "camunda",
        "camunda_adapter": camunda_adapter_cfg,
    }


def _resolve_time_range(camunda_cfg: Mapping[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    since = _parse_iso(camunda_cfg.get("since"))
    until = _parse_iso(camunda_cfg.get("until"))
    if since is not None or until is not None:
        return since, until

    lookback_hours = int(camunda_cfg.get("lookback_hours", 0) or 0)
    if lookback_hours <= 0:
        return None, None
    now = datetime.utcnow()
    return now - timedelta(hours=lookback_hours), now


def _extract_camunda_blocks(config: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    data_cfg = config.get("data", {})
    mapping_cfg = config.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}
    if not mapping_cfg and isinstance(config.get("camunda"), dict):
        mapping_cfg = _build_camunda_mapping_from_legacy(config, data_cfg)

    dataset_name = _resolve_dataset_name(data_cfg, str(data_cfg.get("log_path", "")))
    mapping_cfg = _build_mapping_with_dataset(mapping_cfg, dataset_name)
    camunda_cfg = mapping_cfg.get("camunda_adapter", {}) if isinstance(mapping_cfg.get("camunda_adapter"), dict) else {}
    return dict(data_cfg), dict(mapping_cfg), dict(camunda_cfg)


def _load_camunda_payload_from_config(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    data_cfg, mapping_cfg, camunda_cfg = _extract_camunda_blocks(cfg)
    if _resolve_adapter_kind(mapping_cfg) != "camunda":
        raise ValueError("visualize-graph currently expects mapping.adapter='camunda' for config mode.")

    process_name = str(
        camunda_cfg.get("process_name")
        or data_cfg.get("dataset_name")
        or data_cfg.get("dataset_label")
        or data_cfg.get("process_name")
        or "default_process"
    ) or "default_process"
    version_key = _clean_text(camunda_cfg.get("version_key", ""))

    runtime_cfg_raw = camunda_cfg.get("runtime", {})
    runtime_cfg = dict(runtime_cfg_raw) if isinstance(runtime_cfg_raw, dict) else dict(camunda_cfg)

    since, until = _resolve_time_range(camunda_cfg)
    adapter = CamundaRuntimeAdapter(runtime_cfg)

    events, diagnostics = adapter.fetch_historic_activity_events(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )

    # Smart fallback for lookback-limited configs.
    has_explicit_bounds = bool(_clean_text(camunda_cfg.get("since", "")) or _clean_text(camunda_cfg.get("until", "")))
    lookback_hours = int(camunda_cfg.get("lookback_hours", 0) or 0)
    if not events and lookback_hours > 0 and not has_explicit_bounds:
        retry_camunda_cfg = dict(camunda_cfg)
        retry_camunda_cfg["lookback_hours"] = 0
        since, until = _resolve_time_range(retry_camunda_cfg)
        events, diagnostics = adapter.fetch_historic_activity_events(
            process_name=process_name,
            version_key=version_key,
            since=since,
            until=until,
        )

    depth_limit = int(camunda_cfg.get("execution_tree_depth_limit", 4))
    execution_rows = adapter.fetch_runtime_execution_tree(
        process_name=process_name,
        version_key=version_key,
        depth_limit=depth_limit,
        since=since,
        until=until,
    )
    variables_rows = adapter.fetch_multi_instance_variables(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )
    identity_rows = adapter.fetch_identity_links(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )
    task_rows = adapter.fetch_historic_task_events(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )
    process_variables_rows = adapter.fetch_process_variables(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )
    process_instance_links = adapter.fetch_process_instance_links(
        process_name=process_name,
        version_key=version_key,
        since=since,
        until=until,
    )

    return {
        "process_name": process_name,
        "version_key": version_key,
        "camunda_cfg": camunda_cfg,
        "events": events,
        "execution_rows": execution_rows,
        "variables_rows": variables_rows,
        "identity_rows": identity_rows,
        "task_rows": task_rows,
        "process_variables_rows": process_variables_rows,
        "process_instance_links": process_instance_links,
        "diagnostics": diagnostics,
    }


def _load_xes_traces_from_config(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    data_cfg = cfg.get("data", {})
    mapping_cfg = cfg.get("mapping", {})
    if not isinstance(mapping_cfg, dict):
        mapping_cfg = {}

    log_path = _clean_text(data_cfg.get("log_path", ""))
    if not log_path:
        raise ValueError("Config must define data.log_path for XES visualize-graph mode.")

    dataset_name = _resolve_dataset_name(data_cfg, log_path)
    mapping_cfg = _build_mapping_with_dataset(mapping_cfg, dataset_name)
    traces = list(XESAdapter().read(log_path, mapping_cfg))
    return {
        "dataset_name": dataset_name,
        "traces": traces,
    }


def _load_xes_traces_from_data(data_path: str) -> Dict[str, Any]:
    dataset_name = Path(data_path).stem or "default_dataset"
    mapping_cfg = _build_mapping_with_dataset({}, dataset_name)
    traces = list(XESAdapter().read(data_path, mapping_cfg))
    return {
        "dataset_name": dataset_name,
        "traces": traces,
    }


def _summaries_from_camunda_events(events: Sequence[ProcessEventDTO]) -> Dict[str, _CaseSummary]:
    by_case: Dict[str, List[ProcessEventDTO]] = {}
    for event in events:
        by_case.setdefault(event.case_id, []).append(event)

    summaries: Dict[str, _CaseSummary] = {}
    for case_id, case_events in by_case.items():
        ordered = sorted(case_events, key=lambda item: (item.start_time or datetime.min, item.end_time or datetime.min))
        start_ts = float((ordered[0].start_time or ordered[0].end_time or datetime.min).timestamp())
        end_ts = float((ordered[-1].end_time or ordered[-1].start_time or datetime.min).timestamp())
        has_call = any(bool(_clean_text(item.call_proc_inst_id)) for item in ordered)
        summaries[case_id] = _CaseSummary(
            case_id=case_id,
            num_events=len(ordered),
            start_ts=start_ts,
            end_ts=end_ts,
            has_call_activity=has_call,
        )
    return summaries


def _summaries_from_traces(traces: Sequence[RawTrace]) -> Dict[str, _CaseSummary]:
    summaries: Dict[str, _CaseSummary] = {}
    for trace in traces:
        if not trace.events:
            continue
        start_ts = float(trace.events[0].timestamp)
        end_ts = float(trace.events[-1].timestamp)
        summaries[trace.case_id] = _CaseSummary(
            case_id=trace.case_id,
            num_events=len(trace.events),
            start_ts=start_ts,
            end_ts=end_ts,
            has_call_activity=False,
        )
    return summaries


def _rank_case_ids(summaries: Mapping[str, _CaseSummary], pick: str, seed: int) -> List[str]:
    items = list(summaries.values())
    if pick == "random":
        ids = sorted(summary.case_id for summary in items)
        rnd = random.Random(seed)
        rnd.shuffle(ids)
        return ids
    if pick == "longest":
        return [summary.case_id for summary in sorted(items, key=lambda s: (-s.num_events, -s.end_ts, s.case_id))]
    if pick == "shortest":
        return [summary.case_id for summary in sorted(items, key=lambda s: (s.num_events, -s.end_ts, s.case_id))]
    if pick == "with-call-activity":
        filtered = [summary for summary in items if summary.has_call_activity]
        return [summary.case_id for summary in sorted(filtered, key=lambda s: (-s.end_ts, -s.num_events, s.case_id))]
    # latest
    return [summary.case_id for summary in sorted(items, key=lambda s: (-s.end_ts, -s.num_events, s.case_id))]


def _select_case_id(
    summaries: Mapping[str, _CaseSummary],
    *,
    case_id: Optional[str],
    pick: str,
    index: int,
    seed: int,
) -> str:
    if not summaries:
        raise ValueError("No process instances were found.")

    if case_id is not None:
        requested = _clean_text(case_id)
        if requested in summaries:
            return requested
        available = sorted(summaries.keys())
        raise ValueError(f"Case id '{requested}' not found. Available sample: {available[:10]}")

    ranked = _rank_case_ids(summaries, pick=pick, seed=seed)
    if not ranked:
        raise ValueError(f"No process instances match pick strategy '{pick}'.")
    if index < 0 or index >= len(ranked):
        raise ValueError(f"Index {index} is out of range for strategy '{pick}' (available: {len(ranked)}).")
    return ranked[index]


def _print_case_table(
    summaries: Mapping[str, _CaseSummary],
    *,
    pick: str,
    seed: int,
    top: int,
) -> None:
    ranked = _rank_case_ids(summaries, pick=pick, seed=seed)
    print(f"Found {len(ranked)} process instances. Showing top {min(top, len(ranked))} by '{pick}':")
    print("idx\tcase_id\tevents\tstart_ts\tend_ts\thas_call")
    for idx, case in enumerate(ranked[:top]):
        summary = summaries[case]
        print(
            f"{idx}\t{summary.case_id}\t{summary.num_events}\t"
            f"{datetime.utcfromtimestamp(summary.start_ts).isoformat()}\t"
            f"{datetime.utcfromtimestamp(summary.end_ts).isoformat()}\t"
            f"{summary.has_call_activity}"
        )


def _filter_case_rows(payload: Mapping[str, Any], case_id: str) -> Dict[str, Any]:
    events: List[ProcessEventDTO] = [event for event in payload["events"] if event.case_id == case_id]
    task_ids = {_clean_text(event.task_id) for event in events if _clean_text(event.task_id)}
    execution_ids = {_clean_text(event.execution_id) for event in events if _clean_text(event.execution_id)}

    def _row_case_match(row: Mapping[str, Any]) -> bool:
        value = _clean_text(row.get("case_id", ""))
        return value == case_id

    execution_rows = [row for row in payload["execution_rows"] if _row_case_match(row)]
    variables_rows = [row for row in payload["variables_rows"] if _row_case_match(row)]
    process_variables_rows = [row for row in payload["process_variables_rows"] if _row_case_match(row)]

    identity_rows: List[Dict[str, Any]] = []
    for row in payload["identity_rows"]:
        if _row_case_match(row):
            identity_rows.append(row)
            continue
        row_task_id = _clean_text(row.get("task_id", ""))
        if row_task_id and row_task_id in task_ids:
            identity_rows.append(row)

    task_rows: List[Dict[str, Any]] = []
    for row in payload["task_rows"]:
        if _row_case_match(row):
            task_rows.append(row)
            continue
        row_task_id = _clean_text(row.get("task_id", ""))
        if row_task_id and row_task_id in task_ids:
            task_rows.append(row)

    # Backfill execution-specific variables that may miss case_id in some exports.
    for row in payload["process_variables_rows"]:
        case_value = _clean_text(row.get("case_id", ""))
        if case_value:
            continue
        execution_id = _clean_text(row.get("execution_id", ""))
        if execution_id and execution_id in execution_ids:
            process_variables_rows.append(row)

    return {
        "events": events,
        "execution_rows": execution_rows,
        "variables_rows": variables_rows,
        "identity_rows": identity_rows,
        "task_rows": task_rows,
        "process_variables_rows": process_variables_rows,
        "process_instance_links": [
            row
            for row in payload.get("process_instance_links", [])
            if _row_case_match(row)
            or _clean_text(row.get("super_proc_inst_id") or row.get("super_process_instance_id") or "") == case_id
        ],
    }


def _truncate_events(events: Sequence[ProcessEventDTO], max_nodes: int) -> List[ProcessEventDTO]:
    if max_nodes <= 0 or len(events) <= max_nodes:
        return list(events)
    ordered = sorted(events, key=lambda item: (item.start_time or datetime.min, item.end_time or datetime.min))
    return ordered[:max_nodes]


def _build_camunda_case_graph(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    mode: Optional[str],
    max_nodes: int,
) -> Dict[str, Any]:
    selected = _filter_case_rows(payload, case_id)
    events = _truncate_events(selected["events"], max_nodes=max_nodes)
    if not events:
        raise ValueError(f"No events found for case_id '{case_id}'.")

    repository = InMemoryNetworkXRepository()
    assembler = InstanceGraphAssemblerService(knowledge_port=repository)
    cfg = dict(payload.get("camunda_cfg", {}))
    if mode:
        cfg["canonical_mode"] = mode

    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, RuntimeFetchDiagnosticsDTO):
        diagnostics_copy = diagnostics.model_copy(deep=True)
    else:
        diagnostics_copy = RuntimeFetchDiagnosticsDTO()

    result = assembler.build(
        process_name=str(payload["process_name"]),
        version_key=str(payload.get("version_key", "")),
        events=events,
        execution_rows=selected["execution_rows"],
        variables_rows=selected["variables_rows"],
        identity_rows=selected["identity_rows"],
        diagnostics=diagnostics_copy,
        config=cfg,
        task_rows=selected["task_rows"],
        process_variables_rows=selected["process_variables_rows"],
        process_instance_links=selected["process_instance_links"],
    )

    graph = dict(result.graph)
    metadata = dict(graph.get("metadata", {}))
    metadata["selected_case_id"] = case_id
    metadata["build_mode"] = result.mode
    graph["metadata"] = metadata
    return graph


def _build_xes_case_graph(traces: Sequence[RawTrace], *, case_id: str, max_nodes: int) -> Dict[str, Any]:
    selected = None
    for trace in traces:
        if trace.case_id == case_id:
            selected = trace
            break
    if selected is None or not selected.events:
        raise ValueError(f"No trace found for case_id '{case_id}'.")

    events = selected.events[:max_nodes] if max_nodes > 0 else list(selected.events)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for idx, event in enumerate(events):
        node_id = str(event.activity_instance_id or f"{case_id}:{idx}")
        nodes.append(
            {
                "id": node_id,
                "case_id": case_id,
                "activity_def_id": event.activity_id,
                "activity_type": str(event.extra.get("activity_type") or "event"),
                "execution_id": str(event.extra.get("execution_id") or ""),
                "assigned_executor": str(event.extra.get("assigned_executor") or "") or None,
                "executed_by": str(event.extra.get("executed_by") or "") or None,
                "potential_executor_users": _parse_csv_list(event.extra.get("potential_executor_users")) or None,
                "potential_executor_groups": _parse_csv_list(event.extra.get("potential_executor_groups")) or None,
                "process_variables": event.extra.get("process_variables") if isinstance(event.extra.get("process_variables"), dict) else None,
                "call_activity_link": event.extra.get("call_activity_link") if isinstance(event.extra.get("call_activity_link"), dict) else None,
            }
        )
        if idx == 0:
            continue
        prev_node = nodes[idx - 1]["id"]
        edges.append({"source": prev_node, "target": node_id, "edge_type": "sequence"})

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "mode": "xes_sequence",
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "selected_case_id": case_id,
            "build_mode": "xes_sequence",
            "num_call_activity_links": 0,
            "call_activity_links": [],
        },
    }


def _parse_csv_list(value: Any) -> List[str]:
    if value is None:
        return []
    tokens = [token.strip() for token in str(value).split(",")]
    return [token for token in tokens if token and token.lower() not in _NULLISH_TEXT]


def _format_node_label(
    *,
    activity_def_id: str,
    activity_name: str,
    activity_type: str,
    fallback_id: str,
) -> str:
    display_id = activity_def_id or fallback_id
    display_name = activity_name or display_id
    lines: List[str] = [display_name]
    if display_name != display_id:
        lines.append(f"id: {display_id}")
    if activity_type:
        lines.append(f"[{activity_type}]")
    return "\\n".join(lines)


def _classify_node_category(g: nx.DiGraph, node_id: str, attrs: Mapping[str, Any]) -> str:
    kind = _clean_text(attrs.get("kind"))
    if kind == "call_child_case":
        return "call_child_case"
    node_type = _clean_text(attrs.get("activity_type")).lower()
    if node_type:
        if "start" in node_type:
            return "start"
        if "end" in node_type:
            return "end"
        if "gateway" in node_type:
            return "gateway"
        if "usertask" in node_type or ("user" in node_type and "task" in node_type):
            return "user_task"
        if "servicetask" in node_type or ("service" in node_type and "task" in node_type):
            return "service_task"
        if any(token in node_type for token in ("timer", "escalation", "message", "signal", "event", "boundary")):
            return "event_other"
    if int(g.in_degree(node_id)) == 0:
        return "start"
    if int(g.out_degree(node_id)) == 0:
        return "end"
    return "other"


def _node_style(category: str) -> Tuple[str, str, int]:
    style_map = {
        "start": ("#A5D6A7", "circle", 1),
        "end": ("#FFCC80", "doublecircle", 2),
        "gateway": ("#90CAF9", "diamond", 1),
        "user_task": ("#B3E5FC", "box", 1),
        "service_task": ("#80CBC4", "box", 1),
        "event_other": ("#FFF59D", "ellipse", 1),
        "call_child_case": ("#CE93D8", "box", 1),
        "other": ("#CFD8DC", "box", 1),
    }
    return style_map.get(category, style_map["other"])


def _edge_style(edge_type: str) -> Tuple[str, str]:
    mapping = {
        "sequence": ("#546E7A", "solid"),
        "scope": ("#1E88E5", "dashed"),
        "parallel_branch": ("#1E88E5", "dashed"),
        "fork_edge": ("#FB8C00", "dashed"),
        "cancellation_edge": ("#D32F2F", "solid"),
        "call_activity_link": ("#8E24AA", "dotted"),
    }
    return mapping.get(edge_type, ("#757575", "solid"))


def _render_ig_graph_graphviz(graph: Mapping[str, Any], *, out_path: Optional[str], title: str) -> None:
    g = nx.DiGraph()
    for node in graph.get("nodes", []):
        node_id = _clean_text(node.get("id"))
        if not node_id:
            continue
        activity_def_id = _clean_text(node.get("activity_def_id")) or node_id
        activity_name = _clean_text(node.get("activity_name"))
        activity_type = _clean_text(node.get("activity_type"))
        label = _format_node_label(
            activity_def_id=activity_def_id,
            activity_name=activity_name,
            activity_type=activity_type,
            fallback_id=node_id,
        )
        g.add_node(node_id, label=label, activity_type=activity_type, kind="normal")

    for edge in graph.get("edges", []):
        source = _clean_text(edge.get("source"))
        target = _clean_text(edge.get("target"))
        if not source or not target:
            continue
        edge_type = _clean_text(edge.get("edge_type")) or "sequence"
        g.add_edge(source, target, edge_type=edge_type)

    for link in graph.get("metadata", {}).get("call_activity_links", []):
        parent_node = _clean_text(link.get("parent_activity_instance_id"))
        child_case = _clean_text(link.get("child_case_id"))
        if not parent_node or not child_case:
            continue
        child_node_id = f"CALL::{child_case}"
        if not g.has_node(child_node_id):
            g.add_node(
                child_node_id,
                label=f"CALL->{child_case}",
                activity_type="callActivity",
                kind="call_child_case",
            )
        g.add_edge(parent_node, child_node_id, edge_type="call_activity_link")

    if g.number_of_nodes() == 0:
        raise ValueError("Selected IG has no nodes to render.")

    dot = Digraph(comment=title)
    dot.attr(rankdir="LR", fontname="Helvetica")
    dot.attr("node", fontname="Helvetica", fontsize="10", style="filled", color="#263238")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    safe_id_map: Dict[str, str] = {}
    for idx, node_id in enumerate(g.nodes(), start=1):
        safe_id_map[str(node_id)] = f"n{idx}"

    for node_id, attrs in g.nodes(data=True):
        category = _classify_node_category(g, node_id, attrs)
        fill_color, shape, peripheries = _node_style(category)
        kwargs: Dict[str, str] = {
            "label": str(attrs.get("label", node_id)),
            "shape": shape,
            "fillcolor": fill_color,
        }
        if peripheries > 1:
            kwargs["peripheries"] = str(peripheries)
        dot.node(safe_id_map[str(node_id)], **kwargs)

    for source, target, edge_data in g.edges(data=True):
        edge_type = _clean_text(edge_data.get("edge_type")) or "sequence"
        color, style = _edge_style(edge_type)
        penwidth = "1.8" if edge_type != "sequence" else "1.3"
        edge_kwargs: Dict[str, str] = {
            "color": color,
            "style": style,
            "penwidth": penwidth,
        }
        if edge_type != "sequence":
            edge_kwargs["label"] = edge_type
        dot.edge(safe_id_map[str(source)], safe_id_map[str(target)], **edge_kwargs)

    try:
        if out_path:
            target = Path(out_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fmt = target.suffix.lstrip(".").lower() if target.suffix else "png"
            dot.format = fmt or "png"
            dot.render(filename=target.stem, directory=str(target.parent), cleanup=True)
            print(f"IG graph saved to: {target}")
            return
        dot.view(cleanup=True)
    except ExecutableNotFound as exc:
        raise RuntimeError(
            "Graphviz executable 'dot' was not found. Install Graphviz and add it to PATH "
            "to render IG visuals."
        ) from exc


def _render_ig_graph_matplotlib(graph: Mapping[str, Any], *, out_path: Optional[str], title: str) -> None:
    g = nx.DiGraph()
    for node in graph.get("nodes", []):
        node_id = _clean_text(node.get("id"))
        if not node_id:
            continue
        activity_def_id = _clean_text(node.get("activity_def_id")) or node_id
        activity_name = _clean_text(node.get("activity_name"))
        activity_type = _clean_text(node.get("activity_type"))
        label = _format_node_label(
            activity_def_id=activity_def_id,
            activity_name=activity_name,
            activity_type=activity_type,
            fallback_id=node_id,
        )
        g.add_node(node_id, label=label, activity_type=activity_type, kind="normal")

    for edge in graph.get("edges", []):
        source = _clean_text(edge.get("source"))
        target = _clean_text(edge.get("target"))
        if not source or not target:
            continue
        edge_type = _clean_text(edge.get("edge_type")) or "sequence"
        g.add_edge(source, target, edge_type=edge_type)

    for link in graph.get("metadata", {}).get("call_activity_links", []):
        parent_node = _clean_text(link.get("parent_activity_instance_id"))
        child_case = _clean_text(link.get("child_case_id"))
        if not parent_node or not child_case:
            continue
        child_node_id = f"CALL::{child_case}"
        if not g.has_node(child_node_id):
            g.add_node(child_node_id, label=f"CALL->{child_case}", activity_type="callActivity", kind="call_child_case")
        g.add_edge(parent_node, child_node_id, edge_type="call_activity_link")

    if g.number_of_nodes() == 0:
        raise ValueError("Selected IG has no nodes to render.")

    figure_size = max(10, min(24, int(0.35 * g.number_of_nodes()) + 8))
    plt.figure(figsize=(figure_size, figure_size))
    pos = nx.spring_layout(g, seed=42, k=max(0.2, 2.0 / max(2, g.number_of_nodes())))

    node_colors = []
    for node_id, attrs in g.nodes(data=True):
        category = _classify_node_category(g, node_id, attrs)
        color, _, _ = _node_style(category)
        node_colors.append(color)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=1100, alpha=0.95)

    edges_by_type: Dict[str, List[Tuple[str, str]]] = {}
    for source, target, edge_data in g.edges(data=True):
        edge_type = _clean_text(edge_data.get("edge_type")) or "sequence"
        edges_by_type.setdefault(edge_type, []).append((source, target))

    for edge_type, pairs in edges_by_type.items():
        color, style = _edge_style(edge_type)
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=pairs,
            edge_color=color,
            style=style,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            width=2.0 if edge_type != "sequence" else 1.5,
            connectionstyle="arc3,rad=0.05",
        )

    labels = {node_id: str(node_data.get("label", node_id)) for node_id, node_data in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if out_path:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=180)
        print(f"IG graph saved to: {output}")
    else:
        plt.show()
    plt.close()


def _render_ig_graph(graph: Mapping[str, Any], *, out_path: Optional[str], title: str) -> None:
    try:
        _render_ig_graph_graphviz(graph, out_path=out_path, title=title)
    except RuntimeError:
        _render_ig_graph_matplotlib(graph, out_path=out_path, title=title)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize one instance graph (IG) for parser sanity checks.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config", help="Path to experiment YAML config.")
    source_group.add_argument("--data", help="Path to source XES file (XES-only mode).")

    parser.add_argument("--case-id", default=None, help="Exact process instance id (case_id / PROC_INST_ID_).")
    parser.add_argument(
        "--pick",
        choices=["latest", "random", "longest", "shortest", "with-call-activity"],
        default="latest",
        help="Selection strategy when --case-id is not provided.",
    )
    parser.add_argument("--index", type=int, default=0, help="Index in ranked candidate list (default: 0).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --pick random.")
    parser.add_argument("--list-cases", action="store_true", help="Print ranked case list before rendering.")
    parser.add_argument("--top", type=int, default=30, help="How many cases to print with --list-cases.")
    parser.add_argument("--mode", choices=["activity-centric", "execution-centric"], default=None, help="Override canonical IG mode for Camunda.")
    parser.add_argument("--max-nodes", type=int, default=500, help="Maximum events/nodes to render for selected case.")
    parser.add_argument("--out", default=None, help="Optional output PNG path. If omitted, opens interactive window.")
    parser.add_argument("--title", default=None, help="Optional custom plot title.")

    args = parser.parse_args(argv)

    source_kind = "config"
    adapter_kind = "xes"
    camunda_payload: Optional[Dict[str, Any]] = None
    traces_payload: Optional[Dict[str, Any]] = None

    if args.config:
        cfg = load_yaml_config(args.config)
        mapping_cfg = cfg.get("mapping", {})
        if not isinstance(mapping_cfg, dict):
            mapping_cfg = {}
        if not mapping_cfg and isinstance(cfg.get("camunda"), dict):
            mapping_cfg = _build_camunda_mapping_from_legacy(cfg, cfg.get("data", {}))
        adapter_kind = _resolve_adapter_kind(mapping_cfg)
        if adapter_kind == "camunda":
            camunda_payload = _load_camunda_payload_from_config(args.config)
        else:
            traces_payload = _load_xes_traces_from_config(args.config)
    else:
        source_kind = "data"
        adapter_kind = "xes"
        traces_payload = _load_xes_traces_from_data(args.data)

    if adapter_kind == "camunda":
        assert camunda_payload is not None
        summaries = _summaries_from_camunda_events(camunda_payload["events"])
    else:
        assert traces_payload is not None
        summaries = _summaries_from_traces(traces_payload["traces"])

    if args.list_cases:
        _print_case_table(summaries, pick=args.pick, seed=args.seed, top=max(1, int(args.top)))
        if args.case_id is None and source_kind == "config" and args.out is None:
            return 0

    selected_case_id = _select_case_id(
        summaries,
        case_id=args.case_id,
        pick=args.pick,
        index=int(args.index),
        seed=int(args.seed),
    )
    if args.case_id is None:
        print(
            "Selected case_id by strategy: "
            f"{selected_case_id} (pick={args.pick}, index={int(args.index)})"
        )
    else:
        print(f"Selected case_id: {selected_case_id}")

    if adapter_kind == "camunda":
        graph = _build_camunda_case_graph(
            camunda_payload,
            case_id=selected_case_id,
            mode=args.mode,
            max_nodes=int(args.max_nodes),
        )
        process_name = str(camunda_payload["process_name"])
        default_title = f"IG | process={process_name} | case={selected_case_id}"
    else:
        graph = _build_xes_case_graph(
            traces_payload["traces"],
            case_id=selected_case_id,
            max_nodes=int(args.max_nodes),
        )
        dataset_name = str(traces_payload["dataset_name"])
        default_title = f"IG (XES sequence) | dataset={dataset_name} | case={selected_case_id}"

    title = str(args.title).strip() if args.title else default_title
    _render_ig_graph(graph, out_path=args.out, title=title)

    metadata = graph.get("metadata", {})
    print(
        "Rendered IG summary: "
        f"nodes={metadata.get('num_nodes', len(graph.get('nodes', [])))}, "
        f"edges={metadata.get('num_edges', len(graph.get('edges', [])))}, "
        f"mode={metadata.get('build_mode', metadata.get('mode', 'unknown'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
