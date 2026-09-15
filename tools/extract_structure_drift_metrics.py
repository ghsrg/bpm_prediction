from __future__ import annotations

"""Extract reproducible structural-drift evidence from process_structure artifacts."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


EXPECTED_VERSIONS = ("v1", "v2", "v3", "v4", "v5")
TASK_TAGS = {
    "task",
    "usertask",
    "servicetask",
    "manualtask",
    "scripttask",
    "businessruletask",
    "sendtask",
    "receivetask",
}
GATEWAY_TAGS = {
    "exclusivegateway": "exclusive",
    "parallelgateway": "parallel",
    "inclusivegateway": "inclusive",
    "eventbasedgateway": "event_based",
}


class StructureExtractionError(ValueError):
    """Raised when a structure artifact cannot provide unambiguous evidence."""


@dataclass(frozen=True)
class StructureSource:
    dataset: str
    stratum: str
    process_id: str
    process_fingerprint: str
    version: str
    path: Path


@dataclass
class StructureSnapshot:
    source: StructureSource
    source_sha256: str
    schema_version: str
    repository_backend: str
    topology_layout: str
    raw_node_count: int | None
    raw_sequence_flow_count: int | None
    projected_node_metadata_count: int | None
    projected_allowed_edge_count: int
    source_topology_signal: bool
    task_keys: tuple[str, ...]
    canonical_relations: set[tuple[str, str]]
    gateway_counts: dict[str, int]
    start_event_count: int
    end_event_count: int
    cycle_present: bool
    max_task_out_degree: int
    mean_task_out_degree: float
    branching_task_count: int
    merge_task_count: int
    warnings: list[tuple[str, str]] = field(default_factory=list)


def _tag(node: dict[str, Any]) -> str:
    for key in ("bpmn_tag", "type", "activity_type"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _is_task(node: dict[str, Any]) -> bool:
    return _tag(node) in TASK_TAGS


def _is_event(tag: str) -> bool:
    return tag.endswith("event") or tag == "event"


def _process_fingerprint(process_id: str) -> str:
    match = re.search(r"_([0-9a-fA-F]{8})$", process_id)
    return match.group(1).lower() if match else ""


def _stratum(process_id: str, dataset: str) -> str:
    if dataset == "loan":
        return "loan"
    if "-simple" in process_id:
        return "simple"
    if "-medium" in process_id or "-middle" in process_id:
        return "middle"
    if "-complex" in process_id:
        return "complex"
    raise StructureExtractionError(f"cannot infer CDLG stratum from process directory {process_id!r}")


def _dataset_matches(directory_name: str, dataset: str) -> bool:
    return directory_name.startswith("loan_") if dataset == "loan" else directory_name.startswith("cdlg-")


def discover_sources(structure_root: Path, dataset: str) -> list[StructureSource]:
    if not structure_root.is_dir():
        raise StructureExtractionError(f"structure root does not exist: {structure_root}")

    sources: list[StructureSource] = []
    processes = [path for path in sorted(structure_root.iterdir()) if path.is_dir() and _dataset_matches(path.name, dataset)]
    if not processes:
        raise StructureExtractionError(f"no {dataset} process directories found under {structure_root}")
    if dataset == "loan" and len(processes) != 1:
        names = ", ".join(path.name for path in processes)
        raise StructureExtractionError(f"expected exactly one loan process directory, found: {names}")

    for process_dir in processes:
        version_files = {
            child.name: child / "process_structure.json"
            for child in process_dir.iterdir()
            if child.is_dir() and (child / "process_structure.json").is_file()
        }
        actual_versions = set(version_files)
        expected_versions = set(EXPECTED_VERSIONS)
        if actual_versions != expected_versions:
            missing = sorted(expected_versions - actual_versions)
            unexpected = sorted(actual_versions - expected_versions)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise StructureExtractionError(
                f"process {process_dir.name!r} must contain exactly v1..v5 process_structure.json ({', '.join(details)})"
            )
        for version in EXPECTED_VERSIONS:
            sources.append(
                StructureSource(
                    dataset=dataset,
                    stratum=_stratum(process_dir.name, dataset),
                    process_id=process_dir.name,
                    process_fingerprint=_process_fingerprint(process_dir.name),
                    version=version,
                    path=version_files[version],
                )
            )
    return sources


def _require_list(value: Any, field_name: str, source: StructureSource) -> list[Any]:
    if not isinstance(value, list):
        raise StructureExtractionError(f"{source.path}: {field_name} must be a list")
    return value


def _normalize_allowed_edges(value: Any, field_name: str, source: StructureSource) -> set[tuple[str, str]]:
    raw_edges = _require_list(value, field_name, source)
    edges: set[tuple[str, str]] = set()
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise StructureExtractionError(f"{source.path}: {field_name} entries must be two-item edge pairs")
        source_id, target_id = (str(item).strip() for item in edge)
        if not source_id or not target_id:
            raise StructureExtractionError(f"{source.path}: {field_name} contains an empty edge endpoint")
        edges.add((source_id, target_id))
    return edges


def _has_directed_cycle(nodes: Iterable[str], relations: set[tuple[str, str]]) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for source, target in relations:
        adjacency[source].append(target)
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for target in adjacency.get(node, []):
            if state.get(target) == 1:
                return True
            if state.get(target, 0) == 0 and visit(target):
                return True
        state[node] = 2
        return False

    return any(state.get(node, 0) == 0 and visit(node) for node in nodes)


def load_snapshot(source: StructureSource) -> StructureSnapshot:
    try:
        raw_bytes = source.path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructureExtractionError(f"cannot read {source.path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StructureExtractionError(f"{source.path}: root payload must be an object")
    dto = payload.get("dto")
    if not isinstance(dto, dict):
        raise StructureExtractionError(f"{source.path}: dto must be an object")

    raw_nodes = _require_list(dto.get("nodes", []), "dto.nodes", source)
    raw_edges = _require_list(dto.get("edges", []), "dto.edges", source)
    graph_topology = dto.get("graph_topology", {})
    if graph_topology is None:
        graph_topology = {}
    if not isinstance(graph_topology, dict):
        raise StructureExtractionError(f"{source.path}: dto.graph_topology must be an object when present")
    topology_allowed_raw = graph_topology.get("allowed_edges")
    dto_allowed_raw = dto.get("allowed_edges")
    node_metadata = dto.get("node_metadata", {})
    if node_metadata is None:
        node_metadata = {}
    if not isinstance(node_metadata, dict):
        raise StructureExtractionError(f"{source.path}: dto.node_metadata must be an object when present")
    source_topology_signal = bool(raw_nodes or raw_edges or topology_allowed_raw or dto_allowed_raw or node_metadata)

    if raw_edges and not raw_nodes:
        raise StructureExtractionError(f"{source.path}: explicit dto.edges require dto.nodes")

    if not raw_nodes and not raw_edges:
        allowed_raw = topology_allowed_raw if topology_allowed_raw else dto_allowed_raw
        if allowed_raw is None or not node_metadata:
            raise StructureExtractionError(
                f"{source.path}: no usable topology representation; expected dto.nodes + dto.edges or "
                "dto.node_metadata + allowed_edges"
            )
        canonical_relations = _normalize_allowed_edges(allowed_raw, "dto.graph_topology.allowed_edges", source)
        endpoint_ids = {node_id for edge in canonical_relations for node_id in edge}
        missing_metadata = sorted(node_id for node_id in endpoint_ids if not isinstance(node_metadata.get(node_id), dict))
        if missing_metadata:
            raise StructureExtractionError(
                f"{source.path}: node metadata is missing for projected topology endpoint(s): {missing_metadata}"
            )
        task_keys = tuple(sorted(endpoint_ids))
        out_degree = {name: 0 for name in task_keys}
        in_degree = {name: 0 for name in task_keys}
        for source_name, target_name in canonical_relations:
            out_degree[source_name] += 1
            in_degree[target_name] += 1
        return StructureSnapshot(
            source=source,
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            schema_version=str(payload.get("schema_version", "")),
            repository_backend=str(payload.get("repository_backend", "")),
            topology_layout="projected_allowed_edges",
            raw_node_count=None,
            raw_sequence_flow_count=None,
            projected_node_metadata_count=len(node_metadata),
            projected_allowed_edge_count=len(canonical_relations),
            source_topology_signal=source_topology_signal,
            task_keys=task_keys,
            canonical_relations=canonical_relations,
            gateway_counts={key: 0 for key in GATEWAY_TAGS.values()},
            start_event_count=0,
            end_event_count=0,
            cycle_present=_has_directed_cycle(task_keys, canonical_relations),
            max_task_out_degree=max(out_degree.values(), default=0),
            mean_task_out_degree=(sum(out_degree.values()) / len(out_degree)) if out_degree else 0.0,
            branching_task_count=sum(value > 1 for value in out_degree.values()),
            merge_task_count=sum(value > 1 for value in in_degree.values()),
        )

    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict) or not isinstance(raw_node.get("id"), str) or not raw_node["id"].strip():
            raise StructureExtractionError(f"{source.path}: every dto.nodes entry needs a non-empty id")
        node_id = raw_node["id"]
        if node_id in nodes:
            raise StructureExtractionError(f"{source.path}: duplicate node id {node_id!r}")
        nodes[node_id] = raw_node

    task_names: dict[str, str] = {}
    labels: dict[str, list[str]] = defaultdict(list)
    warnings: list[tuple[str, str]] = []
    gateway_counts = {key: 0 for key in GATEWAY_TAGS.values()}
    start_event_count = 0
    end_event_count = 0
    for node_id, node in nodes.items():
        tag = _tag(node)
        if _is_task(node):
            name = node.get("name")
            if not isinstance(name, str) or not name.strip():
                raise StructureExtractionError(f"{source.path}: task {node_id!r} has a missing task label")
            task_names[node_id] = name.strip()
            labels[name.strip()].append(node_id)
        elif tag in GATEWAY_TAGS:
            gateway_counts[GATEWAY_TAGS[tag]] += 1
        elif tag == "startevent":
            start_event_count += 1
        elif tag == "endevent":
            end_event_count += 1
        elif _is_event(tag):
            continue
        else:
            warnings.append(("unsupported_node_type", f"node {node_id!r} has unsupported type {tag or '<missing>'!r}"))
    duplicates = {label: ids for label, ids in labels.items() if len(ids) > 1}
    if duplicates:
        details = ", ".join(f"{label!r}: {sorted(ids)}" for label, ids in sorted(duplicates.items()))
        raise StructureExtractionError(f"{source.path}: duplicate task label(s) would make relations ambiguous: {details}")

    adjacency: dict[str, set[str]] = defaultdict(set)
    raw_sequence_flow_count = 0
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            raise StructureExtractionError(f"{source.path}: dto.edges entries must be objects")
        if str(raw_edge.get("edge_type", "sequence")).lower() != "sequence":
            continue
        source_id = raw_edge.get("source")
        target_id = raw_edge.get("target")
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            raise StructureExtractionError(f"{source.path}: sequence edge requires source and target")
        if source_id not in nodes or target_id not in nodes:
            raise StructureExtractionError(f"{source.path}: sequence edge references an unknown node")
        raw_sequence_flow_count += 1
        adjacency[source_id].add(target_id)

    canonical_relations: set[tuple[str, str]] = set()
    task_ids = set(task_names)
    for source_id, source_name in task_names.items():
        pending = list(sorted(adjacency.get(source_id, ()), reverse=True))
        visited: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id in task_ids:
                canonical_relations.add((source_name, task_names[node_id]))
                continue
            pending.extend(sorted(adjacency.get(node_id, ()), reverse=True))

    out_degree = {name: 0 for name in task_names.values()}
    in_degree = {name: 0 for name in task_names.values()}
    for source_name, target_name in canonical_relations:
        out_degree[source_name] += 1
        in_degree[target_name] += 1
    task_keys = tuple(sorted(task_names.values()))
    return StructureSnapshot(
        source=source,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema_version=str(payload.get("schema_version", "")),
        repository_backend=str(payload.get("repository_backend", "")),
        topology_layout="explicit_graph",
        raw_node_count=len(nodes),
        raw_sequence_flow_count=raw_sequence_flow_count,
        projected_node_metadata_count=None,
        projected_allowed_edge_count=len(canonical_relations),
        source_topology_signal=source_topology_signal,
        task_keys=task_keys,
        canonical_relations=canonical_relations,
        gateway_counts=gateway_counts,
        start_event_count=start_event_count,
        end_event_count=end_event_count,
        cycle_present=_has_directed_cycle(task_keys, canonical_relations),
        max_task_out_degree=max(out_degree.values(), default=0),
        mean_task_out_degree=(sum(out_degree.values()) / len(out_degree)) if out_degree else 0.0,
        branching_task_count=sum(value > 1 for value in out_degree.values()),
        merge_task_count=sum(value > 1 for value in in_degree.values()),
        warnings=warnings,
    )


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(values), ensure_ascii=True)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def version_row(snapshot: StructureSnapshot) -> dict[str, Any]:
    task_count = len(snapshot.task_keys)
    return {
        "dataset": snapshot.source.dataset,
        "stratum": snapshot.source.stratum,
        "process_id": snapshot.source.process_id,
        "process_fingerprint": snapshot.source.process_fingerprint,
        "version": snapshot.source.version,
        "structure_source_path": str(snapshot.source.path.resolve()),
        "structure_source_sha256": snapshot.source_sha256,
        "schema_version": snapshot.schema_version,
        "repository_backend": snapshot.repository_backend,
        "topology_layout": snapshot.topology_layout,
        "raw_node_count": snapshot.raw_node_count,
        "raw_sequence_flow_count": snapshot.raw_sequence_flow_count,
        "projected_node_metadata_count": snapshot.projected_node_metadata_count,
        "projected_allowed_edge_count": snapshot.projected_allowed_edge_count,
        "task_count": task_count,
        "task_keys_json": _json_list(snapshot.task_keys),
        "canonical_task_relation_count": len(snapshot.canonical_relations),
        "exclusive_gateway_count": snapshot.gateway_counts["exclusive"],
        "parallel_gateway_count": snapshot.gateway_counts["parallel"],
        "inclusive_gateway_count": snapshot.gateway_counts["inclusive"],
        "event_based_gateway_count": snapshot.gateway_counts["event_based"],
        "total_gateway_count": sum(snapshot.gateway_counts.values()),
        "start_event_count": snapshot.start_event_count,
        "end_event_count": snapshot.end_event_count,
        "canonical_task_relation_cycle_present": str(snapshot.cycle_present).lower(),
        "max_task_out_degree": snapshot.max_task_out_degree,
        "mean_task_out_degree": snapshot.mean_task_out_degree,
        "branching_task_count": snapshot.branching_task_count,
        "branching_task_share": _ratio(snapshot.branching_task_count, task_count),
        "merge_task_count": snapshot.merge_task_count,
        "merge_task_share": _ratio(snapshot.merge_task_count, task_count),
    }


def transition_row(source: StructureSnapshot, target: StructureSnapshot) -> dict[str, Any]:
    source_tasks, target_tasks = set(source.task_keys), set(target.task_keys)
    added_tasks, removed_tasks = target_tasks - source_tasks, source_tasks - target_tasks
    added_relations = target.canonical_relations - source.canonical_relations
    removed_relations = source.canonical_relations - target.canonical_relations
    activity_jaccard = _jaccard(source_tasks, target_tasks)
    relation_jaccard = _jaccard(source.canonical_relations, target.canonical_relations)
    activity_set_delta = 1.0 - activity_jaccard
    relation_set_delta = 1.0 - relation_jaccard
    if not 0.0 <= activity_set_delta <= 1.0 or not 0.0 <= relation_set_delta <= 1.0:
        raise StructureExtractionError(
            f"invalid bounded topology deltas for {source.source.process_id} "
            f"{source.source.version}->{target.source.version}"
        )
    gateway_deltas = {
        key: target.gateway_counts[key] - source.gateway_counts[key] for key in GATEWAY_TAGS.values()
    }
    observable_changes: list[str] = []
    if added_tasks or removed_tasks:
        observable_changes.append("task_set_changed")
    if added_relations or removed_relations:
        observable_changes.append("canonical_task_relations_changed")
    if any(gateway_deltas.values()):
        observable_changes.append("gateway_composition_changed")
    if source.branching_task_count != target.branching_task_count or source.merge_task_count != target.merge_task_count:
        observable_changes.append("branching_or_merge_changed")
    if source.cycle_present != target.cycle_present:
        observable_changes.append("canonical_cycle_presence_changed")
    unchanged_transition = source_tasks == target_tasks and source.canonical_relations == target.canonical_relations
    return {
        "dataset": source.source.dataset,
        "stratum": source.source.stratum,
        "process_id": source.source.process_id,
        "process_fingerprint": source.source.process_fingerprint,
        "source_version": source.source.version,
        "target_version": target.source.version,
        "source_task_count": len(source_tasks),
        "target_task_count": len(target_tasks),
        "added_task_count": len(added_tasks),
        "removed_task_count": len(removed_tasks),
        "added_task_keys_json": _json_list(added_tasks),
        "removed_task_keys_json": _json_list(removed_tasks),
        "activity_jaccard": activity_jaccard,
        "activity_set_delta": activity_set_delta,
        "net_task_count_delta": len(target_tasks) - len(source_tasks),
        "source_canonical_relation_count": len(source.canonical_relations),
        "target_canonical_relation_count": len(target.canonical_relations),
        "added_relation_count": len(added_relations),
        "removed_relation_count": len(removed_relations),
        "relation_jaccard": relation_jaccard,
        "relation_set_delta": relation_set_delta,
        "net_relation_count_delta": len(target.canonical_relations) - len(source.canonical_relations),
        "relation_edit_count": len(added_relations) + len(removed_relations),
        "normalized_relation_edit_rate": _ratio(len(added_relations) + len(removed_relations), max(len(source.canonical_relations), len(target.canonical_relations))),
        "raw_sequence_flow_count_delta": (
            target.raw_sequence_flow_count - source.raw_sequence_flow_count
            if source.raw_sequence_flow_count is not None and target.raw_sequence_flow_count is not None
            else None
        ),
        "projected_allowed_edge_count_delta": target.projected_allowed_edge_count - source.projected_allowed_edge_count,
        "exclusive_gateway_count_delta": gateway_deltas["exclusive"],
        "parallel_gateway_count_delta": gateway_deltas["parallel"],
        "inclusive_gateway_count_delta": gateway_deltas["inclusive"],
        "event_based_gateway_count_delta": gateway_deltas["event_based"],
        "total_gateway_count_delta": sum(gateway_deltas.values()),
        "source_branching_task_count": source.branching_task_count,
        "target_branching_task_count": target.branching_task_count,
        "source_merge_task_count": source.merge_task_count,
        "target_merge_task_count": target.merge_task_count,
        "source_cycle_present": str(source.cycle_present).lower(),
        "target_cycle_present": str(target.cycle_present).lower(),
        "cycle_presence_changed": str(source.cycle_present != target.cycle_present).lower(),
        "unchanged_transition": str(unchanged_transition).lower(),
        "change_type": "unchanged" if unchanged_transition else ";".join(observable_changes),
    }


def relation_delta_rows(source: StructureSnapshot, target: StructureSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, relations in (("added", target.canonical_relations - source.canonical_relations), ("removed", source.canonical_relations - target.canonical_relations)):
        for source_task, target_task in sorted(relations):
            rows.append(
                {
                    "dataset": source.source.dataset,
                    "stratum": source.source.stratum,
                    "process_id": source.source.process_id,
                    "process_fingerprint": source.source.process_fingerprint,
                    "source_version": source.source.version,
                    "target_version": target.source.version,
                    "change_direction": direction,
                    "source_task": source_task,
                    "target_task": target_task,
                }
            )
    return rows


def source_inventory_row(snapshot: StructureSnapshot) -> dict[str, Any]:
    return {
        "dataset": snapshot.source.dataset,
        "stratum": snapshot.source.stratum,
        "process_id": snapshot.source.process_id,
        "process_fingerprint": snapshot.source.process_fingerprint,
        "version": snapshot.source.version,
        "structure_source_path": str(snapshot.source.path.resolve()),
        "structure_source_sha256": snapshot.source_sha256,
        "schema_version": snapshot.schema_version,
        "repository_backend": snapshot.repository_backend,
        "topology_layout": snapshot.topology_layout,
        "source_status": "loaded",
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def stratum_aggregate_rows(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "activity_set_delta",
        "relation_set_delta",
        "net_task_count_delta",
        "net_relation_count_delta",
        "added_task_count",
        "removed_task_count",
        "relation_edit_count",
        "normalized_relation_edit_rate",
        "projected_allowed_edge_count_delta",
        "total_gateway_count_delta",
        "cycle_presence_changed",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transitions:
        grouped[str(row["stratum"])].append(row)
    rows: list[dict[str, Any]] = []
    for stratum, group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "dataset": "cdlg",
            "stratum": stratum,
            "process_count": len({str(item["process_id"]) for item in group}),
            "transition_count": len(group),
        }
        for metric in metric_names:
            values = [float(item[metric]) if metric != "cycle_presence_changed" else float(item[metric] == "true") for item in group]
            mean, std = _mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise StructureExtractionError(f"refusing to write empty required output: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_dictionary(path: Path) -> None:
    path.write_text(
        """# Structure Drift Metric Dictionary

This export measures the topology representation supplied to the prediction model through prebuilt `process_structure.json` artifacts. It does not parse BPMN XML, use BPMN IDs as a metric, or claim that the scenarios represent live business-process change. The evidence supports the narrower statement: **structurally diverse and controlled drift scenarios**.

## Source And Identity

- `structure_source_path`: exact artifact consumed by the export.
- `structure_source_sha256`: SHA-256 of that artifact's bytes. A CDLG folder suffix is a process fingerprint only; it is not a SHA-256 digest.
- `process_fingerprint`: optional eight-hex folder suffix for CDLG process identification.
- `stratum`: `simple`, `middle`, or `complex`; local `cdlg-medium*` directories are reported as `middle`.

## Version Metrics

- `topology_layout`: `explicit_graph` for `dto.nodes` plus `dto.edges`, or `projected_allowed_edges` for `dto.node_metadata` plus allowed task-to-task edges.
- `raw_sequence_flow_count`: count of raw `dto.edges` entries whose `edge_type` is `sequence`; blank when that raw representation is unavailable.
- `projected_allowed_edge_count`: count of directed task-to-task topology edges consumed by prediction. For projected artifacts, these are the supplied allowed edges.
- `projected_node_metadata_count`: metadata-node count for projected artifacts; blank for explicit graphs.
- `canonical_task_relation_count`: count of task-to-task relations after traversing gateway/event nodes in explicit graphs, or the supplied projected allowed-edge count in projected artifacts.
- `*_gateway_count`: raw gateway counts by BPMN tag.
- `canonical_task_relation_cycle_present`: cycle detection on the canonical task relation graph.
- `branching_task_*` and `merge_task_*`: task nodes with canonical out-degree or in-degree greater than one.

## Transition Metrics

Each row compares adjacent versions only: v1->v2, v2->v3, v3->v4, v4->v5.

- `activity_jaccard`: intersection over union of the candidate activity universe. Projected artifacts use candidate node identities from allowed-edge endpoints.
- `relation_jaccard`: intersection over union of directed allowed task-to-task topology.
- `activity_set_delta`: `1 - activity_jaccard`, bounded in `[0, 1]`.
- `relation_set_delta`: `1 - relation_jaccard`, bounded in `[0, 1]`.
- `net_task_count_delta` and `net_relation_count_delta`: signed count changes. They are descriptive only, not structural distances.
- `relation_edit_count`: added plus removed canonical relations.
- `normalized_relation_edit_rate`: relation edits divided by the larger source/target relation count, or zero when both are empty.
- `change_type`: observable categories, not inferred causal labels.

## Reproduction

```powershell
.\\.venv-modern\\Scripts\\python.exe tools\\extract_structure_drift_metrics.py --dataset loan
.\\.venv-modern\\Scripts\\python.exe tools\\extract_structure_drift_metrics.py --dataset cdlg
```
""",
        encoding="utf-8",
    )


def default_output_dir(dataset: str) -> Path:
    base = Path("outputs") / "Export_metrics" / "article_aggregates"
    return base / ("loan" if dataset == "loan" else "CDLG") / "structure_metrics"


def run_export(structure_root: Path, dataset: str, output_dir: Path) -> tuple[int, int, int]:
    sources = discover_sources(structure_root, dataset)
    snapshots = [load_snapshot(source) for source in sources]
    by_process: dict[str, dict[str, StructureSnapshot]] = defaultdict(dict)
    for snapshot in snapshots:
        by_process[snapshot.source.process_id][snapshot.source.version] = snapshot

    transitions: list[dict[str, Any]] = []
    relation_deltas: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for process_id, versions in sorted(by_process.items()):
        process_snapshots = [versions[version] for version in EXPECTED_VERSIONS]
        if all(not snapshot.task_keys for snapshot in process_snapshots) and any(
            snapshot.source_topology_signal for snapshot in process_snapshots
        ):
            raise StructureExtractionError(
                f"process {process_id!r} has zero task candidates across v1..v5 despite non-empty topology source data"
            )
        for snapshot in versions.values():
            validation_rows.append(
                {
                    "severity": "info",
                    "code": "structure_loaded",
                    "process_id": process_id,
                    "version": snapshot.source.version,
                    "message": "loaded prebuilt process_structure artifact",
                }
            )
            for code, message in snapshot.warnings:
                validation_rows.append(
                    {"severity": "warning", "code": code, "process_id": process_id, "version": snapshot.source.version, "message": message}
                )
        for source_version, target_version in zip(EXPECTED_VERSIONS, EXPECTED_VERSIONS[1:]):
            source, target = versions[source_version], versions[target_version]
            row = transition_row(source, target)
            transitions.append(row)
            relation_deltas.extend(relation_delta_rows(source, target))
            if row["unchanged_transition"] == "true":
                validation_rows.append(
                    {
                        "severity": "info",
                        "code": "unchanged_adjacent_transition",
                        "process_id": process_id,
                        "version": f"{source_version}->{target_version}",
                        "message": "no observable structural delta",
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "structure_version_metrics.csv", [version_row(snapshot) for snapshot in snapshots])
    _write_csv(output_dir / "structure_transition_metrics.csv", transitions)
    relation_fields = [
        "dataset", "stratum", "process_id", "process_fingerprint", "source_version", "target_version", "change_direction", "source_task", "target_task"
    ]
    with (output_dir / "structure_relation_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=relation_fields)
        writer.writeheader()
        writer.writerows(relation_deltas)
    _write_csv(output_dir / "structure_source_inventory.csv", [source_inventory_row(snapshot) for snapshot in snapshots])
    _write_csv(output_dir / "structure_validation_report.csv", validation_rows)
    _write_dictionary(output_dir / "structure_metric_dictionary.md")
    if dataset == "cdlg":
        _write_csv(output_dir / "structure_stratum_aggregates.csv", stratum_aggregate_rows(transitions))
    warnings = sum(row["severity"] == "warning" for row in validation_rows)
    return len(by_process), len(transitions), warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("loan", "cdlg"))
    parser.add_argument("--structure-root", type=Path, default=Path("data") / "knowledge_graph")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir(args.dataset)
    try:
        process_count, transition_count, warning_count = run_export(args.structure_root, args.dataset, output_dir)
    except StructureExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Structure drift export complete: dataset={args.dataset} processes={process_count} "
        f"transitions={transition_count} warnings={warning_count} output_dir={output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
