"""Offline diagnostics for topology drift and fixed-head prediction failures."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set, Tuple
import xml.etree.ElementTree as ET


Activity = str
Version = str
Transition = Tuple[Activity, Activity]

_BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"
_GATEWAY_TAGS = {"exclusiveGateway", "parallelGateway", "inclusiveGateway", "eventBasedGateway"}
_ACTIVITY_TAGS = {
    "task",
    "userTask",
    "serviceTask",
    "scriptTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
    "manualTask",
    "startEvent",
    "endEvent",
}
_NODE_TAGS = _ACTIVITY_TAGS | _GATEWAY_TAGS


@dataclass(frozen=True)
class VersionTopologySnapshot:
    version: Version
    activities: Set[Activity]
    transitions: Set[Transition]


@dataclass(frozen=True)
class VersionTopologyDiff:
    source_version: Version
    target_version: Version
    added_activities: Set[Activity]
    removed_activities: Set[Activity]
    common_activities: Set[Activity]
    added_transitions: Set[Transition]
    removed_transitions: Set[Transition]
    changed_successors: Dict[Activity, Dict[str, Set[Activity]]]


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    model_label: str
    trace_idx: int
    step: int
    process_version: Version
    prefix_last_activity: Activity
    target_label: Activity
    pred_label: Activity
    strict_correct: bool
    pred_in_mask: bool
    target_in_mask: bool
    strict_error_but_allowed: bool
    mask_cardinality: int


@dataclass(frozen=True)
class PredictionAttribution:
    record: PredictionRecord
    target_seen_in_train: bool
    target_count_in_train: int
    target_is_new_in_eval_version: bool
    pred_removed_in_eval_version: bool
    prefix_transition_changed: bool
    old_allowed_targets: Set[Activity]
    new_allowed_targets: Set[Activity]
    error_bucket: str


def parse_bpmn_topology_snapshot(
    path: str | Path,
    *,
    version: Version,
    label_attr: str = "name",
) -> VersionTopologySnapshot:
    """Parse BPMN XML into the gateway-collapsed prediction topology."""
    if label_attr not in {"id", "name"}:
        raise ValueError("label_attr must be either 'id' or 'name'.")
    root = ET.parse(path).getroot()
    nodes: Dict[str, Dict[str, str]] = {}
    outgoing: Dict[str, list[str]] = defaultdict(list)
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in _NODE_TAGS:
            node_id = str(element.attrib.get("id", "")).strip()
            if node_id:
                nodes[node_id] = {
                    "id": node_id,
                    "name": str(element.attrib.get("name", "")).strip() or node_id,
                    "type": tag,
                }
    for element in root.iter(_BPMN_NS + "sequenceFlow"):
        source = str(element.attrib.get("sourceRef", "")).strip()
        target = str(element.attrib.get("targetRef", "")).strip()
        if source and target:
            outgoing[source].append(target)

    activities = {_node_label(node, label_attr=label_attr) for node in nodes.values() if node["type"] in _ACTIVITY_TAGS}
    transitions: Set[Transition] = set()
    for source_id, node in nodes.items():
        if node["type"] not in _ACTIVITY_TAGS:
            continue
        for target_id in _reachable_activity_targets(source_id, nodes=nodes, outgoing=outgoing):
            transitions.add(
                (
                    _node_label(node, label_attr=label_attr),
                    _node_label(nodes[target_id], label_attr=label_attr),
                )
            )
    return VersionTopologySnapshot(version=str(version), activities=activities, transitions=transitions)


def parse_xes_prefix_last_activity_lookup(
    path: str | Path,
    *,
    activity_key: str = "concept:name",
) -> Dict[tuple[int, int], Activity]:
    """Map (trace_idx, prefix_len) to the last activity visible in that prefix."""
    root = ET.parse(path).getroot()
    lookup: Dict[tuple[int, int], Activity] = {}
    trace_idx = 0
    for trace in root.iter():
        if _local_name(trace.tag) != "trace":
            continue
        events: list[str] = []
        for child in list(trace):
            if _local_name(child.tag) != "event":
                continue
            activity = _event_string_value(child, activity_key)
            if activity:
                events.append(activity)
        for prefix_len, activity in enumerate(events, start=1):
            lookup[(int(trace_idx), int(prefix_len))] = str(activity)
        trace_idx += 1
    return lookup


def diff_topology_versions(
    source: VersionTopologySnapshot,
    target: VersionTopologySnapshot,
) -> VersionTopologyDiff:
    """Compare two prediction-level topology snapshots."""
    added_activities = set(target.activities) - set(source.activities)
    removed_activities = set(source.activities) - set(target.activities)
    common_activities = set(source.activities) & set(target.activities)
    added_transitions = set(target.transitions) - set(source.transitions)
    removed_transitions = set(source.transitions) - set(target.transitions)
    changed_successors = _changed_successors(source.transitions, target.transitions)
    return VersionTopologyDiff(
        source_version=source.version,
        target_version=target.version,
        added_activities=added_activities,
        removed_activities=removed_activities,
        common_activities=common_activities,
        added_transitions=added_transitions,
        removed_transitions=removed_transitions,
        changed_successors=changed_successors,
    )


def attribute_prediction_error(
    record: PredictionRecord,
    *,
    train_activity_counts: Mapping[Activity, int],
    topology_diff: VersionTopologyDiff,
) -> PredictionAttribution:
    """Assign one diagnostic bucket to a prediction record."""
    target_count = int(train_activity_counts.get(record.target_label, 0))
    target_seen = target_count > 0
    target_new = record.target_label in topology_diff.added_activities
    pred_removed = record.pred_label in topology_diff.removed_activities
    successor_change = topology_diff.changed_successors.get(record.prefix_last_activity, {})
    old_allowed = set(successor_change.get("source_successors", set()))
    new_allowed = set(successor_change.get("target_successors", set()))
    transition_changed = bool(old_allowed or new_allowed)

    if record.strict_correct:
        bucket = "correct"
    elif not target_seen:
        bucket = "unseen_target_class"
    elif pred_removed:
        bucket = "removed_prediction_class"
    elif not record.pred_in_mask:
        bucket = "oos_prediction"
    elif not record.target_in_mask:
        bucket = "target_not_in_mask"
    elif transition_changed:
        bucket = "changed_transition_zone"
    elif record.strict_error_but_allowed:
        bucket = "strict_error_but_allowed"
    else:
        bucket = "unknown"

    return PredictionAttribution(
        record=record,
        target_seen_in_train=target_seen,
        target_count_in_train=target_count,
        target_is_new_in_eval_version=target_new,
        pred_removed_in_eval_version=pred_removed,
        prefix_transition_changed=transition_changed,
        old_allowed_targets=old_allowed,
        new_allowed_targets=new_allowed,
        error_bucket=bucket,
    )


def _successors(transitions: Iterable[Transition]) -> Dict[Activity, Set[Activity]]:
    result: Dict[Activity, Set[Activity]] = {}
    for src, dst in transitions:
        result.setdefault(str(src), set()).add(str(dst))
    return result


def _changed_successors(
    source_transitions: Iterable[Transition],
    target_transitions: Iterable[Transition],
) -> Dict[Activity, Dict[str, Set[Activity]]]:
    source_successors = _successors(source_transitions)
    target_successors = _successors(target_transitions)
    changed: Dict[Activity, Dict[str, Set[Activity]]] = {}
    for activity in sorted(set(source_successors) | set(target_successors)):
        source_set = set(source_successors.get(activity, set()))
        target_set = set(target_successors.get(activity, set()))
        if source_set == target_set:
            continue
        changed[activity] = {
            "source_successors": source_set,
            "target_successors": target_set,
            "removed_successors": source_set - target_set,
            "added_successors": target_set - source_set,
        }
    return changed


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _node_label(node: Mapping[str, str], *, label_attr: str) -> str:
    if label_attr == "id":
        return str(node.get("id", "")).strip()
    return str(node.get("name", "") or node.get("id", "")).strip()


def _event_string_value(event: ET.Element, key: str) -> str | None:
    for child in list(event):
        if _local_name(child.tag) != "string":
            continue
        if str(child.attrib.get("key", "")).strip() == str(key):
            value = str(child.attrib.get("value", "")).strip()
            return value or None
    return None


def _reachable_activity_targets(
    source_id: str,
    *,
    nodes: Mapping[str, Mapping[str, str]],
    outgoing: Mapping[str, list[str]],
) -> Set[str]:
    targets: Set[str] = set()
    queue: deque[str] = deque(outgoing.get(source_id, []))
    seen: Set[str] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if not node:
            continue
        node_type = str(node.get("type", ""))
        if node_type in _GATEWAY_TAGS:
            queue.extend(outgoing.get(current, []))
        elif node_type in _ACTIVITY_TAGS:
            targets.add(current)
        else:
            queue.extend(outgoing.get(current, []))
    return targets
