"""Stage 1 simulator for versioned BPMN logs.

Usage:
  python main.py simulate-versioned-log --config configs/tools/simulate_versioned_log_demo.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

from lxml import etree
import yaml

from src.application.services.bpmn_structure_parser_service import BpmnStructureParserService
from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.config.yaml_loader import load_yaml_with_includes


logger = logging.getLogger(__name__)

_TASK_TAGS = {
    "task",
    "userTask",
    "serviceTask",
    "scriptTask",
    "manualTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
}
_SUPPORTED_NODE_TAGS = _TASK_TAGS | {"startEvent", "endEvent", "exclusiveGateway", "parallelGateway"}
_FLOW_ERROR_TAGS = {
    "intermediateCatchEvent",
    "intermediateThrowEvent",
    "boundaryEvent",
    "eventBasedGateway",
    "inclusiveGateway",
    "complexGateway",
    "transaction",
}
_EVENT_DEFINITION_ERROR_TAGS = {
    "timerEventDefinition",
    "messageEventDefinition",
    "signalEventDefinition",
    "escalationEventDefinition",
    "errorEventDefinition",
    "compensateEventDefinition",
    "conditionalEventDefinition",
    "linkEventDefinition",
    "terminateEventDefinition",
}
_NON_FLOW_WARNING_TAGS = {
    "participant",
    "lane",
    "laneSet",
    "textAnnotation",
    "group",
    "category",
    "association",
    "bpmnDiagram",
    "bpmnPlane",
    "bpmnShape",
    "bpmnEdge",
    "documentation",
}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_dt(raw: str) -> datetime:
    dt = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    t = str(v).strip().lower()
    if t in {"1", "true", "yes", "on"}:
        return True
    if t in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_path(base_dir: Path, raw: str) -> Path:
    p = Path(str(raw).strip()).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _sample_distribution(cfg: Dict[str, Any], rng: random.Random, default_seconds: float) -> float:
    kind = str(cfg.get("type", "fixed")).strip().lower()
    if kind == "fixed":
        return max(0.001, _as_float(cfg.get("seconds", default_seconds), default_seconds))
    if kind == "lognormal":
        mean = max(1e-6, _as_float(cfg.get("mean_seconds", default_seconds), default_seconds))
        sigma = max(1e-6, _as_float(cfg.get("sigma", 0.35), 0.35))
        mu = math.log(mean) - (sigma * sigma) / 2.0
        return max(0.001, rng.lognormvariate(mu, sigma))
    if kind == "gamma":
        mean = max(1e-6, _as_float(cfg.get("mean_seconds", default_seconds), default_seconds))
        k = max(1e-6, _as_float(cfg.get("k", 3.0), 3.0))
        return max(0.001, rng.gammavariate(k, mean / k))
    if kind in {"normal", "normal_truncated", "truncnorm"}:
        mean = _as_float(cfg.get("mean_seconds", default_seconds), default_seconds)
        std = max(1e-6, _as_float(cfg.get("std_seconds", max(0.1, mean * 0.2)), max(0.1, mean * 0.2)))
        mn = _as_float(cfg.get("min_seconds", 0.1), 0.1)
        mx = _as_float(cfg.get("max_seconds", max(mn + 0.1, mean * 10.0)), max(mn + 0.1, mean * 10.0))
        for _ in range(50):
            x = rng.gauss(mean, std)
            if mn <= x <= mx:
                return max(0.001, x)
        return max(0.001, min(max(mean, mn), mx))
    return max(0.001, default_seconds)


def _sample_attr(cfg: Dict[str, Any], rng: random.Random) -> Any:
    kind = str(cfg.get("type", "fixed")).strip().lower()
    if kind == "fixed":
        return cfg.get("value")
    if kind == "categorical":
        values = cfg.get("values", {})
        if not isinstance(values, dict) or not values:
            return None
        labels = list(values.keys())
        weights = [max(0.0, _as_float(values[k], 0.0)) for k in labels]
        if sum(weights) <= 0:
            return labels[0]
        return rng.choices(labels, weights=weights, k=1)[0]
    if kind == "uniform":
        lo = _as_float(cfg.get("min", 0.0), 0.0)
        hi = _as_float(cfg.get("max", 1.0), 1.0)
        if hi < lo:
            lo, hi = hi, lo
        return float(rng.uniform(lo, hi))
    if kind == "beta":
        a = max(1e-6, _as_float(cfg.get("alpha", 2.0), 2.0))
        b = max(1e-6, _as_float(cfg.get("beta", 2.0), 2.0))
        return float(rng.betavariate(a, b))
    if kind == "lognormal":
        mean = max(1e-6, _as_float(cfg.get("mean", 1.0), 1.0))
        sigma = max(1e-6, _as_float(cfg.get("sigma", 0.4), 0.4))
        mu = math.log(mean) - (sigma * sigma) / 2.0
        return float(rng.lognormvariate(mu, sigma))
    if kind == "normal":
        return float(rng.gauss(_as_float(cfg.get("mean", 0.0), 0.0), max(1e-6, _as_float(cfg.get("std", 1.0), 1.0))))
    return cfg.get("value")


def _eval_cond(cond: Dict[str, Any], attrs: Dict[str, Any]) -> bool:
    var = str(cond.get("var", "")).strip()
    op = str(cond.get("op", "==")).strip().lower()
    val = cond.get("value")
    actual = attrs.get(var)
    if op == "==":
        return actual == val
    if op == "!=":
        return actual != val
    if op in {"<", "<=", ">", ">="}:
        try:
            left = float(actual)
            right = float(val)
        except (TypeError, ValueError):
            return False
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        return left >= right
    if op == "in":
        return actual in (val if isinstance(val, list) else [val])
    if op in {"not_in", "notin"}:
        return actual not in (val if isinstance(val, list) else [val])
    return False


def _eval_rule(rule: Dict[str, Any], attrs: Dict[str, Any]) -> bool:
    if "const" in rule:
        return bool(rule.get("const"))
    if "all" in rule and isinstance(rule.get("all"), list):
        return all(_eval_cond(x, attrs) for x in rule["all"] if isinstance(x, dict))
    if "any" in rule and isinstance(rule.get("any"), list):
        return any(_eval_cond(x, attrs) for x in rule["any"] if isinstance(x, dict))
    if {"var", "op", "value"} <= set(rule.keys()):
        return _eval_cond(rule, attrs)
    return False


@dataclass
class VersionSpec:
    version_id: str
    active_from: datetime
    bpmn_path: Path
    process_key: str


@dataclass
class NodeDef:
    node_id: str
    bpmn_tag: str
    label: str
    node_class: str


@dataclass
class EdgeDef:
    edge_id: str
    source: str
    target: str
    is_default: bool


@dataclass
class ExecGraph:
    version_id: str
    nodes: Dict[str, NodeDef]
    outgoing: Dict[str, List[EdgeDef]]
    incoming: Dict[str, List[EdgeDef]]
    start_nodes: List[str]
    end_nodes: List[str]


@dataclass
class SimEvent:
    case_id: str
    version_id: str
    activity_id: str
    activity_label: str
    bpmn_tag: str
    lifecycle: str
    timestamp: datetime
    resource_id: str
    execution_mode: str
    activity_instance_id: str
    local_order: int


@dataclass
class CaseCtx:
    case_id: str
    case_index: int
    version_id: str
    start_dt: datetime
    attrs: Dict[str, Any]
    graph: ExecGraph
    rng: random.Random
    env: Any
    events: List[SimEvent] = field(default_factory=list)
    completion_dt: Optional[datetime] = None
    seq: int = 0
    step_count: int = 0
    activity_seq: Dict[str, int] = field(default_factory=dict)
    join_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    branch_traversals: Dict[str, int] = field(default_factory=dict)
    target_completion_version_index: Optional[int] = None
    carryover_wait_applied: bool = False

    def next_order(self) -> int:
        self.seq += 1
        return self.seq

    def next_instance_id(self, node_id: str) -> str:
        n = int(self.activity_seq.get(node_id, 0)) + 1
        self.activity_seq[node_id] = n
        return f"{self.case_id}:{node_id}:{n}"


def _validate_bpmn_readiness(bpmn_text: str, version_id: str) -> Dict[str, Any]:
    root = ET.fromstring(bpmn_text)
    warnings: List[str] = []
    errors: List[str] = []
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in _NON_FLOW_WARNING_TAGS:
            warnings.append(f"{tag}:ignored_non_flow")
        if tag in _FLOW_ERROR_TAGS:
            errors.append(f"{tag}:unsupported_flow_element")
        if tag in _EVENT_DEFINITION_ERROR_TAGS:
            errors.append(f"{tag}:unsupported_event_definition")
        if tag == "subProcess":
            if str(elem.attrib.get("triggeredByEvent", "")).strip().lower() == "true":
                errors.append("subProcess(triggeredByEvent=true):unsupported_event_subprocess")
    return {"version_id": version_id, "warnings": sorted(set(warnings)), "errors": sorted(set(errors))}


def _compile_graph(dto: ProcessStructureDTO, version_id: str) -> ExecGraph:
    nodes: Dict[str, NodeDef] = {}
    for item in list(dto.nodes or []):
        node_id = str(item.get("id", "")).strip()
        bpmn_tag = str(item.get("bpmn_tag", "")).strip()
        if not node_id or not bpmn_tag:
            continue
        if bpmn_tag not in _SUPPORTED_NODE_TAGS:
            raise ValueError(f"Unsupported BPMN node in version '{version_id}': {node_id} ({bpmn_tag})")
        if bpmn_tag == "startEvent":
            node_class = "start"
        elif bpmn_tag == "endEvent":
            node_class = "end"
        elif bpmn_tag == "exclusiveGateway":
            node_class = "xor"
        elif bpmn_tag == "parallelGateway":
            node_class = "and"
        else:
            node_class = "task"
        label = str(item.get("name", "")).strip() or node_id
        nodes[node_id] = NodeDef(node_id=node_id, bpmn_tag=bpmn_tag, label=label, node_class=node_class)

    outgoing: Dict[str, List[EdgeDef]] = defaultdict(list)
    incoming: Dict[str, List[EdgeDef]] = defaultdict(list)
    for edge in list(dto.edges or []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in nodes or target not in nodes:
            continue
        edge_type = str(edge.get("edge_type", "sequence")).strip()
        if edge_type not in {"sequence", "inlined_event_bridge", "subprocess_rewire", "fork", "cancellation"}:
            continue
        edge_id = str(edge.get("id", "")).strip() or f"{source}->{target}"
        item = EdgeDef(edge_id=edge_id, source=source, target=target, is_default=bool(edge.get("is_default", False)))
        outgoing[source].append(item)
        incoming[target].append(item)

    for key in list(outgoing.keys()):
        outgoing[key] = sorted(outgoing[key], key=lambda e: (e.edge_id, e.target))
    for key in list(incoming.keys()):
        incoming[key] = sorted(incoming[key], key=lambda e: (e.edge_id, e.source))

    start_nodes = sorted([x.node_id for x in nodes.values() if x.node_class == "start"])
    end_nodes = sorted([x.node_id for x in nodes.values() if x.node_class == "end"])
    if not start_nodes:
        raise ValueError(f"Version '{version_id}' has no startEvent.")
    if not end_nodes:
        raise ValueError(f"Version '{version_id}' has no endEvent.")
    return ExecGraph(
        version_id=version_id,
        nodes=nodes,
        outgoing=outgoing,
        incoming=incoming,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
    )


def _parse_versions(cfg: Dict[str, Any], config_base_dir: Path) -> List[VersionSpec]:
    versions_raw = cfg.get("versions", [])
    if not isinstance(versions_raw, list) or not versions_raw:
        raise ValueError("'versions' must be a non-empty list.")
    out: List[VersionSpec] = []
    for i, item in enumerate(versions_raw):
        if not isinstance(item, dict):
            raise ValueError(f"versions[{i}] must be mapping.")
        version_id = str(item.get("version_id", "")).strip()
        if not version_id:
            raise ValueError(f"versions[{i}].version_id is required.")
        active_from = _parse_dt(str(item.get("active_from", "")).strip())
        bpmn_raw = str(item.get("bpmn_path", "")).strip()
        if not bpmn_raw:
            raise ValueError(f"versions[{i}].bpmn_path is required.")
        bpmn_path = _resolve_path(config_base_dir, bpmn_raw)
        if not bpmn_path.exists():
            raise FileNotFoundError(f"BPMN file not found: {bpmn_path}")
        out.append(
            VersionSpec(
                version_id=version_id,
                active_from=active_from,
                bpmn_path=bpmn_path,
                process_key=str(item.get("process_key", "")).strip(),
            )
        )
    out.sort(key=lambda x: (x.active_from, x.version_id))
    return out


def _parse_graphs(versions: List[VersionSpec], process_name: str) -> tuple[Dict[str, ExecGraph], List[Dict[str, Any]]]:
    parser = BpmnStructureParserService(
        subprocess_mode="flattened-no-subprocess-node",
        parser_mode="recover",
        inference_fallback_strategy="use_aggregated_stats",
    )
    graphs: Dict[str, ExecGraph] = {}
    reports: List[Dict[str, Any]] = []
    errors: List[str] = []
    for spec in versions:
        payload = spec.bpmn_path.read_text(encoding="utf-8")
        report = _validate_bpmn_readiness(payload, version_id=spec.version_id)
        reports.append(report)
        if report["warnings"]:
            logger.info(
                "simulate.readiness: version=%s warnings=%d",
                spec.version_id,
                len(report["warnings"]),
            )
        if report["errors"]:
            logger.error(
                "simulate.readiness: version=%s errors=%d",
                spec.version_id,
                len(report["errors"]),
            )
            errors.extend([f"{spec.version_id}: {x}" for x in report["errors"]])
            continue
        parsed = parser.parse_definition(
            definition={
                "proc_def_id": spec.version_id,
                "proc_def_key": spec.process_key,
                "deployment_id": "simulation",
                "version": spec.version_id,
                "bpmn_xml_content": payload,
            },
            catalog=[],
            process_name=process_name,
            process_filters=[spec.process_key] if spec.process_key else [],
        )
        if parsed.dto is None:
            qr = parsed.quarantine_record or {}
            errors.append(f"{spec.version_id}: {qr.get('error_code', 'parse_error')}:{qr.get('error_message', 'unknown')}")
            continue
        graphs[spec.version_id] = _compile_graph(parsed.dto, spec.version_id)
        logger.info(
            "simulate.parse: version=%s nodes=%d starts=%d ends=%d",
            spec.version_id,
            len(graphs[spec.version_id].nodes),
            len(graphs[spec.version_id].start_nodes),
            len(graphs[spec.version_id].end_nodes),
        )
    if errors:
        raise ValueError("BPMN readiness failed:\n- " + "\n- ".join(errors))
    return graphs, reports


@dataclass
class Worker:
    worker_id: str
    factor: float
    role: str
    role_factor: float
    resource: Any
    busy_seconds: float = 0.0


class ResourceManager:
    def __init__(self, env: Any, cfg: Dict[str, Any], simpy_module: Any):
        self.role_to_workers: Dict[str, List[Worker]] = defaultdict(list)
        self.worker_by_id: Dict[str, Worker] = {}
        roles = cfg.get("roles", {}) if isinstance(cfg, dict) else {}
        if not isinstance(roles, dict):
            roles = {}
        for role_name, role_data in roles.items():
            role_name_s = str(role_name).strip()
            role_cfg = role_data if isinstance(role_data, dict) else {}
            role_factor = max(0.01, _as_float(role_cfg.get("factor", 1.0), 1.0))
            workers = role_cfg.get("workers", [])
            if not isinstance(workers, list):
                workers = []
            for item in workers:
                if isinstance(item, str):
                    worker_id = item.strip()
                    factor = 1.0
                elif isinstance(item, dict):
                    worker_id = str(item.get("id", "")).strip()
                    factor = max(0.01, _as_float(item.get("factor", 1.0), 1.0))
                else:
                    continue
                if not worker_id:
                    continue
                worker = Worker(
                    worker_id=worker_id,
                    factor=factor,
                    role=role_name_s,
                    role_factor=role_factor,
                    resource=simpy_module.Resource(env, capacity=1),
                )
                self.role_to_workers[role_name_s].append(worker)
                self.worker_by_id[worker_id] = worker

    def choose_worker(self, roles: List[str]) -> Worker:
        cands: List[Worker] = []
        for role in roles:
            cands.extend(self.role_to_workers.get(str(role).strip(), []))
        if not cands:
            raise ValueError(f"No workers configured for roles={roles}.")
        cands.sort(key=lambda w: (len(w.resource.queue) + int(w.resource.count), len(w.resource.queue), w.worker_id))  # type: ignore[arg-type]
        return cands[0]

    def utilization(self, horizon_seconds: float) -> Dict[str, float]:
        if horizon_seconds <= 0:
            return {}
        return {
            worker_id: min(1.0, max(0.0, worker.busy_seconds / horizon_seconds))
            for worker_id, worker in sorted(self.worker_by_id.items())
        }


class Runtime:
    def __init__(
        self,
        simpy_module: Any,
        cfg: Dict[str, Any],
        process_name: str,
        start_dt: datetime,
        end_dt: datetime,
        versions: List[VersionSpec],
        graphs: Dict[str, ExecGraph],
        readiness_reports: List[Dict[str, Any]],
    ):
        self.simpy = simpy_module
        self.cfg = cfg
        self.process_name = process_name
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.horizon_seconds = max(0.0, (end_dt - start_dt).total_seconds())
        self.versions = sorted(versions, key=lambda x: x.active_from)
        self.graphs = graphs
        self.readiness_reports = readiness_reports
        self.seed = _as_int(cfg.get("simulation", {}).get("random_seed", 42), 42)
        self.master_rng = random.Random(self.seed)
        self.env = self.simpy.Environment()
        self.resource_manager = ResourceManager(self.env, cfg.get("resources", {}), self.simpy)
        self.arrival_cfg = cfg.get("arrival_process", {}) if isinstance(cfg.get("arrival_process", {}), dict) else {}
        self.tasks_cfg = cfg.get("tasks", {}) if isinstance(cfg.get("tasks", {}), dict) else {}
        self.gateways_cfg = cfg.get("gateways", {}) if isinstance(cfg.get("gateways", {}), dict) else {}
        self.carryover_cfg = cfg.get("version_carryover", {}) if isinstance(cfg.get("version_carryover", {}), dict) else {}
        self.case_attrs_cfg = cfg.get("case_attributes", {}) if isinstance(cfg.get("case_attributes", {}), dict) else {}
        out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
        self.emit_assign_human = _as_bool(out_cfg.get("emit_assign_for_human_tasks"), True)
        self.emit_assign_auto = _as_bool(out_cfg.get("emit_assign_for_automatic_tasks"), False)

        self.cases: List[CaseCtx] = []
        self.lifecycle_counts: Counter[str] = Counter()
        self.task_complete_counts: Counter[str] = Counter()
        self.gateway_branch_counts: Counter[str] = Counter()

    def _env_dt(self) -> datetime:
        return self.start_dt + timedelta(seconds=float(self.env.now))

    def _resolve_version(self, case_start_dt: datetime) -> VersionSpec:
        selected = self.versions[0]
        for v in self.versions:
            if v.active_from <= case_start_dt:
                selected = v
            else:
                break
        return selected

    def _version_index(self, version_id: str) -> int:
        for i, version in enumerate(self.versions):
            if version.version_id == version_id:
                return i
        return 0

    def _target_completion_index(self, start_version_index: int, rng: random.Random) -> Optional[int]:
        if not _as_bool(self.carryover_cfg.get("enabled"), False):
            return None
        targets = self.carryover_cfg.get("targets", [])
        if not isinstance(targets, list) or not targets:
            return None
        weighted: List[tuple[float, str]] = []
        for item in targets:
            if not isinstance(item, dict):
                continue
            probability = max(0.0, _as_float(item.get("probability", 0.0), 0.0))
            if probability <= 0.0:
                continue
            weighted.append((probability, str(item.get("completion", "same_version")).strip().lower()))
        total = sum(weight for weight, _ in weighted)
        if total <= 0.0:
            return None
        pick = rng.random() * total
        acc = 0.0
        selected = "same_version"
        for weight, completion in weighted:
            acc += weight
            if pick <= acc:
                selected = completion
                break

        last_index = len(self.versions) - 1
        if selected in {"same", "same_version", "none"}:
            return start_version_index
        if selected in {"next", "next_version", "plus_1"}:
            return min(last_index, start_version_index + 1)
        if selected in {"skip_one", "skip_one_version", "plus_2"}:
            return min(last_index, start_version_index + 2)
        if selected in {"last", "last_version", "final", "final_version"}:
            return last_index
        if selected.startswith("plus_"):
            try:
                delta = int(selected.removeprefix("plus_"))
            except ValueError:
                delta = 0
            return min(last_index, start_version_index + max(0, delta))
        return start_version_index

    def _case_attrs(self, rng: random.Random) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, item in sorted(self.case_attrs_cfg.items()):
            out[str(key)] = _sample_attr(item if isinstance(item, dict) else {"type": "fixed", "value": item}, rng)
        return out

    def _append_event(
        self,
        case: CaseCtx,
        node: NodeDef,
        lifecycle: str,
        resource_id: str,
        mode: str,
        instance_id: str,
    ) -> None:
        evt = SimEvent(
            case_id=case.case_id,
            version_id=case.version_id,
            activity_id=node.node_id,
            activity_label=node.label,
            bpmn_tag=node.bpmn_tag,
            lifecycle=lifecycle,
            timestamp=self._env_dt(),
            resource_id=resource_id,
            execution_mode=mode,
            activity_instance_id=instance_id,
            local_order=case.next_order(),
        )
        case.events.append(evt)
        self.lifecycle_counts[lifecycle] += 1
        if lifecycle == "complete":
            self.task_complete_counts[node.node_id] += 1

    def _task_mode(self, node: NodeDef) -> str:
        cfg = self.tasks_cfg.get(node.node_id, {}) if isinstance(self.tasks_cfg.get(node.node_id, {}), dict) else {}
        mode = str(cfg.get("execution_mode", "")).strip().lower()
        if mode in {"human", "automatic"}:
            return mode
        if node.bpmn_tag in {"userTask", "manualTask"}:
            return "human"
        if node.bpmn_tag in {"serviceTask", "scriptTask", "businessRuleTask", "sendTask", "receiveTask"}:
            return "automatic"
        return "human"

    def _task_duration(self, case: CaseCtx, node_id: str, mode: str, worker: Optional[Worker]) -> float:
        cfg = self.tasks_cfg.get(node_id, {}) if isinstance(self.tasks_cfg.get(node_id, {}), dict) else {}
        dur_cfg = cfg.get("duration", {}) if isinstance(cfg.get("duration", {}), dict) else {}
        base = _sample_distribution(dur_cfg, case.rng, 1.0 if mode == "automatic" else 60.0)
        if worker is not None:
            base *= worker.factor * worker.role_factor
        conditional_delays = cfg.get("conditional_delays", [])
        if isinstance(conditional_delays, list):
            for item in conditional_delays:
                if not isinstance(item, dict):
                    continue
                when = item.get("when", {}) if isinstance(item.get("when", {}), dict) else {}
                if not _eval_rule(when, case.attrs):
                    continue
                probability = max(0.0, min(1.0, _as_float(item.get("probability", 1.0), 1.0)))
                if probability < 1.0 and case.rng.random() >= probability:
                    continue
                delay_cfg = item.get("duration", {}) if isinstance(item.get("duration", {}), dict) else {}
                base += max(0.0, _sample_distribution(delay_cfg, case.rng, 0.0))
        noise = cfg.get("noise", {}) if isinstance(cfg.get("noise", {}), dict) else {}
        p = max(0.0, min(1.0, _as_float(noise.get("extra_delay_probability", 0.0), 0.0)))
        if p > 0 and case.rng.random() < p:
            base += max(0.0, _as_float(noise.get("extra_delay_seconds", 0.0), 0.0))
        return max(0.001, base)

    def _task_wait_delay(self, case: CaseCtx, node_id: str) -> float:
        cfg = self.tasks_cfg.get(node_id, {}) if isinstance(self.tasks_cfg.get(node_id, {}), dict) else {}
        conditional_waits = cfg.get("conditional_waits", [])
        total = 0.0
        if not isinstance(conditional_waits, list):
            return total
        for item in conditional_waits:
            if not isinstance(item, dict):
                continue
            when = item.get("when", {}) if isinstance(item.get("when", {}), dict) else {}
            if not _eval_rule(when, case.attrs):
                continue
            probability = max(0.0, min(1.0, _as_float(item.get("probability", 1.0), 1.0)))
            if probability < 1.0 and case.rng.random() >= probability:
                continue
            delay_cfg = item.get("duration", {}) if isinstance(item.get("duration", {}), dict) else {}
            total += max(0.0, _sample_distribution(delay_cfg, case.rng, 0.0))
        return total

    def _choose_xor_edge(self, case: CaseCtx, node_id: str, outgoing: List[EdgeDef]) -> EdgeDef:
        gw = self.gateways_cfg.get(node_id, {}) if isinstance(self.gateways_cfg.get(node_id, {}), dict) else {}
        branches = gw.get("branches", []) if isinstance(gw.get("branches", []), list) else []
        flow_map = {e.edge_id: e for e in outgoing}
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            flow_id = str(branch.get("flow_id", "")).strip()
            if flow_id not in flow_map:
                continue
            branch_key = f"{node_id}:{flow_id}"
            traversed = int(case.branch_traversals.get(branch_key, 0))
            max_traversals_raw = branch.get("max_traversals_per_case")
            max_traversals = (
                _as_int(max_traversals_raw, 0)
                if max_traversals_raw not in (None, "")
                else 0
            )
            if max_traversals > 0 and traversed >= max_traversals:
                continue
            when = branch.get("when", {}) if isinstance(branch.get("when", {}), dict) else {}
            if _eval_rule(when, case.attrs):
                probability = max(0.0, min(1.0, _as_float(branch.get("probability", 1.0), 1.0)))
                repeat_until_max = _as_bool(branch.get("repeat_until_max_once_selected"), False)
                sticky_repeat = repeat_until_max and traversed > 0 and (max_traversals <= 0 or traversed < max_traversals)
                if not sticky_repeat and probability < 1.0 and case.rng.random() >= probability:
                    continue
                case.branch_traversals[branch_key] = traversed + 1
                self.gateway_branch_counts[branch_key] += 1
                return flow_map[flow_id]
        default_flow_id = str(gw.get("default_flow_id", "")).strip()
        if default_flow_id and default_flow_id in flow_map:
            branch_key = f"{node_id}:{default_flow_id}"
            case.branch_traversals[branch_key] = int(case.branch_traversals.get(branch_key, 0)) + 1
            self.gateway_branch_counts[branch_key] += 1
            return flow_map[default_flow_id]
        defaults = [e for e in outgoing if e.is_default]
        if defaults:
            edge = sorted(defaults, key=lambda x: x.edge_id)[0]
            branch_key = f"{node_id}:{edge.edge_id}"
            case.branch_traversals[branch_key] = int(case.branch_traversals.get(branch_key, 0)) + 1
            self.gateway_branch_counts[branch_key] += 1
            return edge
        edge = sorted(outgoing, key=lambda x: x.edge_id)[0]
        branch_key = f"{node_id}:{edge.edge_id}"
        case.branch_traversals[branch_key] = int(case.branch_traversals.get(branch_key, 0)) + 1
        self.gateway_branch_counts[branch_key] += 1
        return edge

    def _is_terminal_task(self, case: CaseCtx, node: NodeDef) -> bool:
        outgoing = list(case.graph.outgoing.get(node.node_id, []))
        if not outgoing:
            return True
        return all(case.graph.nodes.get(edge.target).node_class == "end" for edge in outgoing if edge.target in case.graph.nodes)

    def _carryover_delay_seconds(self, case: CaseCtx) -> float:
        if case.carryover_wait_applied:
            return 0.0
        target_index = case.target_completion_version_index
        if target_index is None:
            return 0.0
        current_index = self._version_index(_version_at(self._env_dt(), self.versions))
        if target_index <= current_index:
            case.carryover_wait_applied = True
            return 0.0
        target_dt = self.versions[target_index].active_from
        jitter_cfg = self.carryover_cfg.get("jitter_seconds", {})
        if not isinstance(jitter_cfg, dict):
            jitter_cfg = {"type": "uniform", "min": 3600, "max": 604800}
        jitter = max(0.0, _sample_distribution(jitter_cfg, case.rng, 0.0))
        wait_until = target_dt + timedelta(seconds=jitter)
        delay = max(0.0, (wait_until - self._env_dt()).total_seconds())
        case.carryover_wait_applied = True
        return delay

    def _exec_task(self, case: CaseCtx, node: NodeDef) -> Iterable[Any]:
        mode = self._task_mode(node)
        inst_id = case.next_instance_id(node.node_id)
        wait_delay = self._task_wait_delay(case, node.node_id)
        if wait_delay > 0.0:
            yield self.env.timeout(wait_delay)
        if self._is_terminal_task(case, node):
            delay = self._carryover_delay_seconds(case)
            if delay > 0.0:
                yield self.env.timeout(delay)
        if mode == "human":
            cfg = self.tasks_cfg.get(node.node_id, {}) if isinstance(self.tasks_cfg.get(node.node_id, {}), dict) else {}
            roles = cfg.get("roles", [])
            if not isinstance(roles, list) or not roles:
                raise ValueError(f"Task '{node.node_id}' is human but has no roles.")
            worker = self.resource_manager.choose_worker([str(x).strip() for x in roles])
            if self.emit_assign_human:
                self._append_event(case, node, "assign", worker.worker_id, mode, inst_id)
            req = worker.resource.request()
            yield req
            busy_start = float(self.env.now)
            self._append_event(case, node, "start", worker.worker_id, mode, inst_id)
            duration = self._task_duration(case, node.node_id, mode, worker)
            yield self.env.timeout(duration)
            worker.busy_seconds += max(0.0, float(self.env.now) - busy_start)
            worker.resource.release(req)
            self._append_event(case, node, "complete", worker.worker_id, mode, inst_id)
            return
        if self.emit_assign_auto:
            self._append_event(case, node, "assign", "SYSTEM", mode, inst_id)
        self._append_event(case, node, "start", "SYSTEM", mode, inst_id)
        duration = self._task_duration(case, node.node_id, mode, None)
        yield self.env.timeout(duration)
        self._append_event(case, node, "complete", "SYSTEM", mode, inst_id)

    def _exec_node(self, case: CaseCtx, node_id: str) -> Iterable[Any]:
        case.step_count += 1
        if case.step_count > 10000:
            raise RuntimeError(f"Case '{case.case_id}' exceeded max steps.")
        graph = case.graph
        node = graph.nodes[node_id]
        outgoing = list(graph.outgoing.get(node_id, []))
        incoming = list(graph.incoming.get(node_id, []))

        if node.node_class == "start":
            if outgoing:
                yield self.env.process(self._exec_node(case, outgoing[0].target))
            return
        if node.node_class == "end":
            return
        if node.node_class == "task":
            for x in self._exec_task(case, node):
                yield x
            if not outgoing:
                return
            if len(outgoing) == 1:
                yield self.env.process(self._exec_node(case, outgoing[0].target))
                return
            jobs = [self.env.process(self._exec_node(case, e.target)) for e in outgoing]
            yield self.env.all_of(jobs)
            return
        if node.node_class == "xor":
            if not outgoing:
                return
            edge = self._choose_xor_edge(case, node_id, outgoing)
            yield self.env.process(self._exec_node(case, edge.target))
            return
        if node.node_class == "and":
            in_count = len(incoming)
            out_count = len(outgoing)
            is_join = in_count > 1
            is_split = out_count > 1
            if is_join:
                st = case.join_state.get(node_id)
                if st is None:
                    st = {"arrived": 0, "event": self.env.event()}
                    case.join_state[node_id] = st
                st["arrived"] = int(st["arrived"]) + 1
                if int(st["arrived"]) < in_count:
                    yield st["event"]
                    return
                if not st["event"].triggered:
                    st["event"].succeed(True)
                st["arrived"] = 0
                st["event"] = self.env.event()
            if not outgoing:
                return
            if is_split:
                jobs = [self.env.process(self._exec_node(case, e.target)) for e in outgoing]
                yield self.env.all_of(jobs)
                return
            yield self.env.process(self._exec_node(case, outgoing[0].target))
            return
        raise ValueError(f"Unsupported node class '{node.node_class}'.")

    def _arrivals(self) -> List[float]:
        kind = str(self.arrival_cfg.get("type", "poisson")).strip().lower()
        if kind != "poisson":
            raise ValueError(f"Unsupported arrival_process.type='{kind}'.")
        rate_h = max(1e-9, _as_float(self.arrival_cfg.get("rate_per_hour", 1.0), 1.0))
        lam = rate_h / 3600.0
        max_cases_raw = self.arrival_cfg.get("max_cases")
        max_cases = _as_int(max_cases_raw, 0) if max_cases_raw not in (None, "") else 0
        t = 0.0
        out: List[float] = []
        while t <= self.horizon_seconds:
            if max_cases > 0 and len(out) >= max_cases:
                break
            u = max(1e-12, self.master_rng.random())
            t += -math.log(u) / lam
            if t > self.horizon_seconds:
                break
            out.append(float(t))
        return out

    def _run_case(self, arrival: float, case_index: int) -> Iterable[Any]:
        if arrival > float(self.env.now):
            yield self.env.timeout(arrival - float(self.env.now))
        start_dt = self._env_dt()
        version = self._resolve_version(start_dt)
        rng = random.Random(self.seed + case_index * 100_003 + 17)
        start_version_index = self._version_index(version.version_id)
        case = CaseCtx(
            case_id=f"case_{case_index:06d}",
            case_index=case_index,
            version_id=version.version_id,
            start_dt=start_dt,
            attrs=self._case_attrs(rng),
            graph=self.graphs[version.version_id],
            rng=rng,
            env=self.env,
            target_completion_version_index=self._target_completion_index(start_version_index, rng),
        )
        self.cases.append(case)
        yield self.env.process(self._exec_node(case, case.graph.start_nodes[0]))
        case.completion_dt = self._env_dt()

    def run(self) -> Dict[str, Any]:
        arrivals = self._arrivals()
        logger.info(
            "simulate.arrivals: generated=%d horizon_hours=%.2f rate_per_hour=%.3f",
            len(arrivals),
            self.horizon_seconds / 3600.0,
            max(1e-9, _as_float(self.arrival_cfg.get("rate_per_hour", 1.0), 1.0)),
        )
        for i, t in enumerate(arrivals, start=1):
            self.env.process(self._run_case(t, i))
        self.env.run()
        logger.info("simulate.runtime: completed_cases=%d", len(self.cases))

        cases_by_version: Counter[str] = Counter()
        cycle_by_version: Dict[str, List[float]] = defaultdict(list)
        for case in self.cases:
            cases_by_version[case.version_id] += 1
            if case.completion_dt is not None:
                cycle_by_version[case.version_id].append(max(0.0, (case.completion_dt - case.start_dt).total_seconds()))
        mean_cycle = {
            version: (sum(values) / max(1, len(values)))
            for version, values in sorted(cycle_by_version.items())
        }
        rate_h = max(1e-9, _as_float(self.arrival_cfg.get("rate_per_hour", 1.0), 1.0))
        horizon_h = self.horizon_seconds / 3600.0
        expected = rate_h * horizon_h
        std = math.sqrt(expected) if expected >= 0 else 0.0
        max_cases_raw = self.arrival_cfg.get("max_cases")
        max_cases = _as_int(max_cases_raw, 0) if max_cases_raw not in (None, "") else None
        return {
            "status": "ok",
            "process_name": self.process_name,
            "random_seed": self.seed,
            "start_time": _iso(self.start_dt),
            "end_time": _iso(self.end_dt),
            "arrival_stats": {
                "type": "poisson",
                "rate_per_hour": rate_h,
                "horizon_hours": horizon_h,
                "expected_cases": expected,
                "std_cases": std,
                "actual_cases": len(arrivals),
                "max_cases": max_cases,
            },
            "case_count_total": len(self.cases),
            "case_count_by_version": dict(sorted(cases_by_version.items())),
            "event_count_total": int(sum(self.lifecycle_counts.values())),
            "event_count_by_lifecycle": dict(sorted(self.lifecycle_counts.items())),
            "task_count_by_activity": dict(sorted(self.task_complete_counts.items())),
            "gateway_branch_counts": dict(sorted(self.gateway_branch_counts.items())),
            "resource_utilization_estimate": self.resource_manager.utilization(self.horizon_seconds),
            "mean_cycle_time_by_version": mean_cycle,
            "readiness_reports": self.readiness_reports,
        }


def _xes_attr(parent: etree._Element, tag: str, key: str, value: str) -> None:
    child = etree.SubElement(parent, tag)
    child.set("key", key)
    child.set("value", value)


def _write_xes(
    cases: List[CaseCtx],
    process_name: str,
    output_path: Path,
    trace_level_case_attrs: bool,
    duplicate_case_attrs_on_events: bool,
) -> None:
    root = etree.Element("log")
    root.set("xes.version", "1.0")
    root.set("xes.features", "nested-attributes")
    root.set("openxes.version", "1.0RC7")
    root.set("xmlns", "http://www.xes-standard.org/")

    for case in sorted(cases, key=lambda c: (c.start_dt, c.case_id)):
        trace = etree.SubElement(root, "trace")
        _xes_attr(trace, "string", "concept:name", case.case_id)
        _xes_attr(trace, "string", "concept:version", case.version_id)
        _xes_attr(trace, "string", "sim:process_name", process_name)
        _xes_attr(trace, "date", "sim:case_start_time", _iso(case.start_dt))
        _xes_attr(trace, "string", "sim:generated_by", "simulate_versioned_log.py")
        if trace_level_case_attrs:
            for k, v in sorted(case.attrs.items()):
                _xes_attr(trace, "string", str(k), str(v))

        events = sorted(case.events, key=lambda e: (e.timestamp, e.local_order))
        for event in events:
            evt = etree.SubElement(trace, "event")
            _xes_attr(evt, "string", "concept:name", event.activity_id)
            _xes_attr(evt, "string", "sim:activity_label", event.activity_label)
            _xes_attr(evt, "string", "sim:bpmn_element_id", event.activity_id)
            _xes_attr(evt, "string", "sim:bpmn_tag", event.bpmn_tag)
            _xes_attr(evt, "string", "lifecycle:transition", event.lifecycle)
            _xes_attr(evt, "date", "time:timestamp", _iso(event.timestamp))
            _xes_attr(evt, "string", "org:resource", event.resource_id)
            _xes_attr(evt, "string", "concept:version", event.version_id)
            _xes_attr(evt, "string", "sim:execution_mode", event.execution_mode)
            _xes_attr(evt, "string", "sim:activity_instance_id", event.activity_instance_id)
            if duplicate_case_attrs_on_events:
                for k, v in sorted(case.attrs.items()):
                    _xes_attr(evt, "string", str(k), str(v))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(str(output_path), xml_declaration=True, encoding="utf-8", pretty_print=True)


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _percent(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(part) * 100.0 / float(total)


def _describe_numeric(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "avg": None,
            "p50": None,
            "p95": None,
            "std": None,
            "coefficient_of_variation": None,
            "skewness": None,
        }
    xs = sorted(float(x) for x in values)
    n = len(xs)
    mean = sum(xs) / n
    variance = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(variance)
    p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    skewness = sum(((x - mean) / std) ** 3 for x in xs) / n if std > 0 else 0.0
    return {
        "count": n,
        "min": xs[0],
        "max": xs[-1],
        "mean": mean,
        "avg": mean,
        "p50": xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0,
        "p95": xs[p95_idx],
        "std": std,
        "coefficient_of_variation": (std / mean) if mean else None,
        "skewness": skewness,
    }


def _gini(values: List[float]) -> float:
    xs = sorted(max(0.0, float(x)) for x in values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total <= 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _normalized_entropy(counts: Iterable[int]) -> float:
    xs = [float(x) for x in counts if int(x) > 0]
    total = sum(xs)
    if total <= 0 or len(xs) <= 1:
        return 0.0
    entropy = -sum((x / total) * math.log(x / total) for x in xs)
    return entropy / math.log(len(xs))


def _version_at(dt: Optional[datetime], versions: List[VersionSpec]) -> str:
    if dt is None:
        return versions[0].version_id if versions else ""
    selected = versions[0] if versions else None
    for version in versions:
        if version.active_from <= dt:
            selected = version
        else:
            break
    return selected.version_id if selected else ""


def _calendar_month_delta(start_dt: datetime, completion_dt: Optional[datetime]) -> int:
    if completion_dt is None:
        return 0
    return max(0, (completion_dt.year - start_dt.year) * 12 + completion_dt.month - start_dt.month)


def _resource_stats(cases: List[CaseCtx]) -> Dict[str, Any]:
    complete_by_resource: Counter[str] = Counter()
    complete_by_task_resource: Dict[str, Counter[str]] = defaultdict(Counter)
    complete_by_resource_task: Dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        for evt in case.events:
            if evt.lifecycle != "complete":
                continue
            complete_by_resource[evt.resource_id] += 1
            complete_by_task_resource[evt.activity_id][evt.resource_id] += 1
            complete_by_resource_task[evt.resource_id][evt.activity_id] += 1

    total_completes = sum(complete_by_resource.values())
    resource_task_distribution_percent: Dict[str, Dict[str, float]] = {}
    for resource, counts in sorted(complete_by_resource_task.items()):
        resource_total = sum(counts.values())
        resource_task_distribution_percent[resource] = {
            task: _percent(count, resource_total)
            for task, count in sorted(counts.items())
        }
    task_resource_distribution_percent: Dict[str, Dict[str, float]] = {}
    for task, counts in sorted(complete_by_task_resource.items()):
        task_total = sum(counts.values())
        task_resource_distribution_percent[task] = {
            resource: _percent(count, task_total)
            for resource, count in sorted(counts.items())
        }

    resources = sorted(complete_by_resource)
    human_resources = [resource for resource in resources if resource != "SYSTEM"]
    return {
        "unique_resource_count": len(resources),
        "unique_human_resource_count": len(human_resources),
        "resources": resources,
        "human_resources": human_resources,
        "complete_count_by_resource": dict(sorted(complete_by_resource.items())),
        "complete_percent_by_resource": {
            resource: _percent(count, total_completes)
            for resource, count in sorted(complete_by_resource.items())
        },
        "resource_task_distribution_percent": resource_task_distribution_percent,
        "task_resource_distribution_percent": task_resource_distribution_percent,
    }


def _node_coverage_stats(task_node_ids: Iterable[str], activity_counts: Counter[str]) -> Dict[str, Any]:
    nodes = sorted(set(task_node_ids))
    used = [node for node in nodes if activity_counts.get(node, 0) > 0]
    missing = [node for node in nodes if activity_counts.get(node, 0) <= 0]
    usage_values = [float(activity_counts.get(node, 0)) for node in nodes]
    return {
        "task_nodes_total": len(nodes),
        "task_nodes_used": len(used),
        "task_nodes_missing_count": len(missing),
        "task_nodes_missing": missing,
        "coverage_percent": _percent(len(used), len(nodes)),
        "usage_distribution": _describe_numeric(usage_values),
        "usage_gini": _gini(usage_values),
        "activity_distribution_entropy_normalized": _normalized_entropy(int(x) for x in usage_values),
    }


def _parallel_task_pairs(graph: ExecGraph) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()

    def branch_tasks(start_node_id: str) -> set[str]:
        tasks: set[str] = set()
        seen: set[str] = set()
        stack = [start_node_id]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if node.node_class == "and" and len(graph.incoming.get(node_id, [])) > 1:
                continue
            if node.node_class == "task":
                tasks.add(node_id)
            for edge in graph.outgoing.get(node_id, []):
                stack.append(edge.target)
        return tasks

    for node_id, node in graph.nodes.items():
        outgoing = graph.outgoing.get(node_id, [])
        if node.node_class != "and" or len(outgoing) <= 1:
            continue
        branches = [branch_tasks(edge.target) for edge in outgoing]
        for i, left in enumerate(branches):
            for right in branches[i + 1:]:
                for a in left:
                    for b in right:
                        if a != b:
                            pairs.add(frozenset({a, b}))
    return pairs


def _bpms_native_operational_variance_stats(cases: List[CaseCtx], graphs: Dict[str, ExecGraph]) -> Dict[str, Any]:
    parallel_pairs_by_version = {
        version_id: _parallel_task_pairs(graph)
        for version_id, graph in graphs.items()
    }
    task_tag_by_version = {
        version_id: {node_id: node.bpmn_tag for node_id, node in graph.nodes.items()}
        for version_id, graph in graphs.items()
    }
    task_label_by_version = {
        version_id: {node_id: node.label for node_id, node in graph.nodes.items()}
        for version_id, graph in graphs.items()
    }

    trace_count = len(cases)
    interleaving_trace_count = 0
    interleaving_adjacent_count = 0
    adjacent_transition_count = 0
    orientation_counts: Counter[str] = Counter()
    unordered_orientation_seen: Dict[str, set[str]] = defaultdict(set)

    technical_repeated_trace_count = 0
    technical_repeat_count = 0
    incident_like_trace_count = 0
    incident_like_count = 0
    repeated_technical_tasks: Counter[str] = Counter()
    incident_keywords = ("incident", "escalat", "cancel", "reject", "blacklist", "restore", "return", "reopen")
    technical_tags = {"serviceTask", "scriptTask", "businessRuleTask", "sendTask", "receiveTask"}

    assign_resource_by_instance: Dict[str, set[str]] = defaultdict(set)
    complete_by_task_resource: Dict[str, Counter[str]] = defaultdict(Counter)
    human_complete_count = 0
    human_task_complete_count: Counter[str] = Counter()

    for case in cases:
        graph = graphs.get(case.version_id)
        if graph is None:
            continue
        parallel_pairs = parallel_pairs_by_version.get(case.version_id, set())
        tag_by_task = task_tag_by_version.get(case.version_id, {})
        label_by_task = task_label_by_version.get(case.version_id, {})
        complete_events = sorted(
            [evt for evt in case.events if evt.lifecycle == "complete"],
            key=lambda evt: (evt.timestamp, evt.local_order),
        )
        case_interleaving = False
        for left, right in zip(complete_events, complete_events[1:]):
            adjacent_transition_count += 1
            pair = frozenset({left.activity_id, right.activity_id})
            if pair not in parallel_pairs:
                continue
            case_interleaving = True
            interleaving_adjacent_count += 1
            orientation = f"{left.activity_id}->{right.activity_id}"
            unordered = "|".join(sorted(pair))
            orientation_counts[orientation] += 1
            unordered_orientation_seen[unordered].add(orientation)
        if case_interleaving:
            interleaving_trace_count += 1

        counts = Counter(evt.activity_id for evt in complete_events)
        mode_by_task = {evt.activity_id: evt.execution_mode for evt in complete_events}
        case_has_technical_repeat = False
        case_has_incident_like = False
        for task_id, count in counts.items():
            label = label_by_task.get(task_id, task_id).lower()
            normalized_task = task_id.lower()
            mode = str(mode_by_task.get(task_id, "")).strip().lower()
            if mode == "automatic" and count > 1:
                case_has_technical_repeat = True
                repeats = count - 1
                technical_repeat_count += repeats
                repeated_technical_tasks[task_id] += repeats
            if any(keyword in normalized_task or keyword in label for keyword in incident_keywords):
                case_has_incident_like = True
                incident_like_count += count
        if case_has_technical_repeat:
            technical_repeated_trace_count += 1
        if case_has_incident_like:
            incident_like_trace_count += 1

        for evt in case.events:
            if evt.lifecycle == "assign":
                assign_resource_by_instance[evt.activity_instance_id].add(evt.resource_id)
            if evt.lifecycle != "complete":
                continue
            tag = tag_by_task.get(evt.activity_id, "")
            if tag in {"userTask", "manualTask"} or evt.resource_id != "SYSTEM":
                human_complete_count += 1
                human_task_complete_count[evt.activity_id] += 1
                complete_by_task_resource[evt.activity_id][evt.resource_id] += 1

    actual_reassignment_instances = {
        instance_id: resources
        for instance_id, resources in assign_resource_by_instance.items()
        if len(resources) > 1
    }
    multi_resource_task_count = sum(1 for counts in complete_by_task_resource.values() if len(counts) > 1)
    non_dominant_completion_count = 0
    task_substitution_percent: Dict[str, float] = {}
    for task_id, counts in sorted(complete_by_task_resource.items()):
        task_total = sum(counts.values())
        dominant = max(counts.values()) if counts else 0
        non_dominant = max(0, task_total - dominant)
        non_dominant_completion_count += non_dominant
        task_substitution_percent[task_id] = _percent(non_dominant, task_total)

    both_orientation_pair_count = sum(1 for orientations in unordered_orientation_seen.values() if len(orientations) > 1)
    return {
        "metric_label": "BPMS-Native Operational Variance",
        "metric_note": "These metrics describe BPMS-native operational variance in flattened event logs, not data corruption noise.",
        "concurrency_interleaving": {
            "parallel_task_pair_count": len(set().union(*parallel_pairs_by_version.values())) if parallel_pairs_by_version else 0,
            "trace_count_with_parallel_adjacent_interleaving": interleaving_trace_count,
            "trace_percent_with_parallel_adjacent_interleaving": _percent(interleaving_trace_count, trace_count),
            "parallel_adjacent_transition_count": interleaving_adjacent_count,
            "parallel_adjacent_transition_percent": _percent(interleaving_adjacent_count, adjacent_transition_count),
            "parallel_pairs_with_both_flattened_orders": both_orientation_pair_count,
            "top_parallel_adjacent_orientations": dict(orientation_counts.most_common(10)),
        },
        "technical_retries_and_incidents": {
            "trace_count_with_repeated_technical_task": technical_repeated_trace_count,
            "trace_percent_with_repeated_technical_task": _percent(technical_repeated_trace_count, trace_count),
            "technical_repeat_count": technical_repeat_count,
            "top_repeated_technical_tasks": dict(repeated_technical_tasks.most_common(10)),
            "trace_count_with_incident_like_activity": incident_like_trace_count,
            "trace_percent_with_incident_like_activity": _percent(incident_like_trace_count, trace_count),
            "incident_like_activity_count": incident_like_count,
        },
        "resource_reassignment_and_delegation": {
            "actual_reassignment_instance_count": len(actual_reassignment_instances),
            "actual_reassignment_instance_percent_of_human_completions": _percent(len(actual_reassignment_instances), human_complete_count),
            "human_task_count_with_multiple_resources": multi_resource_task_count,
            "human_task_count": len(human_task_complete_count),
            "human_task_percent_with_multiple_resources": _percent(multi_resource_task_count, len(human_task_complete_count)),
            "non_dominant_resource_completion_count": non_dominant_completion_count,
            "non_dominant_resource_completion_percent": _percent(non_dominant_completion_count, human_complete_count),
            "top_task_substitution_percent": dict(
                sorted(task_substitution_percent.items(), key=lambda item: item[1], reverse=True)[:10]
            ),
        },
    }


def _build_dataset_stats(
    cases: List[CaseCtx],
    versions: List[VersionSpec],
    graphs: Dict[str, ExecGraph],
) -> Dict[str, Any]:
    versions_sorted = sorted(versions, key=lambda item: item.active_from)
    version_index = {version.version_id: i for i, version in enumerate(versions_sorted)}
    all_task_nodes = {
        node_id
        for graph in graphs.values()
        for node_id, node in graph.nodes.items()
        if node.node_class == "task"
    }

    def build_scope(scope_cases: List[CaseCtx], task_nodes: Iterable[str], scope_graphs: Dict[str, ExecGraph]) -> Dict[str, Any]:
        trace_complete_counts: List[float] = []
        trace_transition_counts: List[float] = []
        trace_cycle_counts: List[float] = []
        trace_repeated_activity_counts: List[float] = []
        activity_counts: Counter[str] = Counter()
        cycle_depth_counts: Counter[str] = Counter({"none": 0, "double": 0, "triple": 0, "more_than_3": 0})
        carryover_counts: Counter[str] = Counter({f"plus_{i}": 0 for i in range(1, 5)})
        carryover_counts["none"] = 0
        carryover_counts["plus_4_or_more"] = 0
        calendar_carryover_counts: Counter[str] = Counter({f"plus_{i}month": 0 for i in range(1, 5)})
        calendar_carryover_counts["same_month"] = 0
        calendar_carryover_counts["plus_4month_or_more"] = 0

        for case in scope_cases:
            complete_events = sorted(
                [evt for evt in case.events if evt.lifecycle == "complete"],
                key=lambda evt: (evt.timestamp, evt.local_order),
            )
            activities = [evt.activity_id for evt in complete_events]
            counts = Counter(activities)
            activity_counts.update(counts)
            complete_count = len(activities)
            repeated_activity_count = sum(1 for count in counts.values() if count > 1)
            cycle_count = sum(count - 1 for count in counts.values() if count > 1)
            max_repetition = max(counts.values()) if counts else 0
            trace_complete_counts.append(float(complete_count))
            trace_transition_counts.append(float(max(0, complete_count - 1)))
            trace_cycle_counts.append(float(cycle_count))
            trace_repeated_activity_counts.append(float(repeated_activity_count))
            if max_repetition <= 1:
                cycle_depth_counts["none"] += 1
            elif max_repetition == 2:
                cycle_depth_counts["double"] += 1
            elif max_repetition == 3:
                cycle_depth_counts["triple"] += 1
            else:
                cycle_depth_counts["more_than_3"] += 1

            completion_version = _version_at(case.completion_dt, versions_sorted)
            delta = int(version_index.get(completion_version, 0)) - int(version_index.get(case.version_id, 0))
            if delta <= 0:
                carryover_counts["none"] += 1
            elif delta == 4:
                carryover_counts["plus_4"] += 1
            elif delta > 4:
                carryover_counts["plus_4_or_more"] += 1
            else:
                carryover_counts[f"plus_{delta}"] += 1

            month_delta = _calendar_month_delta(case.start_dt, case.completion_dt)
            if month_delta <= 0:
                calendar_carryover_counts["same_month"] += 1
            elif month_delta == 4:
                calendar_carryover_counts["plus_4month"] += 1
            elif month_delta > 4:
                calendar_carryover_counts["plus_4month_or_more"] += 1
            else:
                calendar_carryover_counts[f"plus_{month_delta}month"] += 1

        trace_count = len(scope_cases)
        return {
            "trace_count": trace_count,
            "activity_complete_count_per_trace": _describe_numeric(trace_complete_counts),
            "activity_transition_count_per_trace": _describe_numeric(trace_transition_counts),
            "cycles": {
                "cycle_count_total": int(sum(trace_cycle_counts)),
                "cycle_count_per_trace": _describe_numeric(trace_cycle_counts),
                "repeated_activity_count_per_trace": _describe_numeric(trace_repeated_activity_counts),
                "trace_count_by_max_repetition": dict(sorted(cycle_depth_counts.items())),
                "trace_percent_by_max_repetition": {
                    key: _percent(value, trace_count)
                    for key, value in sorted(cycle_depth_counts.items())
                },
            },
            "version_carryover": {
                "trace_count_by_completion_delta": dict(sorted(carryover_counts.items())),
                "trace_percent_by_completion_delta": {
                    key: _percent(value, trace_count)
                    for key, value in sorted(carryover_counts.items())
                },
            },
            "calendar_carryover": {
                "trace_count_by_completion_month_delta": dict(sorted(calendar_carryover_counts.items())),
                "trace_percent_by_completion_month_delta": {
                    key: _percent(value, trace_count)
                    for key, value in sorted(calendar_carryover_counts.items())
                },
            },
            "node_coverage": _node_coverage_stats(task_nodes, activity_counts),
            "activity_complete_count_by_node": dict(sorted(activity_counts.items())),
            "resources": _resource_stats(scope_cases),
            "bpms_native_operational_variance": _bpms_native_operational_variance_stats(scope_cases, scope_graphs),
        }

    by_version: Dict[str, Any] = {}
    for version in versions_sorted:
        version_cases = [case for case in cases if case.version_id == version.version_id]
        graph = graphs.get(version.version_id)
        graph_items = graph.nodes.items() if graph is not None else []
        version_task_nodes = {
            node_id
            for node_id, node in graph_items
            if node.node_class == "task"
        }
        by_version[version.version_id] = build_scope(version_cases, version_task_nodes, {version.version_id: graph} if graph is not None else {})

    return {
        "schema_version": "1.0",
        "metric_notes": {
            "activity_complete_count_per_trace": "Number of complete lifecycle task events in a trace.",
            "activity_transition_count_per_trace": "Complete task events minus one.",
            "cycles": "Cycle depth is inferred from repeated completed activity ids inside one trace.",
            "version_carryover": "Completion version is derived from trace completion timestamp and version active_from boundaries.",
            "calendar_carryover": "Completion month delta is derived from trace start and completion calendar months.",
            "bpms_native_operational_variance": "BPMS-native operational variance covers flattened parallel interleaving, technical retries/incidents, and resource substitution/delegation.",
            "resources": "Resource distributions are computed from complete lifecycle task events.",
        },
        "total": build_scope(cases, all_task_nodes, graphs),
        "by_version": by_version,
    }


def _write_generated_data_config(
    path: Path,
    xes_path: Path,
    process_name: str,
    emit_assign_human: bool,
    emit_assign_auto: bool,
) -> Dict[str, Any]:
    start_transitions = ["assign", "start"] if (emit_assign_human or emit_assign_auto) else ["start"]
    payload = {
        "data": {
            "dataset_label": f"{process_name}_simulated",
            "log_path": str(xes_path),
        },
        "mapping": {
            "xes_adapter": {
                "case_id_key": "concept:name",
                "activity_key": "concept:name",
                "timestamp_key": "time:timestamp",
                "resource_key": "org:resource",
                "lifecycle_key": "lifecycle:transition",
                "version_key": "concept:version",
                "start_transitions": start_transitions,
                "complete_transitions": ["complete"],
                "pairing_strategy": "lifo",
                "use_classifier": True,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return payload


def run(cfg: Dict[str, Any], *, config_base_dir: Optional[Path] = None) -> Dict[str, Any]:
    try:
        import simpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The simulator requires 'simpy'. Install dependencies from requirements.txt.") from exc

    base_dir = Path(config_base_dir or Path.cwd()).resolve()
    simulation = cfg.get("simulation", {}) if isinstance(cfg.get("simulation", {}), dict) else {}
    process_name = str(simulation.get("process_name", "simulated_process")).strip() or "simulated_process"
    start_dt = _parse_dt(str(simulation.get("start_time", "")).strip())
    end_dt = _parse_dt(str(simulation.get("end_time", "")).strip())
    if end_dt <= start_dt:
        raise ValueError("simulation.end_time must be greater than simulation.start_time.")

    versions = _parse_versions(cfg, config_base_dir=base_dir)
    logger.info(
        "simulate.config: process_name=%s versions=%s start=%s end=%s",
        process_name,
        [v.version_id for v in versions],
        _iso(start_dt),
        _iso(end_dt),
    )
    graphs, readiness_reports = _parse_graphs(versions, process_name)
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    xes_path = _resolve_path(base_dir, str(output_cfg.get("xes_path", "outputs/simulation/simulated.xes")))
    summary_path = _resolve_path(base_dir, str(output_cfg.get("summary_json_path", "outputs/simulation/simulated.summary.json")))
    default_stats_path = summary_path.with_name(f"{summary_path.stem}.dataset_stats.json")
    dataset_stats_path = _resolve_path(base_dir, str(output_cfg.get("dataset_stats_json_path", str(default_stats_path))))
    data_cfg_path = _resolve_path(base_dir, str(output_cfg.get("generated_data_config_path", "configs/data/generated_simulated.yaml")))
    overwrite = _as_bool(output_cfg.get("overwrite"), True)
    if not overwrite:
        for p in (xes_path, summary_path, dataset_stats_path, data_cfg_path):
            if p.exists():
                raise FileExistsError(f"Output exists and overwrite=false: {p}")

    runtime = Runtime(
        simpy,
        cfg=cfg,
        process_name=process_name,
        start_dt=start_dt,
        end_dt=end_dt,
        versions=versions,
        graphs=graphs,
        readiness_reports=readiness_reports,
    )
    summary = runtime.run()
    trace_level_case_attrs = _as_bool(output_cfg.get("trace_level_case_attrs"), True)
    duplicate_case_attrs_on_events = _as_bool(output_cfg.get("duplicate_case_attrs_on_events"), False)
    _write_xes(
        runtime.cases,
        process_name=process_name,
        output_path=xes_path,
        trace_level_case_attrs=trace_level_case_attrs,
        duplicate_case_attrs_on_events=duplicate_case_attrs_on_events,
    )
    _write_summary(summary_path, summary)
    dataset_stats = _build_dataset_stats(runtime.cases, versions, graphs)
    _write_summary(dataset_stats_path, dataset_stats)
    data_cfg_payload = _write_generated_data_config(
        data_cfg_path,
        xes_path=xes_path,
        process_name=process_name,
        emit_assign_human=runtime.emit_assign_human,
        emit_assign_auto=runtime.emit_assign_auto,
    )
    logger.info(
        "simulate.output: xes=%s summary=%s dataset_stats=%s data_config=%s",
        xes_path,
        summary_path,
        dataset_stats_path,
        data_cfg_path,
    )
    return {
        "status": "ok",
        "mode": "simulate-versioned-log",
        "process_name": process_name,
        "xes_path": str(xes_path),
        "summary_json_path": str(summary_path),
        "dataset_stats_json_path": str(dataset_stats_path),
        "generated_data_config_path": str(data_cfg_path),
        "summary": summary,
        "dataset_stats": dataset_stats,
        "generated_data_config": data_cfg_payload,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate versioned BPMN process logs and export XES.")
    parser.add_argument("--config", required=True, help="Path to simulator YAML config.")
    parser.add_argument("--out", default="", help="Optional path to JSON run summary.")
    parser.add_argument("--seed", default="", help="Optional random seed override.")
    parser.add_argument("--xes-out", default="", help="Optional XES path override.")
    parser.add_argument("--summary-out", default="", help="Optional summary path override.")
    parser.add_argument("--data-config-out", default="", help="Optional generated data-config path override.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_arg_parser().parse_args(argv)
    cfg_path = Path(str(args.config)).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_yaml_with_includes(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    simulation = cfg.setdefault("simulation", {})
    output = cfg.setdefault("output", {})
    if not isinstance(simulation, dict) or not isinstance(output, dict):
        raise ValueError("'simulation' and 'output' must be mappings.")

    if str(args.seed).strip():
        simulation["random_seed"] = _as_int(args.seed, 42)
    if str(args.xes_out).strip():
        output["xes_path"] = str(args.xes_out).strip()
    if str(args.summary_out).strip():
        output["summary_json_path"] = str(args.summary_out).strip()
    if str(args.data_config_out).strip():
        output["generated_data_config_path"] = str(args.data_config_out).strip()

    result = run(cfg, config_base_dir=cfg_path.parent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if str(args.out).strip():
        out_path = Path(str(args.out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
