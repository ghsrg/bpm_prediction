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
        noise = cfg.get("noise", {}) if isinstance(cfg.get("noise", {}), dict) else {}
        p = max(0.0, min(1.0, _as_float(noise.get("extra_delay_probability", 0.0), 0.0)))
        if p > 0 and case.rng.random() < p:
            base += max(0.0, _as_float(noise.get("extra_delay_seconds", 0.0), 0.0))
        return max(0.001, base)

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
            when = branch.get("when", {}) if isinstance(branch.get("when", {}), dict) else {}
            if _eval_rule(when, case.attrs):
                self.gateway_branch_counts[f"{node_id}:{flow_id}"] += 1
                return flow_map[flow_id]
        default_flow_id = str(gw.get("default_flow_id", "")).strip()
        if default_flow_id and default_flow_id in flow_map:
            self.gateway_branch_counts[f"{node_id}:{default_flow_id}"] += 1
            return flow_map[default_flow_id]
        defaults = [e for e in outgoing if e.is_default]
        if defaults:
            edge = sorted(defaults, key=lambda x: x.edge_id)[0]
            self.gateway_branch_counts[f"{node_id}:{edge.edge_id}"] += 1
            return edge
        edge = sorted(outgoing, key=lambda x: x.edge_id)[0]
        self.gateway_branch_counts[f"{node_id}:{edge.edge_id}"] += 1
        return edge

    def _exec_task(self, case: CaseCtx, node: NodeDef) -> Iterable[Any]:
        mode = self._task_mode(node)
        inst_id = case.next_instance_id(node.node_id)
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
        case = CaseCtx(
            case_id=f"case_{case_index:06d}",
            case_index=case_index,
            version_id=version.version_id,
            start_dt=start_dt,
            attrs=self._case_attrs(rng),
            graph=self.graphs[version.version_id],
            rng=rng,
            env=self.env,
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
    data_cfg_path = _resolve_path(base_dir, str(output_cfg.get("generated_data_config_path", "configs/data/generated_simulated.yaml")))
    overwrite = _as_bool(output_cfg.get("overwrite"), True)
    if not overwrite:
        for p in (xes_path, summary_path, data_cfg_path):
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
    data_cfg_payload = _write_generated_data_config(
        data_cfg_path,
        xes_path=xes_path,
        process_name=process_name,
        emit_assign_human=runtime.emit_assign_human,
        emit_assign_auto=runtime.emit_assign_auto,
    )
    logger.info(
        "simulate.output: xes=%s summary=%s data_config=%s",
        xes_path,
        summary_path,
        data_cfg_path,
    )
    return {
        "status": "ok",
        "mode": "simulate-versioned-log",
        "process_name": process_name,
        "xes_path": str(xes_path),
        "summary_json_path": str(summary_path),
        "generated_data_config_path": str(data_cfg_path),
        "summary": summary,
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
