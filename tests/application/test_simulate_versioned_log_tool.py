from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.adapters.ingestion.xes_adapter import XESAdapter


pytest.importorskip("lxml")
pytest.importorskip("simpy")
from tools.simulate_versioned_log import (
    CaseCtx,
    EdgeDef,
    ExecGraph,
    NodeDef,
    Runtime,
    VersionSpec,
    _build_dataset_stats,
    _parse_dt,
    run,
)


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _simple_bpmn_user_task(task_id: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Examples">
  <process id="loan_process" isExecutable="true">
    <startEvent id="start_evt" />
    <userTask id="{task_id}" name="Check Application" />
    <endEvent id="end_evt" />
    <sequenceFlow id="f1" sourceRef="start_evt" targetRef="{task_id}" />
    <sequenceFlow id="f2" sourceRef="{task_id}" targetRef="end_evt" />
  </process>
</definitions>
"""


def _simple_bpmn_service_task(task_id: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Examples">
  <process id="loan_process" isExecutable="true">
    <startEvent id="start_evt" />
    <serviceTask id="{task_id}" name="Auto Decision" />
    <endEvent id="end_evt" />
    <sequenceFlow id="f1" sourceRef="start_evt" targetRef="{task_id}" />
    <sequenceFlow id="f2" sourceRef="{task_id}" targetRef="end_evt" />
  </process>
</definitions>
"""


def _xor_loop_bpmn() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Examples">
  <process id="loan_process" isExecutable="true">
    <startEvent id="start_evt" />
    <serviceTask id="work" name="Work" />
    <exclusiveGateway id="gw_retry" />
    <serviceTask id="retry" name="Retry" />
    <endEvent id="end_evt" />
    <sequenceFlow id="f1" sourceRef="start_evt" targetRef="work" />
    <sequenceFlow id="f2" sourceRef="work" targetRef="gw_retry" />
    <sequenceFlow id="f_loop" sourceRef="gw_retry" targetRef="retry" />
    <sequenceFlow id="f3" sourceRef="retry" targetRef="work" />
    <sequenceFlow id="f_exit" sourceRef="gw_retry" targetRef="end_evt" />
  </process>
</definitions>
"""


def test_simulate_versioned_log_generates_xes_summary_and_data_config(tmp_path: Path):
    bpmn_v1 = _write_file(tmp_path / "loan_v1.bpmn", _simple_bpmn_user_task("check_application"))
    bpmn_v2 = _write_file(tmp_path / "loan_v2.bpmn", _simple_bpmn_service_task("auto_decision"))
    xes_path = tmp_path / "out" / "sim.xes"
    summary_path = tmp_path / "out" / "sim.summary.json"
    stats_path = tmp_path / "out" / "sim.dataset_stats.json"
    data_cfg_path = tmp_path / "data" / "generated_sim.yaml"

    cfg = {
        "simulation": {
            "process_name": "loan_demo_sim",
            "random_seed": 7,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T03:00:00Z",
        },
        "versions": [
            {"version_id": "v1", "active_from": "2025-01-01T00:00:00Z", "bpmn_path": str(bpmn_v1)},
            {"version_id": "v2", "active_from": "2025-01-01T01:30:00Z", "bpmn_path": str(bpmn_v2)},
        ],
        "arrival_process": {"type": "poisson", "rate_per_hour": 8.0, "max_cases": 40},
        "resources": {"roles": {"clerk": {"workers": [{"id": "clerk_1", "factor": 1.0}]}}},
        "tasks": {
            "check_application": {
                "execution_mode": "human",
                "roles": ["clerk"],
                "duration": {"type": "fixed", "seconds": 30},
            },
            "auto_decision": {
                "execution_mode": "automatic",
                "duration": {"type": "fixed", "seconds": 2},
            },
        },
        "output": {
            "xes_path": str(xes_path),
            "summary_json_path": str(summary_path),
            "dataset_stats_json_path": str(stats_path),
            "generated_data_config_path": str(data_cfg_path),
            "overwrite": True,
            "emit_assign_for_human_tasks": True,
            "emit_assign_for_automatic_tasks": False,
        },
    }

    result = run(cfg, config_base_dir=tmp_path)
    assert result["status"] == "ok"
    assert xes_path.exists()
    assert summary_path.exists()
    assert stats_path.exists()
    assert data_cfg_path.exists()

    summary = result["summary"]
    assert summary["case_count_total"] > 0
    assert summary["case_count_total"] <= 40
    assert "arrival_stats" in summary
    assert summary["arrival_stats"]["type"] == "poisson"
    assert result["dataset_stats_json_path"] == str(stats_path)

    dataset_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert dataset_stats["total"]["trace_count"] == summary["case_count_total"]
    assert set(dataset_stats["by_version"]) <= {"v1", "v2"}
    assert dataset_stats["total"]["node_coverage"]["task_nodes_total"] == 2
    assert dataset_stats["total"]["node_coverage"]["task_nodes_missing_count"] == 0
    assert dataset_stats["total"]["resources"]["unique_resource_count"] >= 1
    assert "version_carryover" in dataset_stats["total"]

    data_cfg = yaml.safe_load(data_cfg_path.read_text(encoding="utf-8"))
    assert data_cfg["mapping"]["xes_adapter"]["start_transitions"] == ["assign", "start"]

    adapter = XESAdapter()
    traces = list(adapter.read(str(xes_path), mapping_config=data_cfg["mapping"]))
    assert len(traces) == summary["case_count_total"]


def test_simulate_versioned_log_blocks_unsupported_bpmn_elements(tmp_path: Path):
    unsupported_bpmn = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" targetNamespace="Examples">
  <process id="loan_process" isExecutable="true">
    <startEvent id="start_evt" />
    <inclusiveGateway id="gw_inc" />
    <endEvent id="end_evt" />
    <sequenceFlow id="f1" sourceRef="start_evt" targetRef="gw_inc" />
    <sequenceFlow id="f2" sourceRef="gw_inc" targetRef="end_evt" />
  </process>
</definitions>
"""
    bpmn_path = _write_file(tmp_path / "loan_bad.bpmn", unsupported_bpmn)

    cfg = {
        "simulation": {
            "process_name": "loan_demo_sim",
            "random_seed": 1,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T01:00:00Z",
        },
        "versions": [
            {"version_id": "v1", "active_from": "2025-01-01T00:00:00Z", "bpmn_path": str(bpmn_path)}
        ],
        "arrival_process": {"type": "poisson", "rate_per_hour": 2.0, "max_cases": 5},
        "output": {
            "xes_path": str(tmp_path / "out.xes"),
            "summary_json_path": str(tmp_path / "out.summary.json"),
            "generated_data_config_path": str(tmp_path / "generated.yaml"),
            "overwrite": True,
        },
    }

    with pytest.raises(ValueError, match="inclusiveGateway"):
        run(cfg, config_base_dir=tmp_path)


def test_xor_branch_can_be_bounded_per_case_for_loops():
    runtime = Runtime.__new__(Runtime)
    runtime.gateway_branch_counts = Counter()
    runtime.gateways_cfg = {
        "gw_retry": {
            "default_flow_id": "f_exit",
            "branches": [
                {
                    "flow_id": "f_loop",
                    "probability": 1.0,
                    "max_traversals_per_case": 2,
                    "repeat_until_max_once_selected": True,
                    "when": {"const": True},
                },
                {"flow_id": "f_exit", "when": {"const": True}},
            ],
        }
    }
    case = CaseCtx(
        case_id="case_1",
        case_index=1,
        version_id="v1",
        start_dt=_parse_dt("2025-01-01T00:00:00Z"),
        attrs={},
        graph=None,  # type: ignore[arg-type]
        rng=random.Random(1),
        env=None,
    )
    outgoing = [
        EdgeDef(edge_id="f_loop", source="gw_retry", target="retry", is_default=False),
        EdgeDef(edge_id="f_exit", source="gw_retry", target="end_evt", is_default=True),
    ]

    assert runtime._choose_xor_edge(case, "gw_retry", outgoing).edge_id == "f_loop"
    assert runtime._choose_xor_edge(case, "gw_retry", outgoing).edge_id == "f_loop"
    assert runtime._choose_xor_edge(case, "gw_retry", outgoing).edge_id == "f_exit"


def test_dataset_stats_counts_four_version_carryover_delta():
    versions = [
        VersionSpec(f"v{i}", _parse_dt(f"2025-0{i}-01T00:00:00Z"), Path(f"v{i}.bpmn"), "")
        for i in range(1, 6)
    ]
    graph = ExecGraph(
        version_id="v1",
        nodes={"task_a": NodeDef("task_a", "serviceTask", "Task A", "task")},
        outgoing={},
        incoming={},
        start_nodes=[],
        end_nodes=[],
    )
    case = CaseCtx(
        case_id="case_1",
        case_index=1,
        version_id="v1",
        start_dt=_parse_dt("2025-01-01T00:00:00Z"),
        attrs={},
        graph=graph,
        rng=random.Random(1),
        env=None,
    )
    case.completion_dt = _parse_dt("2025-05-02T00:00:00Z")

    stats = _build_dataset_stats(cases=[case], versions=versions, graphs={"v1": graph})

    assert stats["total"]["version_carryover"]["trace_count_by_completion_delta"]["plus_4"] == 1
    assert stats["total"]["version_carryover"]["trace_count_by_completion_delta"]["plus_4_or_more"] == 0


def test_dataset_stats_counts_calendar_month_carryover():
    versions = [
        VersionSpec("v1", _parse_dt("2025-01-01T00:00:00Z"), Path("v1.bpmn"), ""),
    ]
    graph = ExecGraph(
        version_id="v1",
        nodes={"task_a": NodeDef("task_a", "serviceTask", "Task A", "task")},
        outgoing={},
        incoming={},
        start_nodes=[],
        end_nodes=[],
    )
    cases = []
    for i, completion in enumerate(
        [
            "2025-01-20T00:00:00Z",
            "2025-02-05T00:00:00Z",
            "2025-03-05T00:00:00Z",
            "2025-04-05T00:00:00Z",
            "2025-05-05T00:00:00Z",
            "2025-08-05T00:00:00Z",
        ],
        start=1,
    ):
        case = CaseCtx(
            case_id=f"case_{i}",
            case_index=i,
            version_id="v1",
            start_dt=_parse_dt("2025-01-15T00:00:00Z"),
            attrs={},
            graph=graph,
            rng=random.Random(i),
            env=None,
        )
        case.completion_dt = _parse_dt(completion)
        cases.append(case)

    stats = _build_dataset_stats(cases=cases, versions=versions, graphs={"v1": graph})

    counts = stats["total"]["calendar_carryover"]["trace_count_by_completion_month_delta"]
    assert counts["same_month"] == 1
    assert counts["plus_1month"] == 1
    assert counts["plus_2month"] == 1
    assert counts["plus_3month"] == 1
    assert counts["plus_4month"] == 1
    assert counts["plus_4month_or_more"] == 1


def test_task_duration_applies_conditional_delay_from_case_attrs():
    runtime = Runtime.__new__(Runtime)
    runtime.tasks_cfg = {
        "review": {
            "execution_mode": "human",
            "duration": {"type": "fixed", "seconds": 10},
            "conditional_delays": [
                {
                    "when": {"var": "delay_class", "op": "==", "value": "black_mark"},
                    "probability": 1.0,
                    "duration": {"type": "fixed", "seconds": 30},
                }
            ],
        }
    }
    case = CaseCtx(
        case_id="case_1",
        case_index=1,
        version_id="v1",
        start_dt=_parse_dt("2025-01-01T00:00:00Z"),
        attrs={"delay_class": "black_mark"},
        graph=None,  # type: ignore[arg-type]
        rng=random.Random(1),
        env=None,
    )

    assert runtime._task_duration(case, "review", "human", None) == 40


def test_task_wait_delay_applies_without_changing_busy_duration():
    runtime = Runtime.__new__(Runtime)
    runtime.tasks_cfg = {
        "review": {
            "execution_mode": "human",
            "duration": {"type": "fixed", "seconds": 10},
            "conditional_waits": [
                {
                    "when": {"var": "delay_class", "op": "==", "value": "black_mark"},
                    "probability": 1.0,
                    "duration": {"type": "fixed", "seconds": 30},
                }
            ],
        }
    }
    case = CaseCtx(
        case_id="case_1",
        case_index=1,
        version_id="v1",
        start_dt=_parse_dt("2025-01-01T00:00:00Z"),
        attrs={"delay_class": "black_mark"},
        graph=None,  # type: ignore[arg-type]
        rng=random.Random(1),
        env=None,
    )

    assert runtime._task_wait_delay(case, "review") == 30
    assert runtime._task_duration(case, "review", "human", None) == 10


def test_version_carryover_delays_terminal_task_timestamp(tmp_path: Path):
    bpmn_v1 = _write_file(tmp_path / "loan_v1.bpmn", _simple_bpmn_service_task("auto_decision"))
    bpmn_v2 = _write_file(tmp_path / "loan_v2.bpmn", _simple_bpmn_service_task("auto_decision"))
    xes_path = tmp_path / "out" / "sim.xes"
    summary_path = tmp_path / "out" / "sim.summary.json"
    stats_path = tmp_path / "out" / "sim.dataset_stats.json"

    cfg = {
        "simulation": {
            "process_name": "loan_demo_sim",
            "random_seed": 11,
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T01:00:00Z",
        },
        "versions": [
            {"version_id": "v1", "active_from": "2025-01-01T00:00:00Z", "bpmn_path": str(bpmn_v1)},
            {"version_id": "v2", "active_from": "2025-01-01T01:30:00Z", "bpmn_path": str(bpmn_v2)},
        ],
        "arrival_process": {"type": "poisson", "rate_per_hour": 4.0, "max_cases": 6},
        "version_carryover": {
            "enabled": True,
            "targets": [{"completion": "next_version", "probability": 1.0}],
            "jitter_seconds": {"type": "fixed", "seconds": 0},
        },
        "tasks": {
            "auto_decision": {
                "execution_mode": "automatic",
                "duration": {"type": "fixed", "seconds": 2},
            },
        },
        "output": {
            "xes_path": str(xes_path),
            "summary_json_path": str(summary_path),
            "dataset_stats_json_path": str(stats_path),
            "generated_data_config_path": str(tmp_path / "generated.yaml"),
            "overwrite": True,
            "emit_assign_for_automatic_tasks": False,
        },
    }

    result = run(cfg, config_base_dir=tmp_path)
    assert result["summary"]["case_count_by_version"] == {"v1": result["summary"]["case_count_total"]}
    assert result["dataset_stats"]["total"]["version_carryover"]["trace_count_by_completion_delta"]["plus_1"] > 0
    for case in result["dataset_stats"]["total"]["version_carryover"]["trace_percent_by_completion_delta"]:
        assert case in {"none", "plus_1", "plus_2", "plus_3", "plus_4", "plus_4_or_more"}

    adapter = XESAdapter()
    data_cfg = yaml.safe_load((tmp_path / "generated.yaml").read_text(encoding="utf-8"))
    traces = list(adapter.read(str(xes_path), mapping_config=data_cfg["mapping"]))
    assert traces
    threshold = _parse_dt("2025-01-01T01:30:00Z").timestamp()
    for trace in traces:
        assert max(event.timestamp for event in trace.events) >= threshold
