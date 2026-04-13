from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.adapters.ingestion.xes_adapter import XESAdapter


pytest.importorskip("lxml")
pytest.importorskip("simpy")
from tools.simulate_versioned_log import run


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


def test_simulate_versioned_log_generates_xes_summary_and_data_config(tmp_path: Path):
    bpmn_v1 = _write_file(tmp_path / "loan_v1.bpmn", _simple_bpmn_user_task("check_application"))
    bpmn_v2 = _write_file(tmp_path / "loan_v2.bpmn", _simple_bpmn_service_task("auto_decision"))
    xes_path = tmp_path / "out" / "sim.xes"
    summary_path = tmp_path / "out" / "sim.summary.json"
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
    assert data_cfg_path.exists()

    summary = result["summary"]
    assert summary["case_count_total"] > 0
    assert summary["case_count_total"] <= 40
    assert "arrival_stats" in summary
    assert summary["arrival_stats"]["type"] == "poisson"

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
