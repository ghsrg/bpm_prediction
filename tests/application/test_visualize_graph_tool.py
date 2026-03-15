from __future__ import annotations

from datetime import datetime, timedelta

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.runtime_fetch_diagnostics import RuntimeFetchDiagnosticsDTO
from tools import visualize_graph


def _event(case_id: str, idx: int, activity: str, call_proc_inst_id: str = "") -> ProcessEventDTO:
    start = datetime.fromisoformat("2026-03-10T08:00:00") + timedelta(minutes=idx)
    end = start + timedelta(seconds=30)
    return ProcessEventDTO(
        case_id=case_id,
        activity_def_id=activity,
        activity_name=activity,
        activity_type="userTask",
        task_id=f"task_{case_id}_{idx}",
        act_inst_id=f"ai_{case_id}_{idx}",
        execution_id=f"ex_{case_id}_{idx}",
        start_time=start,
        end_time=end,
        duration_ms=30000.0,
        call_proc_inst_id=call_proc_inst_id or None,
    )


def _camunda_cfg() -> dict:
    return {
        "mapping": {
            "adapter": "camunda",
            "camunda_adapter": {
                "process_name": "procurement",
                "version_key": "v1",
                "runtime": {"runtime_source": "files", "export_dir": "data/camunda_exports"},
            },
        }
    }


def test_visualize_graph_selects_explicit_case_id(monkeypatch, tmp_path):
    payload = {
        "process_name": "procurement",
        "version_key": "v1",
        "camunda_cfg": {},
        "events": [
            _event("case_1", 0, "A"),
            _event("case_1", 1, "B"),
            _event("case_2", 0, "A"),
            _event("case_2", 1, "C", call_proc_inst_id="child_9"),
        ],
        "execution_rows": [],
        "variables_rows": [],
        "identity_rows": [],
        "task_rows": [],
        "process_variables_rows": [],
        "diagnostics": RuntimeFetchDiagnosticsDTO(),
    }

    monkeypatch.setattr(visualize_graph, "load_yaml_config", lambda _p: _camunda_cfg())
    monkeypatch.setattr(visualize_graph, "_load_camunda_payload_from_config", lambda _p: payload)

    rendered: dict = {}

    def _fake_render(graph, *, out_path, title, hide_loop_back=False):
        rendered["graph"] = graph
        rendered["out_path"] = out_path
        rendered["title"] = title
        rendered["hide_loop_back"] = hide_loop_back

    monkeypatch.setattr(visualize_graph, "_render_ig_graph", _fake_render)

    rc = visualize_graph.main(
        [
            "--config",
            "dummy.yaml",
            "--case-id",
            "case_2",
            "--out",
            str(tmp_path / "ig_case2.png"),
        ]
    )

    assert rc == 0
    assert rendered["graph"]["metadata"]["selected_case_id"] == "case_2"
    assert all(node.get("case_id") == "case_2" for node in rendered["graph"]["nodes"])


def test_visualize_graph_pick_longest(monkeypatch, tmp_path):
    payload = {
        "process_name": "procurement",
        "version_key": "v1",
        "camunda_cfg": {},
        "events": [
            _event("short_case", 0, "A"),
            _event("short_case", 1, "B"),
            _event("long_case", 0, "A"),
            _event("long_case", 1, "B"),
            _event("long_case", 2, "C"),
            _event("long_case", 3, "D"),
        ],
        "execution_rows": [],
        "variables_rows": [],
        "identity_rows": [],
        "task_rows": [],
        "process_variables_rows": [],
        "diagnostics": RuntimeFetchDiagnosticsDTO(),
    }

    monkeypatch.setattr(visualize_graph, "load_yaml_config", lambda _p: _camunda_cfg())
    monkeypatch.setattr(visualize_graph, "_load_camunda_payload_from_config", lambda _p: payload)

    rendered: dict = {}

    def _fake_render(graph, *, out_path, title, hide_loop_back=False):
        rendered["graph"] = graph
        rendered["out_path"] = out_path
        rendered["title"] = title
        rendered["hide_loop_back"] = hide_loop_back

    monkeypatch.setattr(visualize_graph, "_render_ig_graph", _fake_render)

    rc = visualize_graph.main(
        [
            "--config",
            "dummy.yaml",
            "--pick",
            "longest",
            "--out",
            str(tmp_path / "ig_longest.png"),
        ]
    )

    assert rc == 0
    assert rendered["graph"]["metadata"]["selected_case_id"] == "long_case"
    assert len(rendered["graph"]["nodes"]) == 4


def test_visualize_graph_prints_selected_case_when_picked(monkeypatch, tmp_path, capsys):
    payload = {
        "process_name": "procurement",
        "version_key": "v1",
        "camunda_cfg": {},
        "events": [
            _event("short_case", 0, "A"),
            _event("short_case", 1, "B"),
            _event("long_case", 0, "A"),
            _event("long_case", 1, "B"),
            _event("long_case", 2, "C"),
        ],
        "execution_rows": [],
        "variables_rows": [],
        "identity_rows": [],
        "task_rows": [],
        "process_variables_rows": [],
        "diagnostics": RuntimeFetchDiagnosticsDTO(),
    }

    monkeypatch.setattr(visualize_graph, "load_yaml_config", lambda _p: _camunda_cfg())
    monkeypatch.setattr(visualize_graph, "_load_camunda_payload_from_config", lambda _p: payload)
    monkeypatch.setattr(visualize_graph, "_render_ig_graph", lambda graph, *, out_path, title, hide_loop_back=False: None)

    rc = visualize_graph.main(
        [
            "--config",
            "dummy.yaml",
            "--pick",
            "longest",
            "--out",
            str(tmp_path / "ig_longest_print.png"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Selected case_id by strategy: long_case" in out


def test_visualize_graph_list_cases_only_does_not_render(monkeypatch, capsys):
    payload = {
        "process_name": "procurement",
        "version_key": "v1",
        "camunda_cfg": {},
        "events": [
            _event("case_1", 0, "A"),
            _event("case_1", 1, "B"),
            _event("case_2", 0, "A"),
            _event("case_2", 1, "C", call_proc_inst_id="child_9"),
        ],
        "execution_rows": [],
        "variables_rows": [],
        "identity_rows": [],
        "task_rows": [],
        "process_variables_rows": [],
        "diagnostics": RuntimeFetchDiagnosticsDTO(),
    }

    monkeypatch.setattr(visualize_graph, "load_yaml_config", lambda _p: _camunda_cfg())
    monkeypatch.setattr(visualize_graph, "_load_camunda_payload_from_config", lambda _p: payload)

    called = {"render": False}

    def _fake_render(graph, *, out_path, title, hide_loop_back=False):
        _ = (graph, out_path, title)
        called["render"] = True

    monkeypatch.setattr(visualize_graph, "_render_ig_graph", _fake_render)

    rc = visualize_graph.main(["--config", "dummy.yaml", "--list-cases", "--top", "5"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Found 2 process instances" in out
    assert called["render"] is False


def test_visualize_graph_ignores_nullish_call_activity_links():
    payload = {
        "process_name": "procurement",
        "version_key": "v1",
        "camunda_cfg": {},
        "events": [
            _event("case_1", 0, "A"),
            _event("case_1", 1, "B"),
        ],
        "execution_rows": [],
        "variables_rows": [],
        "identity_rows": [],
        "task_rows": [],
        "process_variables_rows": [],
        "diagnostics": RuntimeFetchDiagnosticsDTO(),
    }
    payload["events"][1].call_proc_inst_id = "nan"

    graph = visualize_graph._build_camunda_case_graph(
        payload,
        case_id="case_1",
        mode="activity-centric",
        max_nodes=500,
    )

    links = graph.get("metadata", {}).get("call_activity_links", [])
    assert links == []


def test_format_node_label_includes_activity_name_and_id():
    label = visualize_graph._format_node_label(
        activity_def_id="Task_Approve",
        activity_name="Approve request",
        activity_type="userTask",
        fallback_id="ai_1",
    )

    assert "Approve request" in label
    assert "id: Task_Approve" in label
    assert "[userTask]" in label


def test_format_node_label_includes_occurrence_suffix():
    label = visualize_graph._format_node_label(
        activity_def_id="Activity_1onmiy7",
        activity_name="Review Task",
        activity_type="userTask",
        fallback_id="ai_7",
        occurrence_suffix="#2",
    )
    assert "Review Task #2" in label


def test_visualize_graph_forwards_hide_loop_back_flag(monkeypatch, tmp_path):
    payload = {
        "process_name": "procurement",
        "version_key": "v1",
        "camunda_cfg": {},
        "events": [
            _event("case_1", 0, "A"),
            _event("case_1", 1, "A"),
        ],
        "execution_rows": [],
        "variables_rows": [],
        "identity_rows": [],
        "task_rows": [],
        "process_variables_rows": [],
        "diagnostics": RuntimeFetchDiagnosticsDTO(),
    }

    monkeypatch.setattr(visualize_graph, "load_yaml_config", lambda _p: _camunda_cfg())
    monkeypatch.setattr(visualize_graph, "_load_camunda_payload_from_config", lambda _p: payload)

    captured = {"hide_loop_back": None}

    def _fake_render(graph, *, out_path, title, hide_loop_back=False):
        _ = (graph, out_path, title)
        captured["hide_loop_back"] = hide_loop_back

    monkeypatch.setattr(visualize_graph, "_render_ig_graph", _fake_render)

    rc = visualize_graph.main(
        [
            "--config",
            "dummy.yaml",
            "--case-id",
            "case_1",
            "--hide-loop-back",
            "--out",
            str(tmp_path / "ig_case1.png"),
        ]
    )
    assert rc == 0
    assert captured["hide_loop_back"] is True
