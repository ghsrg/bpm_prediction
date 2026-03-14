from __future__ import annotations

from pathlib import Path
import shutil

from src.adapters.ingestion.camunda_trace_adapter import CamundaTraceAdapter


def _copy_mock_exports(target_dir: Path) -> None:
    source_dir = Path("data/camunda_exports")
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "mock_historic_activity_events.csv",
        "mock_historic_tasks.csv",
        "mock_identity_links.csv",
        "mock_execution_tree.csv",
        "mock_multi_instance_variables.csv",
        "mock_process_variables.csv",
        "mock_process_instance_links.csv",
    ):
        shutil.copy2(source_dir / file_name, target_dir / file_name)


def test_camunda_trace_adapter_reads_mock_files_and_builds_traces(tmp_path: Path):
    export_dir = tmp_path / "exports"
    _copy_mock_exports(export_dir)
    adapter = CamundaTraceAdapter()
    mapping = {
        "dataset_name": "procurement",
        "camunda_adapter": {
            "process_name": "procurement",
            "version_key": "v1",
            "runtime": {
                "runtime_source": "files",
                "export_dir": str(export_dir),
                "history_cleanup_aware": True,
                "legacy_removal_time_policy": "treat_as_eternal",
                "on_missing_removal_time": "auto_fallback",
            },
        },
    }

    traces = list(adapter.read(file_path="__unused__", mapping_config=mapping))
    assert len(traces) == 2
    assert all(len(trace.events) >= 2 for trace in traces)
    assert {trace.process_version for trace in traces} == {"v1"}
    assert "concept:name" in traces[0].events[0].extra
    assert "org:resource" in traces[0].events[0].extra
    assert "assigned_executor" in traces[0].events[0].extra
    assert "executed_by" in traces[0].events[0].extra
    assert "process_variables" in traces[0].events[0].extra


def test_camunda_trace_adapter_without_version_filter_reads_all_versions(tmp_path: Path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_content = (
        "process_name,version_key,case_id,proc_def_id,proc_def_key,proc_def_version,"
        "activity_def_id,activity_name,activity_type,act_inst_id,task_id,execution_id,parent_execution_id,"
        "call_proc_inst_id,start_time,end_time,duration_ms,assignee,candidate_groups\n"
        "procurement,v1,case_1,pd_1,procurement,v1,A,Task A,userTask,ai_1,,ex_1,,,"
        "2026-03-10T08:00:00,2026-03-10T08:00:10,10000,john,grp\n"
        "procurement,v1,case_1,pd_1,procurement,v1,B,Task B,userTask,ai_2,,ex_2,ex_1,,"
        "2026-03-10T08:01:00,2026-03-10T08:01:10,10000,john,grp\n"
        "procurement,v2,case_2,pd_2,procurement,v2,A,Task A,userTask,ai_3,,ex_3,,,"
        "2026-03-11T08:00:00,2026-03-11T08:00:10,10000,mary,grp\n"
        "procurement,v2,case_2,pd_2,procurement,v2,C,Task C,userTask,ai_4,,ex_4,ex_3,,"
        "2026-03-11T08:01:00,2026-03-11T08:01:10,10000,mary,grp\n"
    )
    (export_dir / "historic_activity_events.csv").write_text(csv_content, encoding="utf-8")

    adapter = CamundaTraceAdapter()
    mapping = {
        "dataset_name": "procurement",
        "camunda_adapter": {
            "process_name": "procurement",
            "version_key": "",
            "runtime": {
                "runtime_source": "files",
                "export_dir": str(export_dir),
            },
        },
    }

    traces = list(adapter.read(file_path="__unused__", mapping_config=mapping))
    assert len(traces) == 2
    assert {trace.process_version for trace in traces} == {"v1", "v2"}
