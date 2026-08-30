from __future__ import annotations

from pathlib import Path
import sys
import json
import subprocess

import pytest

from tools import run_cdlg_benchmark as runner


def test_load_run_plan_preserves_explicit_order_and_metadata(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        "runs:\n  - preset: _CDLG-medium2_GATv2-drift\n  - preset: _CDLG-simple1_EOPKG\n",
        encoding="utf-8",
    )
    presets = {
        "_CDLG-medium2_GATv2-drift": {"payload": {}},
        "_CDLG-simple1_EOPKG": {"payload": {}},
    }

    runs = runner.load_run_plan(plan_path, presets)

    assert [run.preset_name for run in runs] == ["_CDLG-medium2_GATv2-drift", "_CDLG-simple1_EOPKG"]
    assert runs[0].complexity == "medium"
    assert runs[0].case_index == 2
    assert runs[0].phase == "drift"
    assert runs[1].model_label == "EOPKG"
    assert runs[1].phase == "train"


def test_load_run_plan_rejects_unknown_preset(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("runs:\n  - preset: _CDLG-simple1_GATv2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        runner.load_run_plan(plan_path, {})


def test_compose_config_applies_ui_payload_and_gateway_mode(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text("experiment: {}\nmapping: {}\n", encoding="utf-8")
    payload = {
        "vars": {
            "config_path": str(base_path),
            "project": "cdlg_benchmark",
            "experiment_name": "_CDLG-simple1_GATv2-42",
            "mode": "train",
            "seed": "42",
            "gateway_mode": "collapse_for_prediction",
        },
        "input_data_form": {"data.dataset_label": "cdlg_simple1"},
        "model_form": {"model.model_label": "GATv2"},
        "features_text": "- name: concept:name\n",
        "policies_text": "profile: mvp1_default\n",
        "graph_mapping_text": "enabled: true\n",
    }

    cfg = runner.compose_preset_config(payload)

    assert cfg["experiment"]["name"] == "_CDLG-simple1_GATv2-42"
    assert cfg["data"]["dataset_label"] == "cdlg_simple1"
    assert cfg["mapping"]["graph_feature_mapping"]["topology_projection"]["gateway_mode"] == "collapse_for_prediction"
    assert cfg["mapping"]["features"] == [{"name": "concept:name"}]


def test_build_run_command_matches_ui_command_shapes(tmp_path: Path) -> None:
    cfg_path = tmp_path / "run.yaml"

    assert runner.build_run_command("train", cfg_path, extra_args="--limit 2") == [
        sys.executable,
        "main.py",
        "--config",
        str(cfg_path),
        "--limit",
        "2",
    ]
    assert runner.build_run_command("sync-stats", cfg_path, sync_as_of="2025-01-01T00:00:00Z") == [
        sys.executable,
        "main.py",
        "sync-stats",
        "--config",
        str(cfg_path),
        "--as-of",
        "2025-01-01T00:00:00Z",
    ]


def test_render_progress_includes_queue_and_preset_identity() -> None:
    run = runner.PlannedRun(
        preset_name="_CDLG-medium2_GATv2-drift",
        complexity="medium",
        case_index=2,
        model_label="GATv2",
        phase="drift",
        payload={},
    )
    tracker = runner.ProgressTracker(started_at=100.0)

    text = tracker.consume(
        run=run,
        queue_total=24,
        completed_count=6,
        event={
            "stage": "eval_drift.windows",
            "status": "update",
            "current": 18,
            "total": 50,
            "message": "Evaluating drift windows",
        },
        now=160.0,
        queue_eta_seconds=6450.0,
    )

    assert "Run 07/24" in text
    assert "_CDLG-medium2_GATv2-drift" in text
    assert "completed 6 | remaining 17" in text
    assert "Оцінка drift-вікон" in text
    assert "18/50 (36.0%)" in text
    assert "stage ETA" in text
    assert "run ETA" in text
    assert "Queue" in text


def test_queue_eta_uses_completed_duration_average_by_phase() -> None:
    estimator = runner.QueueEtaEstimator()
    estimator.record("train", 120.0)
    estimator.record("drift", 30.0)

    assert estimator.estimate(["train", "drift", "drift"]) == 180.0


class _FakeProcess:
    def __init__(self, lines: list[str], returncode: int) -> None:
        self.stdout = lines
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def test_execute_queue_stops_and_blocks_paired_drift_after_failed_train(tmp_path: Path, capsys) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text("experiment: {}\nmapping: {}\n", encoding="utf-8")
    payload = {"vars": {"config_path": str(base_path), "mode": "train"}}
    train = runner.PlannedRun("_CDLG-simple1_GATv2", "simple", 1, "GATv2", "train", payload)
    drift = runner.PlannedRun("_CDLG-simple1_GATv2-drift", "simple", 1, "GATv2", "drift", payload)
    later = runner.PlannedRun("_CDLG-simple1_EOPKG", "simple", 1, "EOPKG", "train", payload)
    launches: list[list[str]] = []

    def process_factory(command, **_kwargs):
        launches.append(command)
        return _FakeProcess(["__BPM_PROGRESS__{\"stage\":\"train.epochs\",\"current\":1,\"total\":2}\n"], 7)

    result = runner.execute_queue([train, drift, later], output_dir=tmp_path, process_factory=process_factory)

    assert result.exit_code == 7
    assert len(launches) == 1
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in manifest] == ["failed", "blocked"]
    assert "Run 01/03" in capsys.readouterr().out


def test_cli_script_bootstraps_repository_import_path() -> None:
    script = Path("tools/run_cdlg_benchmark.py").resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Run 01/" in completed.stdout
    assert completed.stderr == ""
    assert "ModuleNotFoundError: No module named 'src'" not in completed.stderr
