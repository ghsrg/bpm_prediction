from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tools import sync_stats_backfill as tool


def test_iter_as_of_points_weekly_includes_end():
    start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc)
    points = tool._iter_as_of_points(start, end, tool.BackfillStep(mode="days", days=7))
    assert [item.isoformat() for item in points] == [
        "2026-03-01T00:00:00+00:00",
        "2026-03-08T00:00:00+00:00",
        "2026-03-10T00:00:00+00:00",
    ]


def test_sync_stats_backfill_main_runs_sync_stats_for_each_point(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("experiment:\n  mode: train\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        tool,
        "load_yaml_config",
        lambda config_path: {"mapping": {"adapter": "camunda"}, "sync_stats": {}},
    )
    monkeypatch.setattr(
        tool,
        "_discover_bounds",
        lambda config: (
            ["proc_a"],
            datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc),
            "camunda",
        ),
    )

    calls: list[list[str]] = []

    def _fake_sync_stats_main(argv):
        calls.append(list(argv))
        out_path = Path(argv[3])
        as_of_ts = str(argv[5])
        if as_of_ts == "2026-03-01T00:00:00Z":
            payload = {
                "status": "ok",
                "processed_versions": 2,
                "skipped_versions": 1,
                "details": [
                    {
                        "process_namespace": "proc_a",
                        "version": "v1",
                        "is_usable_for_training": True,
                        "quality_reason": "ok",
                        "quality_failures": [],
                        "alignment_is_ok": True,
                        "alignment_reason": "ok",
                        "alignment_failures": [],
                        "alignment_event_match_ratio": 1.0,
                        "alignment_unique_activity_coverage": 1.0,
                        "alignment_node_coverage": 0.75,
                    },
                    {
                        "process_namespace": "proc_a",
                        "version": "v2",
                        "is_usable_for_training": False,
                        "quality_reason": "zero_dominant",
                        "quality_failures": ["zero_dominant"],
                        "alignment_is_ok": False,
                        "alignment_reason": "below_min_node_coverage",
                        "alignment_failures": ["below_min_node_coverage"],
                        "alignment_event_match_ratio": 1.0,
                        "alignment_unique_activity_coverage": 1.0,
                        "alignment_node_coverage": 0.25,
                    },
                ],
                "skipped_details": [
                    {
                        "process_name": "proc_a",
                        "version": "v3",
                        "reason": "no_scope_events_up_to_as_of",
                    }
                ],
            }
        else:
            payload = {
                "status": "ok",
                "processed_versions": 1,
                "skipped_versions": 0,
                "details": [
                    {
                        "process_namespace": "proc_a",
                        "version": "v1",
                        "is_usable_for_training": False,
                        "quality_reason": "zero_dominant",
                        "quality_failures": ["zero_dominant"],
                        "alignment_is_ok": True,
                        "alignment_reason": "ok",
                        "alignment_failures": [],
                        "alignment_event_match_ratio": 0.9,
                        "alignment_unique_activity_coverage": 1.0,
                        "alignment_node_coverage": 0.5,
                    }
                ],
                "skipped_details": [],
            }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    monkeypatch.setattr(tool, "sync_stats_main", _fake_sync_stats_main)

    rc = tool.main(
        [
            "--config",
            str(cfg_path),
            "--out-dir",
            str(out_dir),
            "--step",
            "weekly",
        ]
    )
    assert rc == 0
    assert len(calls) == 3
    assert calls[0][0:2] == ["--config", str(cfg_path)]
    assert calls[0][2] == "--out"
    assert calls[0][4] == "--as-of"
    assert calls[0][5] == "2026-03-01T00:00:00Z"
    assert calls[1][5] == "2026-03-08T00:00:00Z"
    assert calls[2][5] == "2026-03-15T00:00:00Z"

    summary = json.loads((out_dir / "backfill_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["mode"] == "sync-stats-backfill"
    assert summary["runs_total"] == 3
    assert summary["runs_completed"] == 3
    assert summary["selected_processes"] == ["proc_a"]
    aggregate = summary["aggregate"]
    assert aggregate["runs"] == {"total": 3, "ok": 3, "failed": 0, "planned": 0}
    assert aggregate["versions"]["processed"] == 4
    assert aggregate["versions"]["skipped"] == 1
    assert aggregate["versions"]["usable_for_training"] == 1
    assert aggregate["versions"]["not_usable_for_training"] == 3
    assert aggregate["quality"]["reasons"] == {"ok": 1, "zero_dominant": 3}
    assert aggregate["alignment"]["ok"] == 3
    assert aggregate["alignment"]["failed"] == 1
    assert aggregate["alignment"]["min_event_match_ratio"] == 0.9
    assert aggregate["alignment"]["min_unique_activity_coverage"] == 1.0
    assert aggregate["alignment"]["min_node_coverage"] == 0.25
    assert aggregate["alignment"]["failed_reasons"] == {"below_min_node_coverage": 1}
    assert aggregate["skips"]["reasons"] == {"no_scope_events_up_to_as_of": 1}
    assert aggregate["by_process_version"]["proc_a::v1"]["runs_seen"] == 3
    assert aggregate["by_process_version"]["proc_a::v1"]["alignment_failed"] == 0
    assert aggregate["by_process_version"]["proc_a::v2"]["alignment_failed"] == 1
