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
