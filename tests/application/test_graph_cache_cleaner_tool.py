from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from tools import graph_cache_cleaner as tool


def _write_cache_entry(root: Path, dataset: str, fingerprint: str, *, days_ago: int) -> Path:
    entry = root / dataset / fingerprint
    entry.mkdir(parents=True, exist_ok=True)
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    meta = {
        "schema": 1,
        "fingerprint": fingerprint,
        "dataset_name": dataset,
        "created_utc": ts,
        "last_access_utc": ts,
        "counts": {"train_graphs": 1, "validation_graphs": 1, "test_graphs": 1},
    }
    (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    torch.save([{"k": "v"}], entry / "train.pt")
    torch.save([{"k": "v"}], entry / "validation.pt")
    torch.save([{"k": "v"}], entry / "test.pt")
    return entry


def test_graph_cache_cleaner_deletes_only_old_entries(tmp_path: Path):
    cache_dir = tmp_path / "graph_cache"
    old_entry = _write_cache_entry(cache_dir, "bpi2012", "old01", days_ago=90)
    new_entry = _write_cache_entry(cache_dir, "bpi2012", "new01", days_ago=1)
    summary_path = tmp_path / "summary.json"

    rc = tool.main(
        [
            "--cache-dir",
            str(cache_dir),
            "--older-than-days",
            "30",
            "--summary-out",
            str(summary_path),
        ]
    )
    assert rc == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["entries_deleted"] == 1
    assert str(old_entry) in [row["path"] for row in summary["deleted"]]
    assert not old_entry.exists()
    assert new_entry.exists()


def test_graph_cache_cleaner_dry_run_keeps_files(tmp_path: Path):
    cache_dir = tmp_path / "graph_cache"
    old_entry = _write_cache_entry(cache_dir, "bpi2012", "old01", days_ago=90)
    summary_path = tmp_path / "summary.json"

    rc = tool.main(
        [
            "--cache-dir",
            str(cache_dir),
            "--older-than-days",
            "30",
            "--dry-run",
            "--summary-out",
            str(summary_path),
        ]
    )
    assert rc == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["entries_deleted"] == 1
    assert old_entry.exists()
