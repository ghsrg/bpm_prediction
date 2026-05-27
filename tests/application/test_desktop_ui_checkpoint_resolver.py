from __future__ import annotations

import os
from pathlib import Path

from tools.desktop_ui.checkpoint_resolver import (
    DEFAULT_CHECKPOINT_VALUE,
    resolve_checkpoint_candidates,
)


def test_default_checkpoint_value_is_checkpoint_directory():
    assert DEFAULT_CHECKPOINT_VALUE == "checkpoints/"


def test_resolver_finds_checkpoint_by_experiment_name(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    expected = checkpoint_root / "S_Str_UNc_train_best.pth"
    expected.write_text("checkpoint", encoding="utf-8")
    (checkpoint_root / "Other_best.pth").write_text("checkpoint", encoding="utf-8")

    candidates = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="S_Str_UNc_train",
    )

    assert len(candidates) == 1
    assert candidates[0]["path"] == expected
    assert candidates[0]["filename"] == "S_Str_UNc_train_best.pth"


def test_resolver_returns_sorted_multiple_candidates(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    
    first = checkpoint_root / "RunA_first.pth"
    second = checkpoint_root / "RunA_second.pth"
    best = checkpoint_root / "RunA_best.pth"

    first.write_text("checkpoint", encoding="utf-8")
    second.write_text("checkpoint", encoding="utf-8")
    best.write_text("checkpoint", encoding="utf-8")

    # Set mtimes: second is newer than first, best has oldest mtime but should be prioritized
    os.utime(first, (1000, 1000))
    os.utime(second, (2000, 2000))
    os.utime(best, (500, 500))

    candidates = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
    )

    # Sort order expected: best, second, first
    assert [c["path"] for c in candidates] == [best, second, first]


def test_resolver_returns_empty_when_no_match(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "Other_best.pth").write_text("checkpoint", encoding="utf-8")

    assert resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
    ) == []


def test_resolver_filters_by_model_type(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    gatv2_path = checkpoint_root / "RunA_GATv2_best.pth"
    gcn_path = checkpoint_root / "RunA_GCN_best.pth"
    gatv2_path.write_text("checkpoint", encoding="utf-8")
    gcn_path.write_text("checkpoint", encoding="utf-8")

    # Filter for GATv2
    candidates_gatv2 = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
        model_type="GATv2",
    )
    assert len(candidates_gatv2) == 1
    assert candidates_gatv2[0]["path"] == gatv2_path

    # Filter for GCN
    candidates_gcn = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
        model_type="GCN",
    )
    assert len(candidates_gcn) == 1
    assert candidates_gcn[0]["path"] == gcn_path


def test_resolver_returns_correct_metadata(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    path = checkpoint_root / "RunA_best.pth"
    
    # Write exactly 1.5 MB of dummy data
    size_bytes = int(1.5 * 1024 * 1024)
    path.write_bytes(b"\0" * size_bytes)
    
    mtime = 1234567.0
    os.utime(path, (mtime, mtime))

    candidates = resolve_checkpoint_candidates(
        checkpoint_root=checkpoint_root,
        experiment_name="RunA",
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["path"] == path
    assert cand["filename"] == "RunA_best.pth"
    assert cand["size_mb"] == 1.5
    assert cand["mtime"] == mtime
    assert cand["date"] == mtime

